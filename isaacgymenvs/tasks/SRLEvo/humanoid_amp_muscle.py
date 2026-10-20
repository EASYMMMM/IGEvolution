"""Muscle-driven humanoid standing, straight walking and curved walking on a plane.

Policy observations retain the mclegs layout: 111 proprioceptive values
(including six load-cell slots) followed by three future local XY targets.
"""
from enum import Enum
import atexit
import csv
import os

import numpy as np
import torch
from gym import spaces
from isaacgym import gymtorch

from isaacgymenvs.tasks.SRLEvo.humanoid_amp_muscle_base import HumanoidAMPMuscleBase, dof_to_obs
from isaacgymenvs.tasks.SRLEvo.traj_generator import SimpleCurveGenerator as TrajGenerator
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.utils.torch_jit_utils import (
    quat_mul, to_torch, calc_heading_quat_inv, quat_to_tan_norm, my_quat_rotate,
)

NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12


class HumanoidAMPMuscleFlat(HumanoidAMPMuscleBase):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        self.cfg = cfg
        stage = cfg["env"].get("train_stage", 2)
        if stage not in (1, 2, 3, 4):
            raise ValueError(
                "HumanoidAMPMuscleFlat supports train_stage 1 (stand), 2 (straight), "
                "3 (curve), or 4 (mixed omni-walk)"
            )
        self._traj_sample_times = [0.5, 1.0, 1.5]
        self._num_traj_points = len(self._traj_sample_times)
        self._traj_obs_dim = self._num_traj_points * 2
        self._state_init = self.StateInit[cfg["env"]["stateInit"]]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._max_traj_distance = float(cfg["env"].get("maxTrajectoryDistance", 3.0))
        if self._max_traj_distance <= 0:
            raise ValueError("maxTrajectoryDistance must be positive")
        self.enable_log = cfg["env"].get("enableCsvLog", True)
        self.show_muscles = cfg["env"].get("enableMuscleVis", True)
        compliance_cfg = cfg["env"].get("compliance", {})
        self._compliance_reward_weights = tuple(float(compliance_cfg.get(key, default)) for key, default in (
            ("position_reward_weight", 1), ("velocity_reward_weight", 1),
            ("heading_reward_weight", 1), ("activation_cost_weight", 0),
        ))
        if not all(np.isfinite(w) and w >= 0 for w in self._compliance_reward_weights):
            raise ValueError("Compliance reward weights must be finite and nonnegative")

        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.forward_vec_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        episode_duration = self.max_episode_length * self.control_dt
        self._traj_gen = TrajGenerator(
            self.num_envs, self.device, self.control_dt, episode_duration,
            speed_mean=cfg["env"].get("speedMean", 1.0),
            turn_speed_max=cfg["env"].get("turnSpeedMax", 0.5),
            num_turns=cfg["env"].get("numTurns", 2),
            train_stage=self.train_stage, is_test=(not headless) or force_render,
        )
        motion_file = cfg["env"].get("motion_file", "fixed_smp_walk.npy")
        motion_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "../../../assets/amp/motions", motion_file)
        self._load_motion(motion_path)
        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP
        self._amp_obs_space = spaces.Box(-np.inf, np.inf, shape=(self.num_amp_obs,), dtype=np.float32)
        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP),
            device=self.device, dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        self._amp_obs_demo_buf = None
        if self.compliance.active:
            self._compliance_metric_sums = {key: torch.zeros(self.num_envs, device=self.device) for key in (
                "force_N", "reference_offset_m", "root_tracking_error_m", "nominal_root_error_m",
            )}
            self._compliance_peak_error = torch.zeros(self.num_envs, device=self.device)
            self._compliance_log_steps = torch.zeros(self.num_envs, device=self.device)
        if self.is_testing_mode and self.enable_log:
            self._init_csv_logger()
            atexit.register(self._close_csv_logger)
            print("[Logger] Muscle data: legs_kinematics.csv and legs_dynamics.csv")

    def get_obs_size(self):
        return super().get_obs_size() + self._traj_obs_dim

    def post_physics_step(self):
        super().post_physics_step()
        if self.is_testing_mode and self.enable_log:
            self._record_joints_to_csv()
        self._update_hist_amp_obs()
        self._compute_amp_observations()
        self.extras["amp_obs"] = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        if self.viewer and self.debug_viz:
            self._draw_debug_traj()
            if self.is_testing_mode and self.show_muscles:
                self._draw_muscles()
        if self.compliance.active:
            self._record_compliance_metrics()

    def _init_csv_logger(self):
        """初始化CSV文件和双腿表头"""
        self.kin_file = open("legs_kinematics.csv", mode='w', newline='')
        self.dyn_file = open("legs_dynamics.csv", mode='w', newline='')

        self.kin_writer = csv.writer(self.kin_file)
        self.dyn_writer = csv.writer(self.dyn_file)

        # 1. 运动学表头
        kin_header =[
            "time",
            "R_Hip_Y_deg", "R_Knee_deg", "R_Ankle_Y_deg",
            "L_Hip_Y_deg", "L_Knee_deg", "L_Ankle_Y_deg"
        ]
        self.kin_writer.writerow(kin_header)

        # 2. 动力学表头 (记录双腿共 14 根肌肉的激活度和拉力，以及双腿关节力矩)
        dyn_header = ["time"]
        muscles =["GLU", "HFL", "HAM", "VAS", "GAS", "SOL", "TA"]

        # 右腿肌肉表头
        for m in muscles:
            dyn_header +=[f"R_{m}_Act", f"R_{m}_Force_N"]

        # 左腿肌肉表头
        for m in muscles:
            dyn_header +=[f"L_{m}_Act", f"L_{m}_Force_N"]

        # 左右腿关节力矩表头
        dyn_header +=[
            "R_Hip_Y_Tau_Nm", "R_Knee_Tau_Nm", "R_Ankle_Y_Tau_Nm",
            "L_Hip_Y_Tau_Nm", "L_Knee_Tau_Nm", "L_Ankle_Y_Tau_Nm"
        ]
        self.dyn_writer.writerow(dyn_header)
        if self.compliance.active:
            self.compliance_file = open("muscle_compliance.csv", mode="w", newline="", encoding="utf-8")
            self.compliance_writer = csv.writer(self.compliance_file)
            self.compliance_writer.writerow([
                "episode_step", "time_s", "force_x_N", "force_y_N", "filtered_force_x_N", "filtered_force_y_N",
                "delta_x_m", "delta_y_m", "delta_vx_mps", "delta_vy_mps",
                "com_x_m", "com_y_m", "com_z_m", "com_vx_mps", "com_vy_mps", "com_vz_mps",
                "nominal_x_m", "nominal_y_m", "reference_x_m", "reference_y_m", "root_x_m", "root_y_m",
            ])


    def _draw_muscles(self):
        """可视化右腿的拮抗肌肉，蓝色表示放松，红色表示用力 (实心圆柱体版)"""
        if self.muscle_actuator.num_muscles == 56:
            return

        env_id = 0
        env_ptr = self.envs[env_id]

        if not hasattr(self, 'muscle_details'):
            return

        # 提取当前右腿 7 根肌肉的激活度 (索引 44 到 50)
        a = self.muscle_details['a'][env_id] # [58]
        a_right = a[44:51]

        pos_pelvis, rot_pelvis = self._rigid_body_pos[env_id, self.pelvis_bidx], self._rigid_body_rot[env_id, self.pelvis_bidx]
        pos_thigh, rot_thigh = self._rigid_body_pos[env_id, self.thigh_bidx], self._rigid_body_rot[env_id, self.thigh_bidx]
        pos_shin, rot_shin = self._rigid_body_pos[env_id, self.shin_bidx], self._rigid_body_rot[env_id, self.shin_bidx]
        pos_foot, rot_foot = self._rigid_body_pos[env_id, self.foot_bidx], self._rigid_body_rot[env_id, self.foot_bidx]

        def local_to_world(pos, rot, local_offset):
            offset_tensor = torch.tensor(local_offset, device=self.device, dtype=torch.float32).unsqueeze(0)
            world_offset = my_quat_rotate(rot.unsqueeze(0), offset_tensor).squeeze(0)
            return (pos + world_offset).cpu().numpy()

        lines = []
        colors =[]

        def get_color(act):
            act = max(0.0, min(1.0, act))
            return[act, 0.0, 1.0 - act]  # [R, G, B]

        def add_solid_cylinder(p1, p2, color, radius=0.012):
            p1_np = np.array(p1, dtype=np.float32)
            p2_np = np.array(p2, dtype=np.float32)
            d = p2_np - p1_np
            length = np.linalg.norm(d)
            if length < 1e-4: return
            d_norm = d / length

            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if np.abs(np.dot(d_norm, tmp)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            u = np.cross(d_norm, tmp)
            u /= np.linalg.norm(u)
            v = np.cross(d_norm, u)

            layers =[0.0, 0.33 * radius, 0.66 * radius, radius]
            lines_per_layer =[1, 6, 12, 18]

            for r, num_lines in zip(layers, lines_per_layer):
                if r == 0.0:
                    lines.append(np.concatenate([p1_np, p2_np]))
                    colors.append(color)
                else:
                    for i in range(num_lines):
                        theta = 2.0 * np.pi * i / num_lines
                        offset = r * np.cos(theta) * u + r * np.sin(theta) * v
                        lines.append(np.concatenate([p1_np + offset, p2_np + offset]))
                        colors.append(color)

        # 1. GLU: 骨盆前 -> 大腿后 (伸肌)
        p1 = local_to_world(pos_pelvis, rot_pelvis,[0.08, -0.09, 0.0])
        p2 = local_to_world(pos_thigh, rot_thigh,[0.06, 0.0, -0.15])
        add_solid_cylinder(p1, p2, get_color(a_right[0].item()))

        # 2. HFL: 骨盆后 -> 大腿前 (屈肌)
        p1 = local_to_world(pos_pelvis, rot_pelvis, [-0.08, -0.09, 0.0])
        p2 = local_to_world(pos_thigh, rot_thigh,[-0.06, 0.0, -0.15])
        add_solid_cylinder(p1, p2, get_color(a_right[1].item()))

        # 3. HAM (双关节): 骨盆后 -> 小腿后
        p1 = local_to_world(pos_pelvis, rot_pelvis,[-0.08, -0.09, 0.0])
        p2 = local_to_world(pos_shin, rot_shin,[-0.05, 0.0, -0.05])
        add_solid_cylinder(p1, p2, get_color(a_right[2].item()))

        # 4. VAS: 大腿前 -> 小腿前
        p1 = local_to_world(pos_thigh, rot_thigh,[0.05, 0.0, -0.38])
        p2 = local_to_world(pos_shin, rot_shin,[0.05, 0.0, -0.05])
        add_solid_cylinder(p1, p2, get_color(a_right[3].item()))

        # 5. GAS (双关节): 大腿后 -> 脚后跟
        p1 = local_to_world(pos_thigh, rot_thigh,[-0.05, 0.0, -0.38])
        p2 = local_to_world(pos_foot, rot_foot,[-0.04, 0.0, 0.02])
        add_solid_cylinder(p1, p2, get_color(a_right[4].item()))

        # 6. SOL: 小腿后 -> 脚后跟
        p1 = local_to_world(pos_shin, rot_shin,[-0.04, 0.0, -0.35])
        p2 = local_to_world(pos_foot, rot_foot,[-0.04, 0.0, 0.02])
        add_solid_cylinder(p1, p2, get_color(a_right[5].item()))

        # 7. TA: 小腿前 -> 脚背
        p1 = local_to_world(pos_shin, rot_shin,[0.04, 0.0, -0.35])
        p2 = local_to_world(pos_foot, rot_foot, [0.06, 0.0, 0.02])
        add_solid_cylinder(p1, p2, get_color(a_right[6].item()))

        if len(lines) > 0:
            lines_np = np.array(lines, dtype=np.float32)
            colors_np = np.array(colors, dtype=np.float32)
            self.gym.add_lines(self.viewer, env_ptr, len(lines_np), lines_np, colors_np)


    def _record_joints_to_csv(self):
        """每步记录第0个环境的双腿数据"""
        if self.muscle_actuator.num_muscles == 56:
            return

        env_id = 0
        current_time = self.progress_buf[env_id].item() * self.control_dt

        # --- 运动学数据采集 ---
        r_hip_ang = np.rad2deg(self._dof_pos[env_id, self.r_hip_y_idx].item())
        r_knee_ang = np.rad2deg(self._dof_pos[env_id, self.r_knee_idx].item())
        r_ankle_ang = np.rad2deg(self._dof_pos[env_id, self.r_ankle_y_idx].item())

        l_hip_ang = np.rad2deg(self._dof_pos[env_id, self.l_hip_y_idx].item())
        l_knee_ang = np.rad2deg(self._dof_pos[env_id, self.l_knee_idx].item())
        l_ankle_ang = np.rad2deg(self._dof_pos[env_id, self.l_ankle_y_idx].item())

        kin_row =[
            f"{current_time:.4f}",
            f"{r_hip_ang:.4f}", f"{r_knee_ang:.4f}", f"{r_ankle_ang:.4f}",
            f"{l_hip_ang:.4f}", f"{l_knee_ang:.4f}", f"{l_ankle_ang:.4f}"
        ]
        self.kin_writer.writerow(kin_row)

        # --- 动力学数据采集 ---
        dyn_row =[f"{current_time:.4f}"]

        # 提取右腿 7 根肌肉的状态 (索引44-50)
        a_right = self.muscle_details['a'][env_id][44:51]
        F_right = self.muscle_details['F'][env_id][44:51]
        for i in range(7):
            dyn_row +=[f"{a_right[i].item():.4f}", f"{F_right[i].item():.4f}"]

        # 提取左腿 7 根肌肉的状态 (索引51-57)
        a_left = self.muscle_details['a'][env_id][51:58]
        F_left = self.muscle_details['F'][env_id][51:58]
        for i in range(7):
            dyn_row += [f"{a_left[i].item():.4f}", f"{F_left[i].item():.4f}"]

        # 提取双腿关节最终合力矩
        r_hip_tau = self.applied_tau[env_id, self.r_hip_y_idx].item()
        r_knee_tau = self.applied_tau[env_id, self.r_knee_idx].item()
        r_ankle_tau = self.applied_tau[env_id, self.r_ankle_y_idx].item()

        l_hip_tau = self.applied_tau[env_id, self.l_hip_y_idx].item()
        l_knee_tau = self.applied_tau[env_id, self.l_knee_idx].item()
        l_ankle_tau = self.applied_tau[env_id, self.l_ankle_y_idx].item()

        dyn_row +=[
            f"{r_hip_tau:.4f}", f"{r_knee_tau:.4f}", f"{r_ankle_tau:.4f}",
            f"{l_hip_tau:.4f}", f"{l_knee_tau:.4f}", f"{l_ankle_tau:.4f}"
        ]
        self.dyn_writer.writerow(dyn_row)

        # 定期刷新到磁盘
        if self.progress_buf[env_id] % 100 == 0:
            self.kin_file.flush()
            self.dyn_file.flush()


    def _draw_debug_traj(self):
        self.gym.clear_lines(self.viewer)
        num_debug_envs = min(self.num_envs, 5)
        for i in range(num_debug_envs):
            traj = self._traj_gen._traj_pos[i]
            points = traj[::10].cpu().numpy()
            lines = np.zeros((len(points)-1, 6), dtype=np.float32)
            for j in range(len(points)-1):
                lines[j, :3] = points[j]
                lines[j, 3:] = points[j+1]
            colors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32).repeat(len(lines), axis=0)
            self.gym.add_lines(self.viewer, self.envs[i], len(lines), lines, colors)
            if self.compliance.active:
                ids = torch.tensor([i], device=self.device, dtype=torch.long)
                time = self.progress_buf[ids] * self.control_dt
                reference = self._get_reference(ids, time)
                current = reference["position"][0].cpu().numpy()
                future = reference["future"][0].cpu().numpy()
                preview = np.concatenate((current[None, :], future), axis=0)
                preview[:, 2] = 0.08
                green_lines = np.concatenate((preview[:-1], preview[1:]), axis=-1).astype(np.float32)
                green = np.tile(np.array([[0, 1, 0]], dtype=np.float32), (len(green_lines), 1))
                self.gym.add_lines(self.viewer, self.envs[i], len(green_lines), green_lines, green)
                com = self.human_com[i].cpu().numpy()
                force = self._control_external_force[i].cpu().numpy() * 0.005  # 100 N -> 0.5 m
                marker = np.stack([np.concatenate((com - np.eye(3)[axis] * 0.04, com + np.eye(3)[axis] * 0.04)) for axis in range(3)])
                self.gym.add_lines(self.viewer, self.envs[i], 3, marker.astype(np.float32), np.tile([[0, 1, 1]], (3, 1)).astype(np.float32))
                if np.linalg.norm(force) > 1e-6:
                    tip = com + force
                    side = np.array([-force[1], force[0], 0], dtype=np.float32) * 0.15
                    arrow = np.stack((np.concatenate((com, tip)), np.concatenate((tip, tip - force * 0.2 + side)),
                                      np.concatenate((tip, tip - force * 0.2 - side)))).astype(np.float32)
                    self.gym.add_lines(self.viewer, self.envs[i], 3, arrow, np.tile([[1, 1, 0]], (3, 1)).astype(np.float32))


    def _close_csv_logger(self):
        for name in ("kin_file", "dyn_file", "compliance_file"):
            handle = getattr(self, name, None)
            if handle is not None and not handle.closed:
                handle.close()

    def _compute_reward(self, actions):
        if self.compliance.enabled:
            self._compute_compliant_reward()
            return
        current_time = self.progress_buf * self.control_dt
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        root_pos = self._root_states[..., 0:3]
        root_rot = self._root_states[..., 3:7]
        future_traj_points = self._traj_gen.get_observation_points(env_ids, current_time, self._traj_sample_times)
        target_pos_05s = future_traj_points[:, 0, 0:2]
        target_dist = torch.norm(target_pos_05s - target_pos[..., 0:2], dim=-1)
        implied_target_speed = target_dist / 0.5
        dist_sq = torch.sum((target_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
        pos_reward = torch.exp(-2.0 * dist_sq)
        vel_dir = target_pos[..., 0:2] - root_pos[..., 0:2]
        vel_dir = torch.nn.functional.normalize(vel_dir, dim=-1)
        root_vel = self._root_states[..., 7:9]
        current_vel_proj = torch.sum(root_vel * vel_dir, dim=-1)
        vel_reward = torch.exp(-2.0 * (current_vel_proj - implied_target_speed)**2)
        forward_vec_world = my_quat_rotate(root_rot, self.forward_vec_local)
        heading_alignment = torch.sum(forward_vec_world[..., 0:2] * vel_dir, dim=-1)
        heading_reward = torch.clamp(heading_alignment, min=0, max=1.0)

        if self.train_stage == 1:
            root_vel_xyz = self._root_states[..., 7:10]
            vel_penalty = torch.sum(root_vel_xyz[:, 0:2] ** 2, dim=-1)
            root_ang_vel = self._root_states[..., 10:13]
            ang_penalty = root_ang_vel[:, 2] ** 2
            dof_diff = self._dof_pos - self._initial_dof_pos
            joint_penalty = torch.sum(dof_diff ** 2, dim=-1)
            act_penalty = torch.sum(actions ** 2, dim=-1)
            self.rew_buf[:] = (pos_reward * torch.exp(-2.0 * vel_penalty)
                               * torch.exp(-ang_penalty) * torch.exp(-joint_penalty)
                               * torch.exp(-0.001 * act_penalty))
            self.extras["amp_mask"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        else:
            self.rew_buf[:] = pos_reward + vel_reward + heading_reward
            self.extras["amp_mask"] = torch.ones(self.num_envs, device=self.device, dtype=torch.float)

    def _get_reference(self, env_ids, time):
        """Single reference source for reward, observation, termination and drawing."""
        nominal = self._traj_gen.get_position(env_ids, time)
        nominal_future = self._traj_gen.get_observation_points(env_ids, time, self._traj_sample_times)
        position, future = nominal.clone(), nominal_future.clone()
        nominal_velocity = (nominal_future[:, 0, :2] - nominal[:, :2]) / self._traj_sample_times[0]
        velocity = nominal_velocity.clone()
        if self.compliance.enabled:
            position[:, :2] += self.compliance.delta[env_ids]
            future[:, :, :2] += self.compliance.forecast(env_ids, self._traj_sample_times)
            velocity += self.compliance.velocity[env_ids]
        return {"nominal": nominal, "position": position, "future": future,
                "velocity": velocity, "nominal_velocity": nominal_velocity}

    def _compute_compliant_reward(self):
        ids = torch.arange(self.num_envs, device=self.device)
        reference = self._get_reference(ids, self.progress_buf * self.control_dt)
        error = reference["position"][:, :2] - self._root_states[:, :2]
        pos_reward = torch.exp(-2 * torch.sum(error ** 2, dim=-1))
        velocity_error = reference["velocity"] - self._root_states[:, 7:9]
        vel_reward = torch.exp(-2 * torch.sum(velocity_error ** 2, dim=-1))
        nominal_velocity = reference["nominal_velocity"]
        speed = torch.linalg.vector_norm(nominal_velocity, dim=-1)
        heading = nominal_velocity / speed[:, None].clamp_min(1e-6)
        forward = my_quat_rotate(self._root_states[:, 3:7], self.forward_vec_local)
        heading_reward = torch.clamp(torch.sum(forward[:, :2] * heading, dim=-1), 0, 1)
        heading_reward = torch.where(speed > 1e-6, heading_reward, torch.ones_like(heading_reward))
        activation_cost = torch.mean(self.muscle_actuator.a ** 2, dim=-1)
        wp, wv, wh, wa = self._compliance_reward_weights
        if self.train_stage == 1:
            # Retain standing posture and yaw stabilization, allow yielding velocity.
            angular_penalty = self._root_states[:, 12] ** 2
            joint_penalty = torch.sum((self._dof_pos - self._initial_dof_pos) ** 2, dim=-1)
            self.rew_buf[:] = pos_reward * vel_reward * torch.exp(-angular_penalty - joint_penalty - wa * activation_cost)
            self.extras["amp_mask"] = torch.zeros(self.num_envs, device=self.device)
        else:
            self.rew_buf[:] = wp * pos_reward + wv * vel_reward + wh * heading_reward - wa * activation_cost
            self.extras["amp_mask"] = torch.ones(self.num_envs, device=self.device)
        self.extras["compliance_rewards"] = {
            "position_mean": pos_reward.mean(), "velocity_mean": vel_reward.mean(),
            "heading_mean": heading_reward.mean(), "activation_cost_mean": activation_cost.mean(),
        }

    def _record_compliance_metrics(self):
        ids = torch.arange(self.num_envs, device=self.device)
        reference = self._get_reference(ids, self.progress_buf * self.control_dt)
        metrics = {
            "force_N": torch.linalg.vector_norm(self._control_external_force[:, :2], dim=-1),
            "reference_offset_m": torch.linalg.vector_norm(self.compliance.delta, dim=-1),
            "root_tracking_error_m": torch.linalg.vector_norm(reference["position"][:, :2] - self._root_states[:, :2], dim=-1),
            "nominal_root_error_m": torch.linalg.vector_norm(reference["nominal"][:, :2] - self._root_states[:, :2], dim=-1),
        }
        self._compliance_log_steps += 1
        for key, values in metrics.items():
            self._compliance_metric_sums[key] += values
        self._compliance_peak_error = torch.maximum(self._compliance_peak_error, metrics["root_tracking_error_m"])
        self.extras["compliance"] = {key + "_step_mean": value.mean() for key, value in metrics.items()}
        self.extras["compliance"].update({
            "event_count_step_mean": self.compliance.event_count.mean(),
            "early_termination_step_fraction": self._terminate_buf.float().mean(),
            "com_speed_mps_step_mean": torch.linalg.vector_norm(self.human_com_velocity[:, :2], dim=-1).mean(),
            "human_mass_kg_mean": self._human_mass.mean(),
        })
        self.extras.pop("episode", None)
        done = torch.nonzero(self.reset_buf, as_tuple=False).flatten()
        if len(done):
            count = self._compliance_log_steps[done].clamp_min(1)
            self.extras["episode"] = {"compliance/" + key + "_episode_mean": values[done] / count
                                      for key, values in self._compliance_metric_sums.items()}
            self.extras["episode"].update({
                "compliance/root_tracking_error_peak_m": self._compliance_peak_error[done],
                "compliance/early_termination": self._terminate_buf[done].float(),
                "compliance/force_event_count": self.compliance.event_count[done],
            })
        if hasattr(self, "compliance_writer"):
            i = 0
            row = [self.progress_buf[i].item(), self.progress_buf[i].item() * self.control_dt]
            for vector in (self._control_external_force[i, :2], self.compliance.filtered_force[i],
                           self.compliance.delta[i], self.compliance.velocity[i], self.human_com[i],
                           self.human_com_velocity[i], reference["nominal"][i, :2], reference["position"][i, :2],
                           self._root_states[i, :2]):
                row.extend(vector.detach().cpu().tolist())
            self.compliance_writer.writerow(row)
            if self.progress_buf[i] % 100 == 0:
                self.compliance_file.flush()

    def _compute_reset(self):
        super()._compute_reset()
        current_time = self.progress_buf * self.control_dt
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        if getattr(self, "compliance", None) is not None and self.compliance.enabled:
            target_pos = self._get_reference(env_ids, current_time)["position"]
        dist_sq = torch.sum((target_pos[:, :2] - self._root_states[:, :2]) ** 2, dim=-1)
        has_strayed = (dist_sq > self._max_traj_distance ** 2) & (self.progress_buf > 10)
        strayed_tensor = has_strayed.to(dtype=torch.long)
        self.reset_buf[:] |= strayed_tensor
        self._terminate_buf[:] |= strayed_tensor

    def get_num_amp_obs(self):
        return self.num_amp_obs


    @property
    def amp_observation_space(self):
        return self._amp_obs_space


    def fetch_amp_obs_demo(self, num_samples):
        dt = self.control_dt
        motion_ids = self._motion_lib.sample_motions(num_samples)
        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1),[1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps
        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(motion_ids, motion_times)

        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos, self._local_root_obs)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
        return self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())


    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)


    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(motion_file=motion_file, num_dofs=self.num_dof, key_body_ids=self._key_body_ids.cpu().numpy(), device=self.device)


    def reset_idx(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        root_pos = self._root_states[env_ids, :3]
        root_rot = self._root_states[env_ids, 3:7]
        self._traj_gen.reset(env_ids, root_pos, init_rot=root_rot)
        if getattr(self, "compliance", None) is not None:
            self._reset_compliance(env_ids)
            if self.compliance.active:
                for buffer in self._compliance_metric_sums.values():
                    buffer[env_ids] = 0
                self._compliance_peak_error[env_ids] = 0
                self._compliance_log_steps[env_ids] = 0
        self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)

    def _compute_humanoid_obs(self, env_ids=None):
        base_obs = super()._compute_humanoid_obs(env_ids)
        if env_ids is None:
            env_ids_flat = torch.arange(self.num_envs, device=self.device)
            current_time = self.progress_buf * self.control_dt
            root_pos = self._root_states[:, 0:3]
            root_rot = self._root_states[:, 3:7]
        else:
            env_ids_flat = env_ids
            current_time = self.progress_buf[env_ids] * self.control_dt
            root_pos = self._root_states[env_ids, 0:3]
            root_rot = self._root_states[env_ids, 3:7]
        traj_points_world = self._traj_gen.get_observation_points(env_ids_flat, current_time, self._traj_sample_times)
        if self.compliance.enabled:
            traj_points_world = self._get_reference(env_ids_flat, current_time)["future"]
        heading_inv = calc_heading_quat_inv(root_rot)
        num_samples = self._num_traj_points
        heading_inv_expand = heading_inv.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 4)
        delta_pos = traj_points_world - root_pos.unsqueeze(1)
        delta_pos_flat = delta_pos.reshape(-1, 3)
        local_traj_points = my_quat_rotate(heading_inv_expand, delta_pos_flat)
        local_traj_points = local_traj_points.view(root_pos.shape[0], num_samples, 3)
        traj_obs = local_traj_points[..., 0:2].reshape(root_pos.shape[0], -1)
        # 【修改】：拼接本体观测和轨迹观测
        obs = torch.cat([base_obs, traj_obs], dim=-1)
        return obs

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAMPMuscleFlat.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMPMuscleFlat.StateInit.Start or self._state_init == HumanoidAMPMuscleFlat.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAMPMuscleFlat.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        self.muscle_actuator.a[env_ids] = 0.0

    def _reset_default(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._initial_root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self._reset_default_env_ids = env_ids


    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        if (self._state_init == HumanoidAMPMuscleFlat.StateInit.Random or self._state_init == HumanoidAMPMuscleFlat.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        else:
            motion_times = np.zeros(num_envs)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(motion_ids, motion_times)
        # Place reference poses at each environment's own spawn position.
        root_pos[:, 0] = self._initial_root_states[env_ids, 0]
        root_pos[:, 1] = self._initial_root_states[env_ids, 1]
        # root_pos[:,2] += 0.14
        self._set_env_state(env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel)
        self._reset_ref_env_ids, self._reset_ref_motion_ids, self._reset_ref_motion_times = env_ids, motion_ids, motion_times


    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0
        if len(env_ids[ref_init_mask]) > 0: self._reset_ref_state_init(env_ids[ref_init_mask])
        if len(env_ids[~ref_init_mask]) > 0: self._reset_default(env_ids[~ref_init_mask])


    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)
        if len(self._reset_default_env_ids) > 0: self._init_amp_obs_default(self._reset_default_env_ids)
        if len(self._reset_ref_env_ids) > 0: self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids, self._reset_ref_motion_times)


    def _init_amp_obs_default(self, env_ids):
        self._hist_amp_obs_buf[env_ids] = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)


    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.control_dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1),[1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1) + (-dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1))
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(motion_ids.flatten(), motion_times.flatten())

        amp_obs_demo = build_amp_observations(torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1), dof_pos, dof_vel, key_pos, self._local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)


    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._root_states[env_ids, 0:3], self._root_states[env_ids, 3:7] = root_pos, root_rot
        self._root_states[env_ids, 7:10], self._root_states[env_ids, 10:13] = root_vel, root_ang_vel
        self._dof_pos[env_ids], self._dof_vel[env_ids] = dof_pos, dof_vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)): self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)): self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]


    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if env_ids is None:
            self._curr_amp_obs_buf[:] = build_amp_observations(self._root_states, self._dof_pos, self._dof_vel, key_body_pos, self._local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._root_states[env_ids], self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids], self._local_root_obs)


@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos, root_rot = root_states[:, 0:3], root_states[:, 3:7]
    root_vel, root_ang_vel = root_states[:, 7:10], root_states[:, 10:13]
    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)
    root_rot_obs = quat_to_tan_norm(quat_mul(heading_rot, root_rot) if local_root_obs else root_rot)
    local_root_vel, local_root_ang_vel = my_quat_rotate(heading_rot, root_vel), my_quat_rotate(heading_rot, root_ang_vel)

    local_key_body_pos = key_body_pos - root_pos.unsqueeze(-2)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = my_quat_rotate(heading_rot_expand.view(-1, 4), local_key_body_pos.view(-1, 3))
    flat_local_key_pos = flat_end_pos.view(local_key_body_pos.shape[0], -1)

    dof_obs = dof_to_obs(dof_pos)
    return torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
