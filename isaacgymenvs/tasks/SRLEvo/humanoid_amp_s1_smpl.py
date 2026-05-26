'''
SRL-Gym
分段训练S1: 仅优化Humanoid行走, 观测空间中添加了6D的交互力传感器
+ 修改: 随机曲线轨迹跟随任务 (TrajGenerator)
+ 修改: 加入偏离重置 (Early Termination based on distance)
'''
from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from isaacgymenvs.tasks.SRLEvo.humanoid_amp_s1_smpl_base import HumanoidAMP_s1_Smpl_Base, dof_to_obs
from isaacgymenvs.tasks.amp.utils_amp import gym_util
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib

from isaacgymenvs.tasks.SRLEvo.traj_generator import SimpleCurveGenerator as TrajGenerator
#from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, calc_heading_quat_inv, quat_to_tan_norm, my_quat_rotate
from isaacgymenvs.utils.torch_jit_utils import quat_mul, quat_conjugate, quat_to_exp_map, to_torch, calc_heading_quat_inv, quat_to_tan_norm, my_quat_rotate

print("[IMPORT CHECK] humanoid_amp_s1_smpl loaded from:", __file__)

NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
LOWER_BODY_DOF_GROUPS = {
    "r_hip": [14, 15, 16],
    "r_knee": [17],
    "r_ankle": [18, 19, 20],
    "l_hip": [21, 22, 23],
    "l_knee": [24],
    "l_ankle": [25, 26, 27],
}


class HumanoidAMP_s1_Smpl(HumanoidAMP_s1_Smpl_Base):

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self._traj_sample_times = [0.5, 1.0, 1.5] # 轨迹参数
        self._num_traj_points = len(self._traj_sample_times)

        # command obs = 3个未来目标点(local xy, 6维)
        #             + cmd_speed(1维)
        #             + standness(1维)
        #             + walk yaw error 的 cos/sin(2维)
        #             + stand yaw error 的 cos/sin(2维)
        self._traj_obs_dim = self._num_traj_points * 2 + 1 + 1 + 2 + 2
        
        self.cfg = cfg
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMP_s1_Smpl.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)
        self._num_state_hist_steps = cfg["env"].get("numStateHistSteps", 5)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

        self._base_obs_dim = super().get_obs_size()
        self._state_hist_obs_buf = torch.zeros(
            (self.num_envs, self._num_state_hist_steps, self._base_obs_dim),
            device=self.device,
            dtype=torch.float
        )

        self._pelvis_body_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "pelvis"
        )
        self._head_body_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "head"
        )
        self._left_foot_body_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "left_foot"
        )
        self._right_foot_body_id = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], "right_foot"
        )

        self._diag_compare_3dof = self.cfg["env"].get("diagCompare3Dof", False)
        self._diag_compare_interval = self.cfg["env"].get("diagCompare3DofInterval", 30)
        self._diag_compare_env = self.cfg["env"].get("diagCompare3DofEnv", 0)
        self._diag_compare_base_obs = self.cfg["env"].get("diagCompareBaseObs", False)
        self._diag_compare_cmd_obs = self.cfg["env"].get("diagCompareCmdObs", False)
        self._diag_print_dof_torque = self.cfg["env"].get("diagPrintDofTorque", False)
        self._diag_torque_interval = self.cfg["env"].get("diagDofTorqueInterval", 10)
        self._diag_print_action_target = self.cfg["env"].get("diagPrintActionTarget", False)
        self._diag_action_target_interval = self.cfg["env"].get("diagActionTargetInterval", 10)
        self._diag_print_native_pd_compare = self.cfg["env"].get("diagPrintNativePdCompare", False)
        self._diag_native_pd_interval = self.cfg["env"].get("diagNativePdInterval", 30)

        self._export_reset_state_db = self.cfg["env"].get("exportResetStateDb", False)
        self._export_reset_state_db_path = self.cfg["env"].get("exportResetStateDbPath", "humanoid_reset_states.npz")
        self._export_reset_state_db_max_samples = int(self.cfg["env"].get("exportResetStateDbMaxSamples", 5000))
        self._export_reset_state_db_num_per_reset = int(self.cfg["env"].get("exportResetStateDbNumPerReset", 32))
        self._export_reset_state_db_flush_every = int(self.cfg["env"].get("exportResetStateDbFlushEvery", 256))
        self._export_reset_state_db_saved_once = False
        self._reset_state_db = {
            "root_pos": [],
            "root_rot": [],
            "dof_pos": [],
            "root_vel": [],
            "root_ang_vel": [],
            "dof_vel": [],
            "source_mode": [],
        }


        self._diag_body_ids = {
            "pelvis": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "pelvis"),
            "torso": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "torso"),
            "head": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "head"),
            "right_upper_arm": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "right_upper_arm"),
            "left_upper_arm": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "left_upper_arm"),
            "right_thigh": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "right_thigh"),
            "left_thigh": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "left_thigh"),
            "right_shin": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "right_shin"),
            "left_shin": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "left_shin"),
            "right_foot": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "right_foot"),
            "left_foot": self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "left_foot"),
        }

        # standing phase 朝向缓存
        # 保存“进入 standing phase 时”的世界系 XY 朝向单位向量
        self._stand_heading_dir = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self._prev_is_standing_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._standing_phase_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # standing / walking 切换回差阈值
        self._stand_enter_thres = 0.01   # 进入站立：1 cm
        self._stand_exit_thres  = 0.03   # 退出站立：3 cm

        # standing 位置容差
        self._stand_pos_tol = 0.08       # 8 cm 内都算“站到位”

        # 确定当前是否为测试模式 (用于控制 Stage 4 的日志打印)
        is_testing = (not headless) or force_render
        
        # 轨迹生成器
        self.forward_vec_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        episode_duration = self.max_episode_length * 1.2 * self.control_dt
        self._traj_gen = TrajGenerator(
            self.num_envs, self.device, self.control_dt, episode_duration,
            speed_mean=1.0,
            turn_speed_max=0.5,
            num_turns=2,
            train_stage=self.train_stage,
            is_test=is_testing
        )

        motion_file = cfg['env'].get('motion_file', "amp_humanoid_backflip.npy")
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets/amp/motions/" + motion_file)
        self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP
        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)
        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        self._amp_obs_demo_buf = None
        return

    def get_obs_size(self):
        return super().get_obs_size() * self._num_state_hist_steps + self._traj_obs_dim

    def _update_hist_state_obs(self, env_ids=None):
        if self._num_state_hist_steps <= 1:
            return

        if env_ids is None:
            for i in reversed(range(self._num_state_hist_steps - 1)):
                self._state_hist_obs_buf[:, i + 1] = self._state_hist_obs_buf[:, i]
        else:
            for i in reversed(range(self._num_state_hist_steps - 1)):
                self._state_hist_obs_buf[env_ids, i + 1] = self._state_hist_obs_buf[env_ids, i]
        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        if self.viewer and self.debug_viz:
            self._draw_debug_traj()

        if self._diag_compare_3dof:
            self._print_3dof_alignment_diag()
        if self._diag_compare_base_obs:
            self._print_base_obs_alignment_diag()
        if self._diag_compare_cmd_obs:
            self._print_cmd_obs_alignment_diag()
        if self._diag_print_dof_torque:
            self._print_dof_torque_diag()
        if self._diag_print_action_target:
            self._print_action_target_diag()
        if self._diag_print_native_pd_compare:
            self._print_native_pd_compare_diag()
        return    

    def _relative_quat(self, env_id, parent_name, child_name):
        parent_id = self._diag_body_ids[parent_name]
        child_id = self._diag_body_ids[child_name]
        q_parent = self._rigid_body_rot[env_id, parent_id]
        q_child = self._rigid_body_rot[env_id, child_id]
        return quat_mul(quat_conjugate(q_parent), q_child)

    def _relative_ang_vel_local(self, env_id, parent_name, child_name):
        parent_id = self._diag_body_ids[parent_name]
        child_id = self._diag_body_ids[child_name]
        w_parent = self._rigid_body_ang_vel[env_id, parent_id]
        w_child = self._rigid_body_ang_vel[env_id, child_id]
        w_rel_world = w_child - w_parent
        q_parent = self._rigid_body_rot[env_id, parent_id]
        return my_quat_rotate(
            quat_conjugate(q_parent).unsqueeze(0),
            w_rel_world.unsqueeze(0)
        )[0]

    def _relative_ang_vel_world(self, env_id, parent_name, child_name):
        parent_id = self._diag_body_ids[parent_name]
        child_id = self._diag_body_ids[child_name]
        w_parent = self._rigid_body_ang_vel[env_id, parent_id]
        w_child = self._rigid_body_ang_vel[env_id, child_id]
        return w_child - w_parent

    def _relative_ang_vel_child_local(self, env_id, parent_name, child_name):
        child_id = self._diag_body_ids[child_name]
        w_rel_world = self._relative_ang_vel_world(env_id, parent_name, child_name)
        q_child = self._rigid_body_rot[env_id, child_id]
        return my_quat_rotate(
            quat_conjugate(q_child).unsqueeze(0),
            w_rel_world.unsqueeze(0)
        )[0]

    def _relative_ang_vel_joint_local(self, env_id, parent_name, child_name):
        w_rel_world = self._relative_ang_vel_world(env_id, parent_name, child_name)
        q_rel = self._relative_quat(env_id, parent_name, child_name)
        return my_quat_rotate(
            quat_conjugate(q_rel).unsqueeze(0),
            w_rel_world.unsqueeze(0)
        )[0]


    def _print_3dof_alignment_diag(self):
        env_id = int(self._diag_compare_env)
        if env_id < 0 or env_id >= self.num_envs:
            return

        step = int(self.progress_buf[env_id].item())
        if step % self._diag_compare_interval != 0:
            return

        native_dof_pos = self._dof_pos[env_id]
        native_dof_vel = self._dof_vel[env_id]

        compare_specs = [
            ("abdomen", 0, 3, "pelvis", "torso"),
            ("neck", 3, 6, "torso", "head"),
            ("right_shoulder", 6, 9, "torso", "right_upper_arm"),
            ("left_shoulder", 10, 13, "torso", "left_upper_arm"),
            ("right_hip", 14, 17, "pelvis", "right_thigh"),
            ("right_ankle", 18, 21, "right_shin", "right_foot"),
            ("left_hip", 21, 24, "pelvis", "left_thigh"),
            ("left_ankle", 25, 28, "left_shin", "left_foot"),
        ]

        print("\n" + "=" * 80)
        print(f"[3DoF Align Diag] env={env_id} step={step} train_stage={self.train_stage}")
        print("=" * 80)
        for name, start, end, parent_name, child_name in compare_specs:
            native_pos = native_dof_pos[start:end].detach().cpu().numpy()
            native_vel = native_dof_vel[start:end].detach().cpu().numpy()

            recon_pos = quat_to_exp_map(
                self._relative_quat(env_id, parent_name, child_name).unsqueeze(0)
            )[0].detach().cpu().numpy()
            recon_vel_parent = self._relative_ang_vel_local(env_id, parent_name, child_name).detach().cpu().numpy()
            recon_vel_world = self._relative_ang_vel_world(env_id, parent_name, child_name).detach().cpu().numpy()
            recon_vel_child = self._relative_ang_vel_child_local(env_id, parent_name, child_name).detach().cpu().numpy()
            recon_vel_joint = self._relative_ang_vel_joint_local(env_id, parent_name, child_name).detach().cpu().numpy()

            print(f"{name}:")
            print(f"  native pos = {np.array2string(native_pos, precision=4, suppress_small=True)}")
            print(f"  recon  pos = {np.array2string(recon_pos, precision=4, suppress_small=True)}")
            print(f"  native vel = {np.array2string(native_vel, precision=4, suppress_small=True)}")
            print(f"  recon vel parent_local = {np.array2string(recon_vel_parent, precision=4, suppress_small=True)}")
            print(f"  recon vel child_local  = {np.array2string(recon_vel_child, precision=4, suppress_small=True)}")
            print(f"  recon vel joint_local  = {np.array2string(recon_vel_joint, precision=4, suppress_small=True)}")
            print(f"  recon vel world        = {np.array2string(recon_vel_world, precision=4, suppress_small=True)}")

    def _print_dof_torque_diag(self):
        env_id = int(self._diag_compare_env)
        if env_id < 0 or env_id >= self.num_envs:
            return

        step = int(self.progress_buf[env_id].item())
        if step % self._diag_torque_interval != 0:
            return

        dof_force = self.dof_force_tensor[env_id].detach().cpu().numpy()
        motor_effort = self.motor_efforts.detach().cpu().numpy()

        print("=" * 80)
        print(f"[DOF Torque Diag] env={env_id} step={step} train_stage={self.train_stage}")
        print("=" * 80)
        for name, idxs in LOWER_BODY_DOF_GROUPS.items():
            idxs_np = np.asarray(idxs, dtype=np.int32)
            group_force = dof_force[idxs_np]
            group_effort = np.maximum(np.abs(motor_effort[idxs_np]), 1e-8)
            group_ratio = np.abs(group_force) / group_effort
            print(
                f"{name}: "
                f"force={np.array2string(group_force, precision=4, suppress_small=True)} | "
                f"|force|/effort={np.array2string(group_ratio, precision=4, suppress_small=True)} | "
                f"max_ratio={group_ratio.max():.4f} mean_ratio={group_ratio.mean():.4f}"
            )
    def _print_action_target_diag(self):
        env_id = int(self._diag_compare_env)
        if env_id < 0 or env_id >= self.num_envs:
            return

        step = int(self.progress_buf[env_id].item())
        if step % self._diag_action_target_interval != 0:
            return

        raw_action = self.actions[env_id].detach().cpu().numpy()
        if self._pd_control:
            target_pos = self._action_to_pd_targets(self.actions[env_id:env_id + 1])[0].detach().cpu().numpy()
        else:
            target_pos = raw_action.copy()

        print("=" * 80)
        print(f"[Action Target Diag] env={env_id} step={step} train_stage={self.train_stage}")
        print("=" * 80)
        print(
            "raw_action = "
            + np.array2string(raw_action, precision=4, suppress_small=True, max_line_width=100000)
        )
        print(
            "target_pos = "
            + np.array2string(target_pos, precision=4, suppress_small=True, max_line_width=100000)
        )
        print(
            "raw_action stats: "
            f"min={raw_action.min():.4f} max={raw_action.max():.4f} "
            f"mean={raw_action.mean():.4f} norm={np.linalg.norm(raw_action):.4f}"
        )
        print(
            "target_pos stats: "
            f"min={target_pos.min():.4f} max={target_pos.max():.4f} "
            f"mean={target_pos.mean():.4f} norm={np.linalg.norm(target_pos):.4f}"
        )
    def _print_native_pd_compare_diag(self):
        if not self._pd_control:
            return

        env_id = int(self._diag_compare_env)
        if env_id < 0 or env_id >= self.num_envs:
            return

        step = int(self.progress_buf[env_id].item())
        if step % self._diag_native_pd_interval != 0:
            return

        native_dof_pos = self._dof_pos[env_id]
        native_dof_vel = self._dof_vel[env_id]
        target_pos = self._action_to_pd_targets(self.actions[env_id:env_id + 1])[0]
        dof_force = self.dof_force_tensor[env_id]
        pd_stiffness = self._pd_stiffness
        pd_damping = self._pd_damping

        compare_specs = [
            ("abdomen", 0, 3),
            ("neck", 3, 6),
            ("right_shoulder", 6, 9),
            ("left_shoulder", 10, 13),
            ("right_hip", 14, 17),
            ("right_knee", 17, 18),
            ("right_ankle", 18, 21),
            ("left_hip", 21, 24),
            ("left_knee", 24, 25),
            ("left_ankle", 25, 28),
        ]

        print("=" * 80)
        print(f"[Native PD Compare] env={env_id} step={step} train_stage={self.train_stage}")
        print("=" * 80)
        for name, start, end in compare_specs:
            q_native = native_dof_pos[start:end]
            qd_native = native_dof_vel[start:end]
            q_target = target_pos[start:end]
            tau_measured = dof_force[start:end]
            tau_pred = pd_stiffness[start:end] * (q_target - q_native) - pd_damping[start:end] * qd_native
            tau_delta = tau_measured - tau_pred

            q_native_np = q_native.detach().cpu().numpy()
            qd_native_np = qd_native.detach().cpu().numpy()
            q_target_np = q_target.detach().cpu().numpy()
            tau_measured_np = tau_measured.detach().cpu().numpy()
            tau_pred_np = tau_pred.detach().cpu().numpy()
            tau_delta_np = tau_delta.detach().cpu().numpy()

            print(f"{name}:")
            print(f"  q_native     = {np.array2string(q_native_np, precision=4, suppress_small=True)}")
            print(f"  qd_native    = {np.array2string(qd_native_np, precision=4, suppress_small=True)}")
            print(f"  q_target     = {np.array2string(q_target_np, precision=4, suppress_small=True)}")
            print(f"  tau_measured = {np.array2string(tau_measured_np, precision=4, suppress_small=True)}")
            print(f"  tau_pred     = {np.array2string(tau_pred_np, precision=4, suppress_small=True)}")
            print(f"  tau_delta    = {np.array2string(tau_delta_np, precision=4, suppress_small=True)}")
            print(
                f"  norms: measured={np.linalg.norm(tau_measured_np):.4f} "
                f"pred={np.linalg.norm(tau_pred_np):.4f} "
                f"delta={np.linalg.norm(tau_delta_np):.4f}"
            )


    def _build_reconstructed_base_obs(self, env_id):
        root_states = self._root_states[env_id:env_id + 1]
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]
        root_vel = root_states[:, 7:10]
        root_ang_vel = root_states[:, 10:13]

        root_h = root_pos[:, 2:3]
        heading_rot = calc_heading_quat_inv(root_rot)

        if self._local_root_obs:
            root_rot_obs = quat_mul(heading_rot, root_rot)
        else:
            root_rot_obs = root_rot
        root_rot_obs = quat_to_tan_norm(root_rot_obs)

        local_root_vel = my_quat_rotate(heading_rot, root_vel)
        local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

        recon_dof_pos = self._dof_pos[env_id].clone()
        recon_dof_vel = self._dof_vel[env_id].clone()
        compare_specs = [
            (0, 3, "pelvis", "torso"),
            (3, 6, "torso", "head"),
            (6, 9, "torso", "right_upper_arm"),
            (10, 13, "torso", "left_upper_arm"),
            (14, 17, "pelvis", "right_thigh"),
            (18, 21, "right_shin", "right_foot"),
            (21, 24, "pelvis", "left_thigh"),
            (25, 28, "left_shin", "left_foot"),
        ]

        for start, end, parent_name, child_name in compare_specs:
            recon_dof_pos[start:end] = quat_to_exp_map(
                self._relative_quat(env_id, parent_name, child_name).unsqueeze(0)
            )[0]
            recon_dof_vel[start:end] = self._relative_ang_vel_child_local(env_id, parent_name, child_name)

        dof_obs = dof_to_obs(recon_dof_pos.unsqueeze(0))

        key_body_pos = self._rigid_body_pos[env_id:env_id + 1][:, self._key_body_ids, :]
        local_key_body_pos = key_body_pos - root_pos.unsqueeze(-2)
        heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, local_key_body_pos.shape[1], 1))
        flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], 3)
        flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 4)
        local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
        flat_local_key_pos = local_end_pos.view(1, -1)

        load_cell_force = torch.zeros((1, 6), device=self.device, dtype=torch.float)

        recon_base_obs = torch.cat((
            root_h,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            recon_dof_vel.unsqueeze(0),
            load_cell_force,
            flat_local_key_pos
        ), dim=-1)
        return recon_base_obs[0]

    def _print_base_obs_alignment_diag(self):
        env_id = int(self._diag_compare_env)
        if env_id < 0 or env_id >= self.num_envs:
            return

        step = int(self.progress_buf[env_id].item())
        if step % self._diag_compare_interval != 0:
            return

        native_base_obs = super()._compute_humanoid_obs(torch.tensor([env_id], device=self.device))[0]
        recon_base_obs = self._build_reconstructed_base_obs(env_id)
        abs_diff = torch.abs(native_base_obs - recon_base_obs)

        slices = [
            ("root_h", 0, 1),
            ("root_rot_obs", 1, 7),
            ("local_root_vel", 7, 10),
            ("local_root_ang_vel", 10, 13),
            ("dof_obs", 13, 65),
            ("dof_vel", 65, 93),
            ("load_cell", 93, 99),
            ("key_body_pos", 99, 111),
        ]

        print("\n" + "=" * 80)
        print(f"[Base Obs Align Diag] env={env_id} step={step} train_stage={self.train_stage}")
        print("=" * 80)
        for name, start, end in slices:
            native_chunk = native_base_obs[start:end].detach().cpu().numpy()
            recon_chunk = recon_base_obs[start:end].detach().cpu().numpy()
            diff_chunk = abs_diff[start:end].detach().cpu().numpy()
            print(f"{name}: max_abs_diff={diff_chunk.max():.6f}, mean_abs_diff={diff_chunk.mean():.6f}")
            print(f"  native[:6] = {np.array2string(native_chunk[:6], precision=4, suppress_small=True)}")
            print(f"  recon [:6] = {np.array2string(recon_chunk[:6], precision=4, suppress_small=True)}")

    def _build_reconstructed_cmd_obs(self, env_id):
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        current_time = self.progress_buf[env_ids] * self.control_dt
        root_pos = self._root_states[env_ids, 0:3]
        root_rot = self._root_states[env_ids, 3:7]

        target_pos = self._traj_gen.get_position(env_ids, current_time)
        traj_points_world = self._traj_gen.get_observation_points(
            env_ids, current_time, self._traj_sample_times
        )

        traj_delta = traj_points_world[:, 0, 0:2] - target_pos[:, 0:2]
        target_dist = torch.norm(traj_delta, dim=-1, keepdim=True)

        prev_state = self._standing_phase_state[env_ids]
        prev_is_standing = self._prev_is_standing_phase[env_ids]

        enter_mask = target_dist[:, 0] < self._stand_enter_thres
        exit_mask = target_dist[:, 0] > self._stand_exit_thres

        is_standing_phase = torch.where(
            enter_mask,
            torch.ones_like(prev_state),
            torch.where(
                exit_mask,
                torch.zeros_like(prev_state),
                prev_state
            )
        )

        current_heading_xy = self._calc_current_heading_xy(root_rot)
        saved_heading_xy = self._stand_heading_dir[env_ids].clone()
        entering_standing = torch.logical_and(
            is_standing_phase,
            torch.logical_not(prev_is_standing)
        )
        if torch.any(entering_standing):
            saved_heading_xy[entering_standing] = current_heading_xy[entering_standing]

        heading_inv = calc_heading_quat_inv(root_rot)
        num_samples = self._num_traj_points
        heading_inv_expand = heading_inv.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 4)

        delta_pos = traj_points_world - root_pos.unsqueeze(1)
        delta_pos_flat = delta_pos.reshape(-1, 3)
        local_traj_points = my_quat_rotate(heading_inv_expand, delta_pos_flat)
        local_traj_points = local_traj_points.view(root_pos.shape[0], num_samples, 3)
        traj_obs = local_traj_points[..., 0:2].reshape(root_pos.shape[0], -1)

        cmd_speed = target_dist / self._traj_sample_times[0]
        standness = (self._stand_exit_thres - target_dist) / (
            self._stand_exit_thres - self._stand_enter_thres + 1e-6
        )
        standness = torch.clamp(standness, 0.0, 1.0)

        walk_dir = torch.nn.functional.normalize(traj_delta, dim=-1, eps=1e-6)

        walk_yaw_cos = torch.sum(current_heading_xy * walk_dir, dim=-1, keepdim=True)
        walk_yaw_sin = (
            current_heading_xy[:, 0:1] * walk_dir[:, 1:2]
            - current_heading_xy[:, 1:2] * walk_dir[:, 0:1]
        )

        stand_yaw_cos = torch.sum(current_heading_xy * saved_heading_xy, dim=-1, keepdim=True)
        stand_yaw_sin = (
            current_heading_xy[:, 0:1] * saved_heading_xy[:, 1:2]
            - current_heading_xy[:, 1:2] * saved_heading_xy[:, 0:1]
        )

        return torch.cat([
            traj_obs,
            cmd_speed,
            standness,
            walk_yaw_cos,
            walk_yaw_sin,
            stand_yaw_cos,
            stand_yaw_sin,
        ], dim=-1)[0]

    def _print_cmd_obs_alignment_diag(self):
        env_id = int(self._diag_compare_env)
        if env_id < 0 or env_id >= self.num_envs:
            return

        step = int(self.progress_buf[env_id].item())
        if step % self._diag_compare_interval != 0:
            return

        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        base_obs = super()._compute_humanoid_obs(env_ids)

        current_time = self.progress_buf[env_ids] * self.control_dt
        root_pos = self._root_states[env_ids, 0:3]
        root_rot = self._root_states[env_ids, 3:7]
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        traj_points_world = self._traj_gen.get_observation_points(
            env_ids, current_time, self._traj_sample_times
        )

        traj_delta = traj_points_world[:, 0, 0:2] - target_pos[:, 0:2]
        target_dist_flat = torch.norm(traj_delta, dim=-1)

        prev_state = self._standing_phase_state[env_ids]
        prev_is_standing = self._prev_is_standing_phase[env_ids]

        enter_mask = target_dist_flat < self._stand_enter_thres
        exit_mask = target_dist_flat > self._stand_exit_thres

        is_standing_phase = torch.where(
            enter_mask,
            torch.ones_like(prev_state),
            torch.where(
                exit_mask,
                torch.zeros_like(prev_state),
                prev_state
            )
        )

        current_heading_xy = self._calc_current_heading_xy(root_rot)
        saved_heading_xy = self._stand_heading_dir[env_ids].clone()
        entering_standing = torch.logical_and(
            is_standing_phase,
            torch.logical_not(prev_is_standing)
        )
        if torch.any(entering_standing):
            saved_heading_xy[entering_standing] = current_heading_xy[entering_standing]

        heading_inv = calc_heading_quat_inv(root_rot)
        num_samples = self._num_traj_points
        heading_inv_expand = heading_inv.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 4)
        delta_pos = traj_points_world - root_pos.unsqueeze(1)
        delta_pos_flat = delta_pos.reshape(-1, 3)
        local_traj_points = my_quat_rotate(heading_inv_expand, delta_pos_flat)
        local_traj_points = local_traj_points.view(root_pos.shape[0], num_samples, 3)

        native_traj_obs = local_traj_points[..., 0:2].reshape(root_pos.shape[0], -1)
        native_cmd_speed = target_dist_flat.unsqueeze(-1) / self._traj_sample_times[0]
        native_standness = (self._stand_exit_thres - target_dist_flat.unsqueeze(-1)) / (
            self._stand_exit_thres - self._stand_enter_thres + 1e-6
        )
        native_standness = torch.clamp(native_standness, 0.0, 1.0)

        walk_dir = torch.nn.functional.normalize(traj_delta, dim=-1, eps=1e-6)
        native_walk_yaw_cos = torch.sum(current_heading_xy * walk_dir, dim=-1, keepdim=True)
        native_walk_yaw_sin = (
            current_heading_xy[:, 0:1] * walk_dir[:, 1:2]
            - current_heading_xy[:, 1:2] * walk_dir[:, 0:1]
        )
        native_stand_yaw_cos = torch.sum(current_heading_xy * saved_heading_xy, dim=-1, keepdim=True)
        native_stand_yaw_sin = (
            current_heading_xy[:, 0:1] * saved_heading_xy[:, 1:2]
            - current_heading_xy[:, 1:2] * saved_heading_xy[:, 0:1]
        )

        native_cmd_obs = torch.cat([
            native_traj_obs,
            native_cmd_speed,
            native_standness,
            native_walk_yaw_cos,
            native_walk_yaw_sin,
            native_stand_yaw_cos,
            native_stand_yaw_sin,
        ], dim=-1)[0]

        recon_cmd_obs = self._build_reconstructed_cmd_obs(env_id)
        abs_diff = torch.abs(native_cmd_obs - recon_cmd_obs)

        slices = [
            ("traj_obs", 0, 6),
            ("cmd_speed", 6, 7),
            ("standness", 7, 8),
            ("walk_yaw_cos", 8, 9),
            ("walk_yaw_sin", 9, 10),
            ("stand_yaw_cos", 10, 11),
            ("stand_yaw_sin", 11, 12),
        ]

        print("\n" + "=" * 80)
        print(f"[Cmd Obs Align Diag] env={env_id} step={step} train_stage={self.train_stage}")
        print("=" * 80)
        for name, start, end in slices:
            native_chunk = native_cmd_obs[start:end].detach().cpu().numpy()
            recon_chunk = recon_cmd_obs[start:end].detach().cpu().numpy()
            diff_chunk = abs_diff[start:end].detach().cpu().numpy()
            print(f"{name}: max_abs_diff={diff_chunk.max():.6f}, mean_abs_diff={diff_chunk.mean():.6f}")
            print(f"  native = {np.array2string(native_chunk, precision=4, suppress_small=True)}")
            print(f"  recon  = {np.array2string(recon_chunk, precision=4, suppress_small=True)}")

    def _calc_current_heading_xy(self, root_rot):
        """
        根据 root_rot 计算当前身体前向在世界坐标系 XY 平面的单位向量
        """
        num_envs = root_rot.shape[0]
        forward_local = torch.zeros((num_envs, 3), device=root_rot.device, dtype=root_rot.dtype)
        forward_local[:, 0] = 1.0

        forward_world = my_quat_rotate(root_rot, forward_local)
        forward_xy = forward_world[:, 0:2]
        forward_xy = torch.nn.functional.normalize(forward_xy, dim=-1, eps=1e-6)
        return forward_xy

    def _update_standing_heading_cache(self, is_standing_phase, root_rot):
        """
        在 'walking -> standing' 切换的那一帧，保存当前朝向
        """
        current_heading_xy = self._calc_current_heading_xy(root_rot)

        entering_standing = torch.logical_and(
            is_standing_phase,
            torch.logical_not(self._prev_is_standing_phase)
        )

        if torch.any(entering_standing):
            self._stand_heading_dir[entering_standing] = current_heading_xy[entering_standing]

        self._prev_is_standing_phase[:] = is_standing_phase
        return current_heading_xy

    def _reset_standing_heading(self, env_ids):
        if len(env_ids) == 0:
            return

        root_rot = self._root_states[env_ids, 3:7]
        current_heading_xy = self._calc_current_heading_xy(root_rot)
        self._stand_heading_dir[env_ids] = current_heading_xy

        current_time = self.progress_buf[env_ids] * self.control_dt
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        future_traj_points = self._traj_gen.get_observation_points(env_ids, current_time, self._traj_sample_times)
        target_pos_05s = future_traj_points[:, 0, 0:2]
        target_dist = torch.norm(target_pos_05s - target_pos[..., 0:2], dim=-1)

        init_standing = target_dist < self._stand_enter_thres
        self._standing_phase_state[env_ids] = init_standing
        self._prev_is_standing_phase[env_ids] = init_standing
        return
    
    def _update_standing_phase_state(self, target_dist):
        enter_mask = target_dist < self._stand_enter_thres
        exit_mask = target_dist > self._stand_exit_thres

        self._standing_phase_state = torch.where(
            enter_mask,
            torch.ones_like(self._standing_phase_state),
            torch.where(
                exit_mask,
                torch.zeros_like(self._standing_phase_state),
                self._standing_phase_state
            )
        )
        return self._standing_phase_state

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

    def _compute_reward(self, actions):
        current_time = self.progress_buf * self.control_dt
        env_ids = torch.arange(self.num_envs, device=self.device)

        # ---------------------------------------------------------
        # 1) 当前轨迹信息
        # ---------------------------------------------------------
        target_pos = self._traj_gen.get_position(env_ids, current_time)   # [N, 3]
        root_pos = self._root_states[:, 0:3]                              # [N, 3]
        root_rot = self._root_states[:, 3:7]                              # [N, 4]

        future_traj_points = self._traj_gen.get_observation_points(
            env_ids, current_time, self._traj_sample_times
        )
        target_pos_05s = future_traj_points[:, 0, 0:2]                    # [N, 2]

        # 轨迹未来 0.5s 的位移，用于判断 standing / walking
        traj_delta = target_pos_05s - target_pos[:, 0:2]                  # [N, 2]
        target_dist = torch.norm(traj_delta, dim=-1)                      # [N]
        implied_target_speed = target_dist / 0.5

        # 用你已有的 hysteresis 状态机
        is_standing_phase = self._update_standing_phase_state(target_dist)

        # ---------------------------------------------------------
        # 2) 通用 reward 项
        # ---------------------------------------------------------
        # walking 位置奖励：继续鼓励贴近当前轨迹点
        pos_err_xy = target_pos[:, 0:2] - root_pos[:, 0:2]
        dist_sq = torch.sum(pos_err_xy ** 2, dim=-1)
        pos_reward = torch.exp(-2.0 * dist_sq)

        # 当前身体前向（世界系 XY）
        current_heading_xy = self._update_standing_heading_cache(is_standing_phase, root_rot)

        # walking 的方向：用“轨迹切线方向”，不要再用 target_pos - root_pos
        walk_dir = torch.nn.functional.normalize(traj_delta, dim=-1, eps=1e-6)

        root_vel_xy = self._root_states[:, 7:9]
        current_vel_proj = torch.sum(root_vel_xy * walk_dir, dim=-1)
        vel_reward = torch.exp(-2.0 * (current_vel_proj - implied_target_speed) ** 2)

        walk_heading_alignment = torch.sum(current_heading_xy * walk_dir, dim=-1)
        walk_heading_reward = torch.clamp(walk_heading_alignment, min=0.0, max=1.0)

        # standing 朝向奖励：保持进入 standing 时保存的朝向
        saved_heading_xy = self._stand_heading_dir
        stand_heading_alignment = torch.sum(current_heading_xy * saved_heading_xy, dim=-1)
        stand_heading_alignment = torch.clamp(stand_heading_alignment, min=-1.0, max=1.0)
        stand_heading_reward = torch.exp(-6.0 * (1.0 - stand_heading_alignment))

        stand_heading_dot = torch.sum(current_heading_xy * saved_heading_xy, dim=-1)
        stand_heading_dot = torch.clamp(stand_heading_dot, min=-1.0, max=1.0)
        stand_heading_cross = (
            current_heading_xy[:, 0] * saved_heading_xy[:, 1]
            - current_heading_xy[:, 1] * saved_heading_xy[:, 0]
        )
        stand_heading_err = torch.atan2(stand_heading_cross, stand_heading_dot)
        stand_heading_penalty = stand_heading_err ** 2

        # standing 位置奖励：带容差，不再逼绝对中心
        # 8cm 以内都算“站到位”
        stand_pos_tol = self._stand_pos_tol
        stand_pos_err = torch.norm(pos_err_xy, dim=-1)
        stand_pos_reward = torch.where(
            stand_pos_err < stand_pos_tol,
            torch.ones_like(stand_pos_err),
            torch.exp(-20.0 * (stand_pos_err - stand_pos_tol) ** 2)
        )

        # ---------------------------------------------------------
        # 3) 先统一构造 standing reward（S1 和 S4 共用）
        #    简洁版：位置 + 朝向 + 小幅晃动 + 躯干竖直 + 关节姿态惩罚
        # ---------------------------------------------------------
        root_vel_stand = self._root_states[:, 7:10]
        stand_speed_xy = torch.norm(root_vel_stand[:, 0:2], dim=-1)

        # 允许少量平衡晃动，超过阈值后再惩罚
        stand_vel_tol = 0.15
        stand_vel_excess = torch.clamp(stand_speed_xy - stand_vel_tol, min=0.0)
        stand_vel_reward = torch.exp(-8.0 * stand_vel_excess ** 2)

        # 额外约束：站立时不要原地绕 z 轴持续旋转
        stand_yaw_rate = torch.abs(self._root_states[:, 12])
        stand_yaw_rate_tol = 0.20
        stand_yaw_rate_excess = torch.clamp(stand_yaw_rate - stand_yaw_rate_tol, min=0.0)
        stand_yaw_rate_reward = torch.exp(-12.0 * stand_yaw_rate_excess ** 2)

        # 默认 0 位就是站姿，但允许一定关节偏离来维持平衡
        dof_abs_err = torch.abs(self._dof_pos - self._initial_dof_pos)
        joint_tol = 0.20
        joint_excess = torch.clamp(dof_abs_err - joint_tol, min=0.0)
        stand_joint_penalty = torch.sqrt(torch.mean(joint_excess ** 2, dim=-1) + 1e-8)

        # 躯干竖直奖励：用 pelvis -> head 向量和世界 z 轴的对齐程度
        pelvis_pos = self._rigid_body_pos[:, self._pelvis_body_id, :]
        head_pos = self._rigid_body_pos[:, self._head_body_id, :]
        trunk_vec = head_pos - pelvis_pos

        trunk_len = torch.norm(trunk_vec, dim=-1, keepdim=True).clamp(min=1e-6)
        trunk_dir = trunk_vec / trunk_len

        # 与世界系 z 轴的夹角余弦，1 表示完全竖直
        trunk_upright_cos = torch.clamp(trunk_dir[:, 2], min=-1.0, max=1.0)

        # 这个写法对“有点歪”就会比较敏感，比平方更适合纠正歪站
        stand_trunk_reward = torch.exp(-8.0 * (1.0 - trunk_upright_cos))

        # 双脚 XY 间距不要太大
        left_foot_pos = self._rigid_body_pos[:, self._left_foot_body_id, :]
        right_foot_pos = self._rigid_body_pos[:, self._right_foot_body_id, :]
        foot_delta_xy = left_foot_pos[:, 0:2] - right_foot_pos[:, 0:2]
        foot_dist_xy = torch.norm(foot_delta_xy, dim=-1)

        # 目标是“稍微收一点”，不是强行并脚
        desired_foot_dist = 0.22
        foot_dist_tol = 0.10   # 在 0.22 ± 0.10 的宽范围内都比较宽容
        foot_dist_excess = torch.clamp(
            foot_dist_xy - (desired_foot_dist + foot_dist_tol), min=0.0
        )
        stand_foot_spacing_reward = torch.exp(-10.0 * foot_dist_excess ** 2)

        # (b) 双脚尽量踩地：只在脚抬得明显偏高时才缓慢减分
        # 你当前 flat foot 大约在 z≈0.058，所以 target 先取 0.06
        foot_height_target = 0.06
        foot_height_tol = 0.025

        left_foot_height_excess = torch.clamp(
            left_foot_pos[:, 2] - (foot_height_target + foot_height_tol), min=0.0
        )
        right_foot_height_excess = torch.clamp(
            right_foot_pos[:, 2] - (foot_height_target + foot_height_tol), min=0.0
        )

        stand_foot_ground_reward = torch.exp(
            -80.0 * (left_foot_height_excess ** 2 + right_foot_height_excess ** 2)
        )
        # 最终 standing reward
        # 最终 standing reward
        stand_reward = (
            0.35 * stand_pos_reward
            - 0.50 * stand_heading_penalty
            + 0.15 * stand_vel_reward
            + 0.15 * stand_yaw_rate_reward
            + 0.20 * stand_trunk_reward
            + 0.08 * stand_foot_spacing_reward
            + 0.07 * stand_foot_ground_reward
            - 0.70 * stand_joint_penalty
        )

        stand_reward = torch.clamp(stand_reward, min=0.0)

        stand_reward = torch.clamp(stand_reward, min=0.0)
        # ---------------------------------------------------------
        # 4) 各阶段 reward 组合
        # ---------------------------------------------------------
        if self.train_stage == 1:
            # S1 直接学习与 S4 standing 完全一致的站立目标
            self.rew_buf[:] = stand_reward
            self.extras["amp_mask"] = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.float
            )

        elif self.train_stage == 4:
            stand_reward = stand_reward*3.0 # S4 站立奖励放大，使其数值与walking reward匹配
            walk_reward_s4 = pos_reward + vel_reward + walk_heading_reward
            self.rew_buf[:] = torch.where(is_standing_phase, stand_reward, walk_reward_s4)
            self.extras["amp_mask"] = (~is_standing_phase).float()

        else:
            # stage 2 / 3
            self.rew_buf[:] = pos_reward + vel_reward + walk_heading_reward

    def _compute_reset(self):
        super()._compute_reset()

        # 检测是否偏离轨迹太远
        current_time = self.progress_buf * self.control_dt
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        root_pos = self._root_states[..., 0:3]

        dist_sq = torch.sum((target_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
        
        # 如果偏离超过 1.5 米，强制重置
        max_dist_sq = 1.5 * 1.5
        has_strayed = dist_sq > max_dist_sq
        
        # 刚开始的前几帧不检测（反应时间）
        has_strayed = has_strayed & (self.progress_buf > 10)

        strayed_tensor = has_strayed.to(dtype=torch.long)

        # 更新重置 Buffer
        self.reset_buf = self.reset_buf | strayed_tensor
        self._terminate_buf = self._terminate_buf | strayed_tensor
        return
    
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
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
            
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._amp_local_root_obs)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        return

    def _load_motion(self, motion_file):
        print("================================")
        print("self.num_dof:")
        print(self.num_dof)
        print("self._key_body_ids.cpu().numpy():")
        print(self._key_body_ids.cpu().numpy())
        self._motion_lib = MotionLib(motion_file=motion_file, 
                                     num_dofs=self.num_dof,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return

    def reset_idx(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._maybe_export_reset_state_db(env_ids)

        root_pos = self._root_states[env_ids, 0:3]
        root_rot = self._root_states[env_ids, 3:7]

        self._traj_gen.reset(env_ids, root_pos, init_rot=root_rot)
        self._reset_standing_heading(env_ids)

        self._compute_observations(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _maybe_export_reset_state_db(self, env_ids):
        if (not self._export_reset_state_db) or self._export_reset_state_db_saved_once:
            return

        if self._export_reset_state_db_max_samples <= 0 or len(env_ids) == 0:
            return

        remaining = self._export_reset_state_db_max_samples - len(self._reset_state_db["root_pos"])
        if remaining <= 0:
            self._flush_reset_state_db(force=True)
            return

        env_ids_np = env_ids.detach().cpu().numpy()
        num_take = min(len(env_ids_np), self._export_reset_state_db_num_per_reset, remaining)
        if num_take <= 0:
            return

        if num_take < len(env_ids_np):
            choose_idx = np.random.choice(len(env_ids_np), size=num_take, replace=False)
            sample_env_ids = env_ids_np[choose_idx]
        else:
            sample_env_ids = env_ids_np

        sample_env_ids_t = torch.as_tensor(sample_env_ids, device=self.device, dtype=torch.long)

        source_mode = np.zeros(num_take, dtype=np.int32)
        if len(self._reset_ref_env_ids) > 0:
            ref_env_ids = set(self._reset_ref_env_ids.detach().cpu().numpy().tolist())
            source_mode = np.array([1 if int(eid) in ref_env_ids else 0 for eid in sample_env_ids], dtype=np.int32)

        root_states = self._root_states[sample_env_ids_t].detach().cpu().numpy()
        dof_pos = self._dof_pos[sample_env_ids_t].detach().cpu().numpy()
        dof_vel = self._dof_vel[sample_env_ids_t].detach().cpu().numpy()

        self._reset_state_db["root_pos"].extend(root_states[:, 0:3].astype(np.float32))
        self._reset_state_db["root_rot"].extend(root_states[:, 3:7].astype(np.float32))
        self._reset_state_db["root_vel"].extend(root_states[:, 7:10].astype(np.float32))
        self._reset_state_db["root_ang_vel"].extend(root_states[:, 10:13].astype(np.float32))
        self._reset_state_db["dof_pos"].extend(dof_pos.astype(np.float32))
        self._reset_state_db["dof_vel"].extend(dof_vel.astype(np.float32))
        self._reset_state_db["source_mode"].extend(source_mode.tolist())

        total = len(self._reset_state_db["root_pos"])
        if total % self._export_reset_state_db_flush_every == 0 or total >= self._export_reset_state_db_max_samples:
            self._flush_reset_state_db(force=(total >= self._export_reset_state_db_max_samples))
        return

    def _flush_reset_state_db(self, force=False):
        if len(self._reset_state_db["root_pos"]) == 0:
            return

        save_dir = os.path.dirname(self._export_reset_state_db_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        payload = {
            "root_pos": np.asarray(self._reset_state_db["root_pos"], dtype=np.float32),
            "root_rot": np.asarray(self._reset_state_db["root_rot"], dtype=np.float32),
            "dof_pos": np.asarray(self._reset_state_db["dof_pos"], dtype=np.float32),
            "root_vel": np.asarray(self._reset_state_db["root_vel"], dtype=np.float32),
            "root_ang_vel": np.asarray(self._reset_state_db["root_ang_vel"], dtype=np.float32),
            "dof_vel": np.asarray(self._reset_state_db["dof_vel"], dtype=np.float32),
            "source_mode": np.asarray(self._reset_state_db["source_mode"], dtype=np.int32),
        }
        np.savez(self._export_reset_state_db_path, **payload)

        total = payload["root_pos"].shape[0]
        print("[ResetStateDB] saved {:d} samples to {}".format(total, self._export_reset_state_db_path))

        if force or total >= self._export_reset_state_db_max_samples:
            self._export_reset_state_db_saved_once = True
            print("[ResetStateDB] collection finished.")
        return

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

        target_pos = self._traj_gen.get_position(env_ids_flat, current_time)
        traj_points_world = self._traj_gen.get_observation_points(
            env_ids_flat, current_time, self._traj_sample_times
        )

        # ------------------------------------------------------------------
        # 关键修复：
        # 在 obs 里先同步 standing hysteresis 和 saved heading，
        # 避免“obs 用旧 target、reward 用新 target”的 1-step 错位
        # ------------------------------------------------------------------
        traj_delta = traj_points_world[:, 0, 0:2] - target_pos[:, 0:2]
        target_dist_flat = torch.norm(traj_delta, dim=-1)  # [N]

        if env_ids is None:
            prev_state = self._standing_phase_state
            prev_is_standing = self._prev_is_standing_phase
        else:
            prev_state = self._standing_phase_state[env_ids_flat]
            prev_is_standing = self._prev_is_standing_phase[env_ids_flat]

        enter_mask = target_dist_flat < self._stand_enter_thres
        exit_mask = target_dist_flat > self._stand_exit_thres

        is_standing_phase = torch.where(
            enter_mask,
            torch.ones_like(prev_state),
            torch.where(
                exit_mask,
                torch.zeros_like(prev_state),
                prev_state
            )
        )

        current_heading_xy = self._calc_current_heading_xy(root_rot)
        entering_standing = torch.logical_and(
            is_standing_phase,
            torch.logical_not(prev_is_standing)
        )

        if torch.any(entering_standing):
            self._stand_heading_dir[env_ids_flat[entering_standing]] = current_heading_xy[entering_standing]

        self._standing_phase_state[env_ids_flat] = is_standing_phase
        self._prev_is_standing_phase[env_ids_flat] = is_standing_phase
        saved_heading_xy = self._stand_heading_dir[env_ids_flat]

        # ------------------------------------------------------------------
        # 原来的 traj obs
        # ------------------------------------------------------------------
        heading_inv = calc_heading_quat_inv(root_rot)
        num_samples = self._num_traj_points
        heading_inv_expand = heading_inv.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 4)

        delta_pos = traj_points_world - root_pos.unsqueeze(1)
        delta_pos_flat = delta_pos.reshape(-1, 3)

        local_traj_points = my_quat_rotate(heading_inv_expand, delta_pos_flat)
        local_traj_points = local_traj_points.view(root_pos.shape[0], num_samples, 3)

        traj_obs = local_traj_points[..., 0:2].reshape(root_pos.shape[0], -1)

        # ---------- extra command features ----------
        target_dist = target_dist_flat.unsqueeze(-1)
        cmd_speed = target_dist / self._traj_sample_times[0]

        standness = (self._stand_exit_thres - target_dist) / (
            self._stand_exit_thres - self._stand_enter_thres + 1e-6
        )
        standness = torch.clamp(standness, 0.0, 1.0)

        walk_dir = torch.nn.functional.normalize(traj_delta, dim=-1, eps=1e-6)

        walk_yaw_cos = torch.sum(current_heading_xy * walk_dir, dim=-1, keepdim=True)
        walk_yaw_sin = (
            current_heading_xy[:, 0:1] * walk_dir[:, 1:2]
            - current_heading_xy[:, 1:2] * walk_dir[:, 0:1]
        )

        stand_yaw_cos = torch.sum(current_heading_xy * saved_heading_xy, dim=-1, keepdim=True)
        stand_yaw_sin = (
            current_heading_xy[:, 0:1] * saved_heading_xy[:, 1:2]
            - current_heading_xy[:, 1:2] * saved_heading_xy[:, 0:1]
        )

        cmd_obs = torch.cat([
            traj_obs,
            cmd_speed,
            standness,
            walk_yaw_cos,
            walk_yaw_sin,
            stand_yaw_cos,
            stand_yaw_sin,
        ], dim=-1)

        # ---------------------------------------------------------
        # state obs 多帧堆叠；command 保持单帧
        # ---------------------------------------------------------
        if self._num_state_hist_steps > 1:
            if env_ids is None:
                self._update_hist_state_obs()
                self._state_hist_obs_buf[:, 0] = base_obs

                reset_mask = (self.progress_buf == 0)
                if torch.any(reset_mask):
                    self._state_hist_obs_buf[reset_mask] = base_obs[reset_mask].unsqueeze(1).repeat(
                        1, self._num_state_hist_steps, 1
                    )

                stacked_base_obs = self._state_hist_obs_buf.reshape(self.num_envs, -1)
            else:
                self._update_hist_state_obs(env_ids)
                self._state_hist_obs_buf[env_ids, 0] = base_obs

                reset_mask = (self.progress_buf[env_ids] == 0)
                if torch.any(reset_mask):
                    reset_env_ids = env_ids[reset_mask]
                    self._state_hist_obs_buf[reset_env_ids] = base_obs[reset_mask].unsqueeze(1).repeat(
                        1, self._num_state_hist_steps, 1
                    )

                stacked_base_obs = self._state_hist_obs_buf[env_ids].reshape(env_ids.shape[0], -1)
        else:
            stacked_base_obs = base_obs

        obs = torch.cat([stacked_base_obs, cmd_obs], dim=-1)
        return obs

    def _reset_actors(self, env_ids):
        if (self._state_init == HumanoidAMP_s1_Smpl.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == HumanoidAMP_s1_Smpl.StateInit.Start
              or self._state_init == HumanoidAMP_s1_Smpl.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == HumanoidAMP_s1_Smpl.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return
    
    def _reset_default(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HumanoidAMP_s1_Smpl.StateInit.Random
            or self._state_init == HumanoidAMP_s1_Smpl.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == HumanoidAMP_s1_Smpl.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.control_dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._amp_local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._root_states[env_ids, 0:3] = root_pos
        self._root_states[env_ids, 3:7] = root_rot
        self._root_states[env_ids, 7:10] = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._root_states, self._dof_pos, self._dof_vel, key_body_pos,
                                                                self._amp_local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._root_states[env_ids], self._dof_pos[env_ids], 
                                                                    self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                    self._amp_local_root_obs)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs