"""Flat-ground humanoid base with the mclegs 58-muscle actuator."""

import numpy as np
import os
import sys
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, get_axis_params, calc_heading_quat_inv, \
     exp_map_to_quat, quat_to_tan_norm, my_quat_rotate

from ..base.vec_task import VecTask
from .compliance_controller import ComplianceController

DOF_BODY_IDS =[1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS =[0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
NUM_OBS = 13 + 52 + 28 + 12 + 6
NUM_ACTIONS = 58
KEY_BODY_NAMES =["right_hand", "left_hand", "right_foot", "left_foot"]

class HumanoidAMPMuscleBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        if self._pd_control:
            raise ValueError("HumanoidAMPMuscleFlat requires pdControl=False")
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"].get("randomization_params", {})
        self.enable_diagnostics = self.cfg["env"].get("enableMuscleDiagnostics", True)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self._humanoid_load_cell_obs = self.cfg["env"]["humanoid_load_cell_obs"]
        self.train_stage = self.cfg["env"].get("train_stage", 2)
        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.is_testing_mode = (self.cfg["env"]["numEnvs"] == 1)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dt = self.cfg["sim"]["dt"]
        self.control_dt = self.control_freq_inv * dt

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 3
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env, 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:, 7:13] = 0

        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)

        right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_shoulder_x")
        left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_shoulder_x")
        self._initial_dof_pos[:, right_shoulder_x_handle] = 0 * np.pi
        self._initial_dof_pos[:, left_shoulder_x_handle] = 0 * np.pi

        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self.compliance = ComplianceController(
            self.num_envs, self.device, self.cfg["env"].get("compliance", {}),
            self.cfg["env"].get("perturbation", {}), dt,
        )
        self._control_external_force = torch.zeros((self.num_envs, 3), device=self.device)
        if self.compliance.active:
            self._init_human_com_and_forces()
            self.compliance.reset(torch.arange(self.num_envs, device=self.device))
            if self.viewer and self.compliance.force_enabled:
                self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "muscle_flat_push")

        if self.viewer != None:
            self._init_camera()

        return

    def get_obs_size(self):
        return NUM_OBS

    def _init_human_com_and_forces(self):
        # Explicit anatomical whitelist excludes SRL stubs, even if they have mass.
        human_names = {"pelvis", "torso", "head", "right_upper_arm", "right_lower_arm",
                       "right_hand", "left_upper_arm", "left_lower_arm", "left_hand",
                       "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin", "left_foot"}
        names = self.gym.get_actor_rigid_body_names(self.envs[0], self.humanoid_handles[0])
        missing = human_names - set(names)
        if missing:
            raise ValueError(f"Missing anatomical bodies for human CoM: {sorted(missing)}")
        masses, offsets = [], []
        for env, actor in zip(self.envs, self.humanoid_handles):
            props = self.gym.get_actor_rigid_body_properties(env, actor)
            masses.append([p.mass if name in human_names else 0.0 for name, p in zip(names, props)])
            offsets.append([[p.com.x, p.com.y, p.com.z] for p in props])
        self._human_body_masses = to_torch(masses, device=self.device)
        if not torch.isfinite(self._human_body_masses).all() or (self._human_body_masses < 0).any():
            raise ValueError("Invalid humanoid body masses")
        self._human_mass = self._human_body_masses.sum(dim=-1, keepdim=True)
        if (self._human_mass <= 0).any():
            raise ValueError("Human mass must be positive")
        self._human_mass_fractions = self._human_body_masses / self._human_mass
        self._body_com_local = to_torch(offsets, device=self.device)
        self._external_body_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.human_com = self._compute_human_com()
        self._previous_human_com = self.human_com.clone()
        self.human_com_velocity = torch.zeros_like(self.human_com)
        self._com_velocity_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _compute_human_com(self):
        # Gym body poses are link poses; transform each body's local CoM.
        offset = my_quat_rotate(self._rigid_body_rot.reshape(-1, 4), self._body_com_local.reshape(-1, 3))
        body_com = self._rigid_body_pos + offset.view(self.num_envs, self.num_bodies, 3)
        return (body_com * self._human_mass_fractions[..., None]).sum(dim=1)

    def _reset_compliance(self, env_ids):
        if self.compliance.active:
            self.compliance.reset(env_ids)
            self._external_body_forces[env_ids] = 0
            self._control_external_force[env_ids] = 0
            com = self._compute_human_com()
            self.human_com[env_ids] = com[env_ids]
            self._previous_human_com[env_ids] = com[env_ids]
            self.human_com_velocity[env_ids] = 0
            self._com_velocity_valid[env_ids] = False

    def step(self, actions):
        """Task-local VecTask sequence with forces before every simulate."""
        if not self.compliance.active:
            return super().step(actions)
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.pre_physics_step(action_tensor)
        self._control_external_force.zero_()
        for _ in range(self.control_freq_inv):
            force = self.compliance.advance(self.cfg["sim"]["dt"])
            self._control_external_force += force / self.control_freq_inv
            if self.compliance.force_enabled:
                self._external_body_forces.copy_(self._human_mass_fractions[..., None] * force[:, None, :])
                # Tensor API applies these world-axis forces at each rigid body's CoM.
                self.gym.apply_rigid_body_force_tensors(
                    self.sim, gymtorch.unwrap_tensor(self._external_body_forces), None, gymapi.ENV_SPACE)
            self.gym.simulate(self.sim)
        if self.force_render:
            self.render()
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)
        self.post_physics_step()
        self.control_steps += 1
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](self.obs_buf)
        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.humanoid_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        asset_file = "mjcf/amp_humanoid_175_muscle.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = False
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = False
        load_cell_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL")
        self.load_cell_ssidx = self.gym.create_asset_force_sensor(humanoid_asset, load_cell_idx, sensor_pose, sensor_props)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)
        if self.num_dof != 28 or len(motor_efforts) != 28:
            raise ValueError("The 58-muscle model requires the 28-DOF mclegs humanoid asset")

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.1, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles =[]
        self.envs = []
        self.dof_limits_lower =[]
        self.dof_limits_upper =[]

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = 0

            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)

        # ---------- 提取物理参数并初始化多关节肌肉模型 ----------
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])

        for i in range(len(dof_props)):
            dof_props['stiffness'][i] = 0.0
            dof_props['damping'][i] = max(2.0, dof_props['damping'][i])
            if dof_props['effort'][i] < 1.0 or dof_props['effort'][i] > 1e10:
                dof_props['effort'][i] = 300.0

        for env_ptr, handle in zip(self.envs, self.humanoid_handles):
            self.gym.set_actor_dof_properties(env_ptr, handle, dof_props)

        dof_damping = torch.tensor([p['damping'] for p in dof_props], device=self.device, dtype=torch.float)
        dof_effort = torch.tensor([p['effort'] for p in dof_props], device=self.device, dtype=torch.float)

        from .muscle_model import TensorizedMuscleGroup

        q_init = torch.zeros(self.num_dof, device=self.device)

        # 🔥 获取左右腿矢状面关节索引，传给肌肉模型用于耦合
        self.r_hip_y_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_hip_y")
        self.r_knee_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_knee")
        self.r_ankle_y_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_ankle_y")
        self.l_hip_y_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_hip_y")
        self.l_knee_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_knee")
        self.l_ankle_y_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_ankle_y")

        sagittal_dofs = {
            'right':[self.r_hip_y_idx, self.r_knee_idx, self.r_ankle_y_idx],
            'left':[self.l_hip_y_idx, self.l_knee_idx, self.l_ankle_y_idx]
        }

        self.muscle_actuator = TensorizedMuscleGroup(
            num_envs=self.num_envs,
            num_dofs=self.num_dof,
            device=self.device,
            dt=self.cfg["sim"]["dt"],
            control_freq_inv=self.control_freq_inv,
            dof_lower=self.dof_limits_lower,
            dof_upper=self.dof_limits_upper,
            dof_damping=dof_damping,
            dof_effort=dof_effort,
            q_init=q_init,
            sagittal_dofs=sagittal_dofs,
            use_biarticular=True,
        )

        #if self.is_testing_mode:
        self.pelvis_bidx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "pelvis")
        self.thigh_bidx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "right_thigh")
        self.shin_bidx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "right_shin")
        self.foot_bidx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "right_foot")
        self.l_foot_bidx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "left_foot")

        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            load_cell_sensor = self.vec_sensor_tensor[:,self.load_cell_ssidx,:]
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
            load_cell_sensor = self.vec_sensor_tensor[env_ids,self.load_cell_ssidx,:]

        obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs, load_cell_sensor, self._humanoid_load_cell_obs)
        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        # 新增：重置环境时，将实际肌肉激活度归零
        self.muscle_actuator.a[env_ids] = 0.0
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        target_a = torch.clamp(0.5 * (self.actions + 1.0), 0.0, 1.0)

        if self.is_testing_mode:
            tau, self.muscle_details = self.muscle_actuator.forward(
                q=self._dof_pos,
                dq=self._dof_vel,
                target_a=target_a,
                disable_passive=True,
                return_details=True
            )
        else:
            tau = self.muscle_actuator.forward(
                q=self._dof_pos,
                dq=self._dof_vel,
                target_a=target_a,
                disable_passive=True
            )

        tau = torch.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        tau = torch.clamp(tau, -self.motor_efforts, self.motor_efforts)
        self.applied_tau = tau * self.power_scale

        force_tensor = gymtorch.unwrap_tensor(self.applied_tau)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        # ===================== 【新增：诊断日志输出】 =====================
        # 只在测试模式下，打印前 5 帧的数据到一个 txt 文件中
        if self.is_testing_mode and self.enable_diagnostics and self.progress_buf[0] < 5:
            env_id = 0
            mode = "58D" if self.muscle_actuator.num_muscles == 58 else "56D"
            with open(f"debug_compare_{mode}.txt", "a", encoding="utf-8") as f:
                f.write(f"\n=== 帧数 {self.progress_buf[0].item()} ===\n")

                # 提取三大关节施加到物理引擎里的最终合力矩
                tau_hip = self.applied_tau[env_id, self.r_hip_y_idx].item()
                tau_knee = self.applied_tau[env_id, self.r_knee_idx].item()
                tau_ankle = self.applied_tau[env_id, self.r_ankle_y_idx].item()

                f.write(f"【输出到物理引擎的最终力矩】\n")
                f.write(f"  髋(Hip): {tau_hip:.2f} Nm\n")
                f.write(f"  膝(Knee): {tau_knee:.2f} Nm\n")
                f.write(f"  踝(Ankle): {tau_ankle:.2f} Nm\n\n")

                f.write(f"【策略网络输出与肌肉状态】\n")
                if mode == "58D":
                    a_right = self.muscle_details['a'][env_id][44:51]
                    F_right = self.muscle_details['F'][env_id][44:51]
                    names =["GLU(臀)", "HFL(髋屈)", "HAM(腘绳)", "VAS(股四)", "GAS(腓肠)", "SOL(比目)", "TA(胫前)"]
                    for n, act, force in zip(names, a_right, F_right):
                        f.write(f"  {n}: 激活度={act.item():.3f}, 拉力={force.item():.1f} N\n")
                else:
                    # 56D
                    h_ag = target_a[env_id, 2*self.r_hip_y_idx].item()
                    h_ant = target_a[env_id, 2*self.r_hip_y_idx+1].item()
                    k_ag = target_a[env_id, 2*self.r_knee_idx].item()
                    k_ant = target_a[env_id, 2*self.r_knee_idx+1].item()
                    f.write(f"  旧_髋_Agonist: 激活度={h_ag:.3f}\n")
                    f.write(f"  旧_髋_Antagonist: 激活度={h_ant:.3f}\n")
                    f.write(f"  旧_膝_Agonist: 激活度={k_ag:.3f}\n")
                    f.write(f"  旧_膝_Antagonist: 激活度={k_ant:.3f}\n")
        # ===============================================================

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        if self.compliance.active:
            self.human_com = self._compute_human_com()
            # Average CoM velocity avoids depending on the native velocity reference point.
            velocity = (self.human_com - self._previous_human_com) / self.control_dt
            self.human_com_velocity = torch.where(self._com_velocity_valid[:, None], velocity, torch.zeros_like(velocity))
            self._previous_human_com.copy_(self.human_com)
            self._com_velocity_valid[:] = True
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        if self.viewer and self.compliance.force_enabled:
            # Drain once here and preserve the three existing VecTask shortcuts.
            # The parent's subsequent poll still handles window close and rendering.
            for event in self.gym.query_viewer_action_events(self.viewer):
                if event.value <= 0:
                    continue
                if event.action == "muscle_flat_push":
                    self.compliance.trigger(torch.tensor([0], device=self.device, dtype=torch.long))
                elif event.action == "QUIT":
                    sys.exit()
                elif event.action == "toggle_viewer_sync":
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif event.action == "record_frames":
                    self.record_frames = not self.record_frames

        super().render()
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids =[]
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids =[]
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                              self._cam_prev_char_pos[1] - 3.0,
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    dof_obs_size = 52
    dof_offsets =[0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs, load_cell, humanoid_load_cell_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool) -> Tensor
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

    if humanoid_load_cell_obs:
        load_cell_force = - load_cell
    else:
        load_cell_force = load_cell * 0

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h,
                     root_rot_obs,
                     local_root_vel,
                     local_root_ang_vel,
                     dof_obs,
                     dof_vel,
                     load_cell_force,
                     flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_observations_mirrored(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs, load_cell, humanoid_load_cell_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool) -> Tensor
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

    if humanoid_load_cell_obs:
        load_cell_force = - load_cell
    else:
        load_cell_force = load_cell * 0

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h,
                     root_rot_obs,
                     local_root_vel,
                     local_root_ang_vel,
                     dof_obs,
                     dof_vel,
                     load_cell_force,
                     flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_reward(obs_buf):
    reward = torch.zeros_like(obs_buf[:, 0])
    return reward

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        #has_fallen = torch.logical_and(fall_contact, fall_height)
        has_fallen = fall_contact
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
