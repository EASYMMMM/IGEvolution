'''
SRL-Gym
分段训练S1: 仅优化Humanoid行走, 观测空间中添加了6D的交互力传感器
'''

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, get_axis_params, calc_heading_quat_inv, \
     quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv

from ..base.vec_task import VecTask

"""
DOF_BODY_IDS = [
    1, 2, 3,           # L Leg
    4, 5, 6,           # R Leg
    7,            # Spine
    8, 9,            # Head
    10, 11, 12, 13, 14, # L Arm
    15, 16, 17, 18, 19  # R Arm
]

DOF_OFFSETS = [
    0, 3, 6, 
    9, 12, 15, 
    18, 21, 24, 
    27, 30, 
    33, 36, 39, 42, 45, 
    48, 51, 54, 57 
]

# Root(13) + Obs(19*6=114) + Vel(57) + KeyBody(12) + SRL(6) = 202
NUM_OBS = 202
NUM_ACTIONS = 57
KEY_BODY_NAMES = ["R_Hand", "L_Hand", "R_Ankle", "L_Ankle"]
"""
DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
NUM_OBS = 13 + 52 + 28 + 12 + 6 # [root_h, root_rot, root_vel, root_ang_vel, conceptual_joint_6d_obs, raw_hinge_dof_vel, key_body_pos]
NUM_ACTIONS = 28
KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]
class HumanoidAMP_s1_Smpl_Base_v2(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._amp_local_root_obs = self.cfg["env"].get("AMPlocalRootObs", True)
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_ignore_bodies = self.cfg["env"].get("terminationIgnoreBodies", [])
        self._self_collision_filter = self.cfg["env"].get("selfCollisionFilter", 1)
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        self._diag_print_asset_props = self.cfg["env"].get("diagPrintAssetProps", False)
        self._diag_print_runtime = self.cfg["env"].get("diagPrintRuntime", False)
        self._diag_runtime_interval = max(1, self.cfg["env"].get("diagRuntimeInterval", 20))
        self._diag_zero_action = self.cfg["env"].get("diagZeroAction", False)
        self._diag_print_pre_physics_reset = self.cfg["env"].get("diagPrintPrePhysicsReset", False)
        self._diag_disable_dof_drives = self.cfg["env"].get("diagDisableDofDrives", False)
        self._diag_fix_base_link = self.cfg["env"].get("diagFixBaseLink", False)
        self._diag_runtime_counter = 0
        self._last_pd_targets = None

        self._humanoid_load_cell_obs = self.cfg["env"]["humanoid_load_cell_obs"]
        self.train_stage = self.cfg["env"].get("train_stage", 2)  # 1: 原地站立，2: 直线行走，3: 曲线行走
        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        dt = self.cfg["sim"]["dt"]
        self.control_dt = self.control_freq_inv * dt
        
        # get gym GPU state tensors
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

        # create some wrapper tensors for different slices
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
        
        if self.viewer != None:
            self._init_camera()
            
        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
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
        asset_file = "mjcf/smpl_humanoid_0.xml"

        if "asset" in self.cfg["env"]:
            #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.fix_base_link = self._diag_fix_base_link
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        dof_names = self.gym.get_asset_dof_names(humanoid_asset)
        body_names = self.gym.get_asset_rigid_body_names(humanoid_asset)
        asset_dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        asset_shape_props = self.gym.get_asset_rigid_shape_properties(humanoid_asset)
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        # 人机交互处传感器
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
        self._dof_names = list(dof_names)
        self._body_names = list(body_names)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.1, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            handle = self.gym.create_actor(
                env_ptr,
                humanoid_asset,
                start_pose,
                "humanoid",
                i,
                self._self_collision_filter,
                0,
            )

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            if self._pd_control or self._diag_disable_dof_drives:
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)

            if self._diag_disable_dof_drives:
                dof_prop["driveMode"] = gymapi.DOF_MODE_NONE
                dof_prop["stiffness"] = 0.0
                dof_prop["damping"] = 0.0
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
            elif self._pd_control:
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        actor_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
        actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
        self._body_masses = to_torch(
            [float(prop.mass) for prop in actor_body_props],
            device=self.device,
        )
        self._body_local_com = to_torch(
            [[float(prop.com.x), float(prop.com.y), float(prop.com.z)] for prop in actor_body_props],
            device=self.device,
        )
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
        self._termination_ignore_body_ids = self._build_termination_ignore_body_ids_tensor(env_ptr, handle)
        
        if (self._pd_control):
            self._build_pd_action_offset_scale()

        if self._diag_print_asset_props:
            self._print_v2_import_diagnostics(
                asset_file=asset_file,
                asset_dof_prop=asset_dof_prop,
                actor_dof_prop=dof_prop,
                actuator_props=actuator_props,
                motor_efforts=motor_efforts,
                actor_body_props=actor_body_props,
                asset_shape_props=asset_shape_props,
                actor_shape_props=actor_shape_props,
                fix_base_link=asset_options.fix_base_link,
            )

        return

    @staticmethod
    def _format_vec3(value):
        return f"({float(value.x):.6g}, {float(value.y):.6g}, {float(value.z):.6g})"

    @classmethod
    def _format_inertia(cls, inertia):
        return f"[x={cls._format_vec3(inertia.x)}, y={cls._format_vec3(inertia.y)}, z={cls._format_vec3(inertia.z)}]"

    def _compute_system_com_state(self, env_id):
        body_rot = self._rigid_body_rot[env_id]
        com_offset_world = my_quat_rotate(body_rot, self._body_local_com)
        body_com_pos = self._rigid_body_pos[env_id] + com_offset_world
        body_com_vel = self._rigid_body_vel[env_id] + torch.cross(
            self._rigid_body_ang_vel[env_id],
            com_offset_world,
            dim=-1,
        )

        total_mass = torch.sum(self._body_masses)
        mass_column = self._body_masses.unsqueeze(-1)
        system_com_pos = torch.sum(mass_column * body_com_pos, dim=0) / total_mass
        system_com_vel = torch.sum(mass_column * body_com_vel, dim=0) / total_mass
        return system_com_pos, system_com_vel, total_mass

    def _print_v2_import_diagnostics(self, asset_file, asset_dof_prop, actor_dof_prop,
                                     actuator_props, motor_efforts, actor_body_props,
                                     asset_shape_props, actor_shape_props,
                                     fix_base_link):
        print("=" * 120)
        print("[V2 Import Diagnostics]")
        print("=" * 120)
        print(f"asset_file={asset_file}")
        print(
            f"pd_control={self._pd_control} num_bodies={self.num_bodies} "
            f"num_dof={self.num_dof} num_actuators={len(actuator_props)}"
        )
        print(
            f"sim_dt={self.cfg['sim']['dt']:.6f} controlFrequencyInv={self.control_freq_inv} "
            f"control_dt={self.cfg['sim']['dt'] * self.control_freq_inv:.6f}"
        )
        print(
            f"ground: static_friction={self.plane_static_friction:.4f} "
            f"dynamic_friction={self.plane_dynamic_friction:.4f} "
            f"restitution={self.plane_restitution:.4f}"
        )
        print(
            f"termination: enabled={self._enable_early_termination} "
            f"height={self._termination_height:.4f} allowed_contact_bodies={self._contact_bodies}"
        )
        print(f"termination_ignore_bodies={self._termination_ignore_body_names}")
        print(f"self_collision_filter={self._self_collision_filter}")
        print(f"diag_zero_action={self._diag_zero_action}")
        print(f"diag_disable_dof_drives={self._diag_disable_dof_drives}")
        print(f"diag_fix_base_link={self._diag_fix_base_link} asset_options.fix_base_link={fix_base_link}")

        print("-" * 120)
        print("DOF properties: asset import -> actor used by training")
        fields = actor_dof_prop.dtype.names
        for i, name in enumerate(self._dof_names):
            asset_effort = float(asset_dof_prop["effort"][i]) if "effort" in fields else float("nan")
            actor_effort = float(actor_dof_prop["effort"][i]) if "effort" in fields else float("nan")
            asset_mode = int(asset_dof_prop["driveMode"][i]) if "driveMode" in fields else -1
            actor_mode = int(actor_dof_prop["driveMode"][i]) if "driveMode" in fields else -1
            print(
                f"{i:02d} {name}: "
                f"limit=[{float(actor_dof_prop['lower'][i]): .6f}, {float(actor_dof_prop['upper'][i]): .6f}] "
                f"kp={float(actor_dof_prop['stiffness'][i]):.6g} "
                f"kd={float(actor_dof_prop['damping'][i]):.6g} "
                f"effort(asset/actor)={asset_effort:.6g}/{actor_effort:.6g} "
                f"driveMode(asset/actor)={asset_mode}/{actor_mode}"
            )

        print("-" * 120)
        print("Policy action mapping: target = offset + scale * action, action in [-1, 1]")
        if self._pd_control:
            offsets = self._pd_action_offset.detach().cpu().numpy()
            scales = self._pd_action_scale.detach().cpu().numpy()
            for i, name in enumerate(self._dof_names):
                print(
                    f"{i:02d} {name}: offset={offsets[i]: .6f} scale={scales[i]: .6f} "
                    f"target_range=[{offsets[i] - scales[i]: .6f}, {offsets[i] + scales[i]: .6f}]"
                )
        else:
            for i, name in enumerate(self._dof_names):
                effort = float(self.motor_efforts[i].item())
                print(f"{i:02d} {name}: torque_range=[{-effort:.6g}, {effort:.6g}]")

        print("-" * 120)
        print("Actuator properties returned by the Isaac Gym importer")
        for i, prop in enumerate(actuator_props):
            print(f"{i:02d}: motor_effort={float(motor_efforts[i]):.6g}")

        print("-" * 120)
        print("Actor rigid body mass / center of mass / inertia used by training")
        total_mass = 0.0
        for i, name in enumerate(self._body_names):
            actor_prop = actor_body_props[i]
            total_mass += float(actor_prop.mass)
            print(
                f"{i:02d} {name}: mass={float(actor_prop.mass):.6g} "
                f"com={self._format_vec3(actor_prop.com)} "
                f"inertia={self._format_inertia(actor_prop.inertia)}"
            )
        print(f"actor_total_mass={total_mass:.6g}")

        print("-" * 120)
        print("Rigid shape contact properties: asset import -> actor")
        shape_fields = (
            "friction", "rolling_friction", "torsion_friction",
            "restitution", "contact_offset", "rest_offset"
        )
        for i, (asset_prop, actor_prop) in enumerate(zip(asset_shape_props, actor_shape_props)):
            values = []
            for field in shape_fields:
                if hasattr(actor_prop, field):
                    values.append(
                        f"{field}={float(getattr(asset_prop, field)):.6g}"
                        f"->{float(getattr(actor_prop, field)):.6g}"
                    )
            print(f"{i:02d}: " + " ".join(values))
        print("=" * 120)

    def _build_pd_action_offset_scale(self):
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(len(DOF_OFFSETS) - 1):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - dof_offset

            if dof_size == 3:
                # Preserve the legacy policy interface: zero action means zero
                # overall joint rotation, and each component spans [-pi, pi].
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi
            else:
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]

                # Diagnostic mapping: keep the original scale, but remove the
                # joint-limit midpoint bias so zero action targets zero angle.
                curr_scale = 0.7 * (curr_high - curr_low)
                lim_low[dof_offset] = -curr_scale
                lim_high[dof_offset] = curr_scale

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._termination_ignore_body_ids,
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
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if self._diag_zero_action:
            self.actions.zero_()

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            self._last_pd_targets = pd_tar
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            self._last_pd_targets = None
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        if self._diag_print_runtime:
            self._print_runtime_diagnostics()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def _print_runtime_diagnostics(self):
        if self._diag_runtime_counter % self._diag_runtime_interval != 0:
            self._diag_runtime_counter += 1
            return
        self._diag_runtime_counter += 1

        env_id = 0
        action = self.actions[env_id].detach().cpu().numpy()
        dof_pos = self._dof_pos[env_id].detach().cpu().numpy()
        dof_vel = self._dof_vel[env_id].detach().cpu().numpy()
        dof_force = self.dof_force_tensor[env_id].detach().cpu().numpy()
        contact = self._contact_forces[env_id].detach()
        contact_norm = torch.norm(contact, dim=-1)
        body_height = self._rigid_body_pos[env_id, :, 2].detach()

        termination_ignored = torch.zeros(self.num_bodies, device=self.device, dtype=torch.bool)
        termination_ignored[self._termination_ignore_body_ids] = True
        actual_fall_contact = torch.any(contact > 0.1, dim=-1) & (~termination_ignored)
        actual_fall_height = (body_height < self._termination_height) & (~termination_ignored)

        print("=" * 120)
        print(
            f"[V2 Runtime Diagnostics] sample={self._diag_runtime_counter - 1} "
            f"env={env_id} episode_step={int(self.progress_buf[env_id].item())} "
            f"reset={int(self.reset_buf[env_id].item())} terminate={int(self._terminate_buf[env_id].item())}"
        )
        print(f"action={np.array2string(action, precision=4, suppress_small=True, max_line_width=240)}")
        if self._last_pd_targets is not None:
            pd_target = self._last_pd_targets[env_id].detach().cpu().numpy()
            print(f"pd_target={np.array2string(pd_target, precision=4, suppress_small=True, max_line_width=240)}")
        print(f"dof_pos={np.array2string(dof_pos, precision=4, suppress_small=True, max_line_width=240)}")
        print(f"dof_vel={np.array2string(dof_vel, precision=4, suppress_small=True, max_line_width=240)}")
        print(f"dof_force={np.array2string(dof_force, precision=4, suppress_small=True, max_line_width=240)}")
        print(
            f"root_pos={np.array2string(self._root_states[env_id, 0:3].detach().cpu().numpy(), precision=4)} "
            f"root_lin_vel={np.array2string(self._root_states[env_id, 7:10].detach().cpu().numpy(), precision=4)} "
            f"root_ang_vel={np.array2string(self._root_states[env_id, 10:13].detach().cpu().numpy(), precision=4)}"
        )
        system_com_pos, system_com_vel, total_mass = self._compute_system_com_state(env_id)
        print(
            f"system_com_pos={np.array2string(system_com_pos.detach().cpu().numpy(), precision=4)} "
            f"system_com_vel={np.array2string(system_com_vel.detach().cpu().numpy(), precision=4)} "
            f"total_mass={float(total_mass):.6f}"
        )

        print("contacts with force norm > 0.1:")
        active_ids = torch.nonzero(contact_norm > 0.1, as_tuple=False).flatten().cpu().tolist()
        if len(active_ids) == 0:
            print("  none")
        else:
            for body_id in active_ids:
                force = contact[body_id].cpu().numpy()
                print(
                    f"  {body_id:02d} {self._body_names[body_id]}: "
                    f"force={np.array2string(force, precision=4)} norm={float(contact_norm[body_id]):.4f} "
                    f"height={float(body_height[body_id]):.4f} "
                    f"termination_ignored={bool(termination_ignored[body_id].item())}"
                )

        fall_contact_ids = torch.nonzero(actual_fall_contact, as_tuple=False).flatten().cpu().tolist()
        fall_height_ids = torch.nonzero(actual_fall_height, as_tuple=False).flatten().cpu().tolist()
        print("fall_contact_bodies=" + str([self._body_names[i] for i in fall_contact_ids]))
        print("below_termination_height=" + str([
            f"{self._body_names[i]}({float(body_height[i]):.4f})" for i in fall_height_ids
        ]))
        print("=" * 120)

    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render()
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    @staticmethod
    def _is_auto_termination_ignore_body(body_name):
        return (
            body_name.endswith("_x_link")
            or body_name.endswith("_y_link")
            or body_name.endswith("_z_link")
        )

    def _build_termination_ignore_body_ids_tensor(self, env_ptr, actor_handle):
        ignore_body_names = set(self._contact_bodies)
        ignore_body_names.update(self._termination_ignore_bodies)

        for body_name in self._body_names:
            if self._is_auto_termination_ignore_body(body_name):
                ignore_body_names.add(body_name)

        body_ids = []
        self._termination_ignore_body_names = []
        for body_name in self._body_names:
            if body_name not in ignore_body_names:
                continue

            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)
            self._termination_ignore_body_names.append(body_name)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

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
def hinge_xyz_chain_to_quat(joint_angles):
    # type: (Tensor) -> Tensor
    """Compose nested local X -> Y -> Z hinge rotations."""
    half_x = 0.5 * joint_angles[:, 0]
    half_y = 0.5 * joint_angles[:, 1]
    half_z = 0.5 * joint_angles[:, 2]
    zero = torch.zeros_like(half_x)

    quat_x = torch.stack(
        [torch.sin(half_x), zero, zero, torch.cos(half_x)], dim=-1
    )
    quat_y = torch.stack(
        [zero, torch.sin(half_y), zero, torch.cos(half_y)], dim=-1
    )
    quat_z = torch.stack(
        [zero, zero, torch.sin(half_z), torch.cos(half_z)], dim=-1
    )

    return quat_mul(quat_mul(quat_x, quat_y), quat_z)


@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    # v2 chain-XML route:
    # control/state tensors stay in raw hinge coordinates, while observations
    # reconstruct each conceptual 3DoF joint back into the original 6D
    # tan-norm rotation feature.
    #dof_obs_size = 64
    #dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
    # 更新 DoF Obs 尺寸 (19*6=114)
    #dof_obs_size = 114
    #dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57]
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device, dtype=pose.dtype)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        if (dof_size == 3):
            joint_pose_q = hinge_xyz_chain_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            joint_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            joint_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + joint_obs_size)] = joint_dof_obs
        dof_obs_offset += joint_obs_size

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

    # 6D人机交互力
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

    # 6D人机交互力
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
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, termination_ignore_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, termination_ignore_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, termination_ignore_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
