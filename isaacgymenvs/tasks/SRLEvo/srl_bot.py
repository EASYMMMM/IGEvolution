# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import math
from collections import deque 

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle, \
    quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv, quat_rotate_inverse
from isaacgymenvs.tasks.base.vec_task import VecTask



SRL_END_BODY_NAMES = ["SRL_right_end","SRL_left_end"] 

class SRL_bot(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        self._pd_control = self.cfg["env"].get("pdControl", True)
        self._force_control = self.cfg["env"].get("forceControl", False)
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.gait_period = self.cfg["env"]["gait_period"]
        self.foot_clearance = self.cfg["env"]["foot_clearance"]

        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]
        
        self.alive_reward_scale = self.cfg["env"]["alive_reward_scale"]
        self.progress_reward_scale = self.cfg["env"]["progress_reward_scale"]
        self.torques_cost_scale = self.cfg["env"]["torques_cost_scale"]
        self.dof_acc_cost_scale = self.cfg["env"]["dof_acc_cost_scale"]
        self.dof_vel_cost_scale = self.cfg["env"]["dof_vel_cost_scale"]
        self.dof_pos_cost_sacle = self.cfg["env"]["dof_pos_cost_sacle"]
        self.no_fly_penalty_scale = self.cfg["env"]["no_fly_penalty_scale"]
        self.tracking_ang_vel_reward_scale = self.cfg["env"]["tracking_ang_vel_reward_scale"]
        self.vel_tracking_reward_scale = self.cfg["env"]["vel_tracking_reward_scale"]
        self.gait_similarity_penalty_scale = self.cfg["env"]["gait_similarity_penalty_scale"]
        self.pelvis_height_reward_scale = self.cfg["env"]["pelvis_height_reward_scale"]
        self.orientation_reward_scale = self.cfg["env"]["orientation_reward_scale"]
        self.clearance_penalty_scale = self.cfg["env"]["clearance_penalty_scale"]
        self.lateral_distance_penalty_scale = self.cfg["env"]["lateral_distance_penalty_scale"]
        self.actions_rate_scale = self.cfg["env"]["actions_rate_scale"]
        self.actions_smoothness_scale = self.cfg["env"]["actions_smoothness_scale"]

        self.frame_stack = self.cfg["env"]["frame_stack"]  # 帧堆叠数量
        
        self.cfg["env"]["numObservations"] = 30 * self.frame_stack + 3
        self.cfg["env"]["numActions"] = 6
        # self.default_joint_angles = [0*np.pi, 
        #                              0.15*np.pi,
        #                              0.25*np.pi,
        #                              0*np.pi,
        #                              0.15*np.pi,
        #                              0.25*np.pi]
        self.default_joint_angles = [0*np.pi, 
                                     0.25*np.pi,
                                     0*np.pi,
                                     0*np.pi,
                                     0.25*np.pi,
                                     0*np.pi]


       
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.obs_buffer = torch.zeros((self.num_envs, self.frame_stack, np.int32(self.num_obs/self.frame_stack)), device=self.device)
        self.obs_mirrored_buffer = torch.zeros((self.num_envs, self.frame_stack, np.int32(self.num_obs/self.frame_stack)), device=self.device)
        self.phase_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) #  State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        # self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        # zero_tensor = torch.tensor([0.0], device=self.device)
        # self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
        #                                    torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        # MLY: user define
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.initial_dof_pos = torch.tensor(self.default_joint_angles, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        # ----- user define -----
        self.prev_srl_end_body_pos = torch.zeros((self.num_envs,2,3), device=self.device)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.target_ang_vel_z = torch.zeros(self.num_envs, device=self.device)
        self.target_pelvis_height = torch.full((self.num_envs,), 0.84, device=self.device)  # 0.84
        self.target_vel_x = torch.full((self.num_envs,), 0.0, device=self.device)  # 1.0

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]

        self.srl_root_states = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.srl_root_index ,  :]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        # mirror matrix
        self.mirror_idx_srl_dof = np.array([3,4,5,0.01,1,2])
        self.mirror_mat_srl_dof = torch.zeros((self.mirror_idx_srl_dof.shape[0], self.mirror_idx_srl_dof.shape[0]), dtype=torch.float32, device=self.device)
        for i, perm in enumerate(self.mirror_idx_srl_dof):
            self.mirror_mat_srl_dof[i, int(abs(perm))] = np.sign(perm)

        self.obs_scales={
            "lin_vel" : 1,
            "ang_vel" : 1,
            "dof_pos" : 1.0,
            "dof_vel" : 0.05,
            "height_measurements" : 5.0 }
        self.obs_scales_tensor = torch.tensor([
        self.obs_scales["lin_vel"],
        self.obs_scales["ang_vel"],
        self.obs_scales["dof_pos"],
        self.obs_scales["dof_vel"],
        ], device=self.device)

        if self.viewer != None:
            self._init_camera()
            

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def allocate_buffers(self):
        super().allocate_buffers()
        self.obs_mirrored_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)\
            
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing) 

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        asset_file = "mjcf/srl_bot/srl_bot.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)
            print('Asset file name:'+asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Add actuator list
        self._dof_names = self.gym.get_asset_dof_names(humanoid_asset)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_srl_end_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL_right_end")
        left_srl_end_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL_left_end")
        self.srl_root_index = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL_root", )
        sensor_pose = gymapi.Transform()
        self.right_foot_ssidx = self.gym.create_asset_force_sensor(humanoid_asset, right_srl_end_idx, sensor_pose)
        self.left_foot_ssidx = self.gym.create_asset_force_sensor(humanoid_asset, left_srl_end_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        # TODO: V3 MODEL Z
        start_pose.p = gymapi.Vec3(*get_axis_params(0.94, self.up_axis_idx))
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
            contact_filter = 0
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter , 0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
            # 设置初始 DOF 状态
            dof_state = np.zeros(self.num_dof, dtype=gymapi.DofState.dtype)
            for j in range(self.num_dof):
                dof_state["pos"][j] = self.default_joint_angles[j]

            self.gym.set_actor_dof_states(env_ptr, handle, dof_state, gymapi.STATE_ALL)
            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, handle)
            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)
            
            if self._force_control:
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["stiffness"].fill(0.0)
                dof_prop["damping"].fill(0.0)
                dof_prop["velocity"].fill(14.0)
                dof_prop["effort"].fill(200.0)
                dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
            elif (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
        
        dof_props = self.gym.get_asset_dof_properties(humanoid_asset)
        for i in range(len(dof_props)):
            self.p_gains[i] = float(dof_props[i]['stiffness'])
            self.d_gains[i] = float(dof_props[i]['damping'])

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            self.torque_limits[j] = dof_prop["effort"][j].item()
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])
        if self._force_control:
            self.dof_limits_lower[1] = self.default_joint_angles[1] - 45/180*np.pi
            self.dof_limits_lower[2] = self.default_joint_angles[2] - 45/180*np.pi
            self.dof_limits_lower[4] = self.default_joint_angles[4] - 45/180*np.pi
            self.dof_limits_lower[5] = self.default_joint_angles[5] - 45/180*np.pi
            self.dof_limits_upper[1] = self.default_joint_angles[1] + 45/180*np.pi
            self.dof_limits_upper[2] = self.default_joint_angles[2] + 45/180*np.pi
            self.dof_limits_upper[4] = self.default_joint_angles[4] + 45/180*np.pi
            self.dof_limits_upper[5] = self.default_joint_angles[5] + 45/180*np.pi

        feet_names = ['SRL_left_end','SRL_right_end']
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], feet_names[i])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._srl_end_ids = self._build_srl_end_body_ids_tensor(env_ptr, handle)

        self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)

        self._build_pd_action_offset_scale()

    def _build_srl_end_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []

        srl_end_body_names = SRL_END_BODY_NAMES
        
        for body_name in srl_end_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def compute_reward(self, actions):
        clearance_reward = self.compute_foot_clearance_reward()
        to_target = self.targets - self.initial_root_states[:, 0:3]
        srl_root_pos = self.srl_root_states[:, 0:3]
        srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :]
        self.rew_buf[:], self.reset_buf, self._terminate_buf[:] = compute_srl_reward(
            self.obs_buf,
            self.reset_buf,
            clearance_reward,
            to_target,
            self.progress_buf,
            self.phase_buf,
            self.actions,
            srl_end_body_pos,
            srl_root_pos,
            self.potentials,
            self.prev_potentials,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.gait_period,
            alive_reward_scale = self.alive_reward_scale,
            progress_reward_scale = self.progress_reward_scale,
            torques_cost_scale = self.torques_cost_scale,
            dof_acc_cost_scale = self.dof_acc_cost_scale,
            dof_vel_cost_scale = self.dof_vel_cost_scale,
            dof_pos_cost_sacle = self.dof_pos_cost_sacle,
            no_fly_penalty_scale = self.no_fly_penalty_scale,
            vel_tracking_reward_scale = self.vel_tracking_reward_scale,
            tracking_ang_vel_reward_scale = self.tracking_ang_vel_reward_scale,
            gait_similarity_penalty_scale = self.gait_similarity_penalty_scale,
            pelvis_height_reward_scale = self.pelvis_height_reward_scale,
            orientation_reward_scale = self.orientation_reward_scale,
            clearance_penalty_scale = self.clearance_penalty_scale,
            lateral_distance_penalty_scale = self.lateral_distance_penalty_scale,
            actions_rate_scale = self.actions_rate_scale,
            actions_smoothness_scale = self.actions_smoothness_scale,
        )   

    def compute_observations(self, env_ids=None):
        obs, obs_mirrored, potentials, prev_potentials = self._compute_srl_obs(env_ids)

        if (env_ids is None):
            self.obs_buffer[:, 1:, :] = self.obs_buffer[:, :-1, :]  # 向后移动数据
            self.obs_buffer[:, 0, :] = obs  # 将新的观测数据放到队列的开头

            self.obs_mirrored_buffer[:, 1:, :] = self.obs_mirrored_buffer[:, :-1, :]  # 向后移动数据
            self.obs_mirrored_buffer[:, 0, :] = obs_mirrored  # 将新的观测数据放到队列的开头
            
            # cat task command
            base_obs = self.obs_buffer.reshape(self.num_envs, -1)
            task_params = torch.stack((
                self.target_vel_x,
                self.target_ang_vel_z,
                self.target_pelvis_height,
            ), dim=-1)  # shape [num_envs, 3]
            mirrored_task_params = torch.stack((
                self.target_vel_x,
                - self.target_ang_vel_z,
                self.target_pelvis_height,
            ), dim=-1)  # shape [num_envs, 3]
            self.obs_buf[:] = torch.cat([base_obs, task_params], dim=-1)
            base_obs_mirrored = self.obs_mirrored_buffer.reshape(self.num_envs, -1)
            self.obs_mirrored_buf[:] = torch.cat([base_obs_mirrored, mirrored_task_params], dim=-1)

            self.potentials[:] = potentials
            self.prev_potentials[:] = prev_potentials
        else:
            # 对指定环境进行更新
            self.obs_buffer[env_ids, 1:, :] = self.obs_buffer[env_ids, :-1, :]
            self.obs_buffer[env_ids, 0, :] = obs

            self.obs_mirrored_buffer[env_ids, 1:, :] = self.obs_mirrored_buffer[env_ids, :-1, :]  # 向后移动数据
            self.obs_mirrored_buffer[env_ids, 0, :] = obs_mirrored  # 将新的观测数据放到队列的开头
            
            base_obs = self.obs_buffer[env_ids].reshape(len(env_ids), -1)
            task_params = torch.stack((
                self.target_vel_x[env_ids],
                self.target_ang_vel_z[env_ids],
                self.target_pelvis_height[env_ids],
            ), dim=-1)  # shape [len(env_ids), 3]  
            mirrored_task_params = torch.stack((
                self.target_vel_x[env_ids],
                -self.target_ang_vel_z[env_ids],
                self.target_pelvis_height[env_ids],
            ), dim=-1)  # shape [len(env_ids), 3]  
            self.obs_buf[env_ids] = torch.cat([base_obs, task_params], dim=-1)
            base_obs_mirrored = self.obs_mirrored_buffer[env_ids].reshape(len(env_ids), -1)
            self.obs_mirrored_buf[env_ids] = torch.cat([base_obs_mirrored, mirrored_task_params], dim=-1)

            self.potentials[env_ids] = potentials
            self.prev_potentials[env_ids] = prev_potentials

    
    def _compute_srl_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self.srl_root_states
            root_states[:,3:7] = self.root_states[:,3:7]
            root_states[:,10:13] = self.root_states[:,10:13]
            dof_pos = self.dof_pos
            dof_vel = self.dof_vel
            dof_force_tensor = self.dof_force_tensor
            progress_buf = self.progress_buf
            phase_buf = self.phase_buf
            initial_dof_pos = self.initial_dof_pos 
            gravity_vec = self.gravity_vec
            actions = self.actions
            targets = self.targets
            potentials = self.potentials
            target_vel_x = self.target_vel_x
            srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :]
        else:
            root_states = self.srl_root_states[env_ids]
            root_states[:,3:7] = self.root_states[env_ids,3:7]
            root_states[:,10:13] = self.root_states[env_ids,10:13]
            dof_pos = self.dof_pos[env_ids]
            dof_vel = self.dof_vel[env_ids]
            dof_force_tensor = self.dof_force_tensor[env_ids]
            progress_buf = self.progress_buf[env_ids]
            phase_buf = self.phase_buf[env_ids]
            initial_dof_pos = self.initial_dof_pos[env_ids]
            gravity_vec = self.gravity_vec[env_ids]
            actions = self.actions[env_ids]
            targets = self.targets[env_ids]
            potentials = self.potentials[env_ids]
            target_vel_x = self.target_vel_x[env_ids]
            srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, env_ids]
           
        obs, potentials, prev_potentials, = compute_srl_bot_observations(progress_buf, phase_buf, initial_dof_pos, root_states, dof_pos, dof_vel,
                                                       self.target_yaw, dof_force_tensor, gravity_vec, actions,
                                                       self.obs_scales_tensor, targets, potentials, self.dt, target_vel_x, self.gait_period)

        obs_mirrored  =  compute_srl_bot_observations_mirrored(progress_buf, phase_buf, self.mirror_mat_srl_dof, initial_dof_pos, root_states, dof_pos, dof_vel,
                                                       self.target_yaw, dof_force_tensor, gravity_vec, actions,
                                                       self.obs_scales_tensor, targets, potentials, self.dt, target_vel_x, self.gait_period)          
 
        return obs, obs_mirrored, potentials, prev_potentials
    

    def set_task_target(self):
        self.target_vel_x[:], self.target_pelvis_height[:], self.target_ang_vel_z[:], self.target_yaw[:] = set_task_target(self.target_vel_x,
                                                                                                       self.target_pelvis_height,
                                                                                                       self.target_ang_vel_z,
                                                                                                       self.target_yaw,
                                                                                                       self.progress_buf,
                                                                                                       max_episode_length=self.max_episode_length)

    def reset_done(self):
        _, done_env_ids = super().reset_done()
        # 添加镜像OBS
        self.obs_dict["obs_mirrored"] = torch.clamp(self.obs_mirrored_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        return self.obs_dict, done_env_ids
    
    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

 
        position_noise = torch.zeros((len(env_ids), self.num_dof), device=self.device)
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device)
        position_noise[:,1] = positions.squeeze(-1)
        position_noise[:,4] = - positions.squeeze(-1)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        # TODO: random init
        # self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] , self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, self.up_axis_idx] = 0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :].clone()
        self.prev_srl_end_body_pos[env_ids] = srl_end_body_pos[env_ids,:,:].clone()
        

        self.phase_buf[env_ids] = torch.randint(0, int(self.gait_period), (len(env_ids),), device=self.device, dtype=torch.long)

        for env_id in env_ids:
            self.obs_buffer[env_id] = 0
            self.obs_mirrored_buffer[env_id] = 0

        self._refresh_sim_tensors()

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if self._force_control:
            pd_tar = self._action_to_pd_targets(self.actions)
            torques = self.p_gains*(pd_tar - self.dof_pos) - self.d_gains*self.dof_vel
            self.torques = torch.clip(torques, -self.torque_limits, self.torque_limits).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        elif (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions) # pd_tar.shape: [num_actors, num_dofs]
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        self.phase_buf += 1

    
        self.extras["terminate"] = self._terminate_buf
        self.extras["target_yaw"] = self.target_yaw
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self._refresh_sim_tensors()
        self.compute_observations()
        self.compute_reward(self.actions)

        # TODO: Task Randomization
        # self.set_task_target()
        
        # mirrored info
        self.extras["obs_mirrored"] = self.obs_mirrored_buf.to(self.rl_device)  # 镜像观测
        self.obs_dict["obs_mirrored"] = torch.clamp(self.obs_mirrored_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            points = []
            colors = []
            for i in range(self.num_envs):
         
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return
    
    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        margin = 0.0 * math.pi / 180.0 
        low  = (self.dof_limits_lower + margin).unsqueeze(0)
        high = (self.dof_limits_upper - margin).unsqueeze(0)
        return torch.max(torch.min(pd_tar, high), low)

    
    def _build_pd_action_offset_scale(self):
        # Read joint limits
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        # For each joint (assume all 1 DoF), expand the action range to 70% of joint range
        for j in range(self.num_dof):
            curr_low = lim_low[j]
            curr_high = lim_high[j]
            curr_mid = 0.5 * (curr_high + curr_low)
            
            # 70% of joint range → to leave some margin
            # TODO: soft limit
            curr_scale = 0.45 * (curr_high - curr_low)
            curr_low = curr_mid - curr_scale
            curr_high = curr_mid + curr_scale

            lim_low[j] = curr_low
            lim_high[j] = curr_high

        # Compute offset and scale
        self._pd_action_scale  = 0.5 * (lim_high - lim_low)
        self._pd_action_scale  = to_torch(self._pd_action_scale, device=self.device)

        self._pd_action_offset = torch.tensor(self.default_joint_angles, device=self.device)
        self._pd_action_offset = self._pd_action_offset.unsqueeze(0).repeat(self.num_envs, 1)

    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render()
        return

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self.root_states[0, 0:3].cpu().numpy()
        
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
        char_root_pos = self.root_states[0, 0:3].cpu().numpy()
        
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

    def compute_foot_clearance_reward(self):
        curr = self._rigid_body_pos[:, self._srl_end_ids, :]   # [num_envs,2,3]
        prev = self.prev_srl_end_body_pos                       # [num_envs,2,3]
        self.prev_srl_end_body_pos = curr.clone()

        pz = curr[..., 2]                                       # [num_envs,2]
        dx = curr[..., 0] - prev[..., 0]
        dy = curr[..., 1] - prev[..., 1]
        v_xy = torch.sqrt(dx*dx + dy*dy) / self.dt             # [num_envs,2]
        zeros = torch.zeros_like(v_xy,device=v_xy.device)
        v_xy = torch.where(v_xy<0.8, zeros, v_xy)
        # 参数设定
        pz_target = self.foot_clearance

        this_term = (pz_target - pz) ** 2 * v_xy                 
        clearance_reward = torch.sum(this_term, dim=1)

        return clearance_reward

    def compute_air_time_reward(self):
        """
        计算脚部落地时的 air time 奖励。
        要求：
        - self.feet_air_time: 当前 air time 累积（每个脚一个）
        - self.last_contacts: 上一帧是否接触地面（每个脚一个）
        """
        # --- No fly --- 
        srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :]
        contact_threshold = 0.095  
        foot_height = srl_end_body_pos[:, 0:2, 2]  # 获取左脚的位置 
        contact_now = foot_height <= contact_threshold  # 当前帧是否接触地面（如 Cassie，1.0 N 以上算接触）

        # contact_filt 是合并当前和上一帧的接触状态（避免 PhysX 抖动）
        contact_filt = torch.logical_or(contact_now, self.last_contacts)
        
        # 记录首次落地（上一帧未接触，当前帧接触）
        first_contact = torch.logical_and(~self.last_contacts, contact_filt)  # shape: [num_envs, num_feet]

        # 更新接触状态
        self.last_contacts = contact_now.clone()

        # 更新时间累计（只在未接触时累加）
        self.feet_air_time += self.dt
        self.feet_air_time *= (~contact_filt).float()  # 接触就清零

        # 每只脚单独计算 air-time 奖励
        rew_air_time = (self.feet_air_time - 0.5) * first_contact.float()  # 只在首次接触给予奖励
        rew_air_time = rew_air_time.clamp(min=0.0)  # 可选：限制最小为 0
        rew_air_time = torch.sum(rew_air_time, dim=1)  # 每个环境总和

        return rew_air_time


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def quat_to_euler_xyz(q):
     # type: (Tensor) ->  Tensor 
    # Assumes q shape [N, 4], returns [N, 3] (yaw, pitch, roll)
    qx = q[:, 0]
    qy = q[:, 1]
    qz = q[:, 2]
    qw = q[:, 3]

    t0 = 2.0 * (qw * qx + qy * qz)
    t1 = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll_x = torch.atan2(t0, t1)

    t2 = 2.0 * (qw * qy - qz * qx)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)

    t3 = 2.0 * (qw * qz + qx * qy)
    t4 = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_z = torch.atan2(t3, t4)

    return torch.stack((yaw_z, pitch_y, roll_x), dim=-1)


@torch.jit.script
def compute_srl_reward(
    obs_buf,
    reset_buf,
    clearance_penalty,
    to_target,
    progress_buf,
    phase_buf,
    actions,
    srl_end_body_pos,
    srl_root_pos,
    potentials,
    prev_potentials,
    termination_height,
    death_cost,
    max_episode_length,
    gait_period,
    alive_reward_scale: float = 0,
    progress_reward_scale: float = 0,
    torques_cost_scale: float = 0,
    dof_acc_cost_scale: float = 0,
    dof_vel_cost_scale: float = 0,
    dof_pos_cost_sacle: float = 0,
    contact_force_cost_scale: float = 0,
    tracking_ang_vel_reward_scale: float = 0,
    no_fly_penalty_scale: float = 0,
    vel_tracking_reward_scale: float = 0,
    gait_similarity_penalty_scale: float = 0,
    pelvis_height_reward_scale: float = 0,
    orientation_reward_scale: float = 0,
    clearance_penalty_scale: float = 0,
    lateral_distance_penalty_scale: float = 0,
    actions_rate_scale: float = 0,
    actions_smoothness_scale: float = 0,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float,  float, float, float, float, float, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor]

    # obs = root_h,                             # 1    0
    #       local_root_vel ,                    # 3    1:3
    #       local_root_ang_vel ,                # 3    4:6
    #       euler_err,                          # 3    7:9
    #       srl_dof_obs * obs_scales[2],        # 6    10:15
    #       srl_dof_vel * obs_scales[3],        # 6    16:21
    #       actions ,                           # 6    22:27
    #       sin_phase,                          # 1    28
    #       0*cos_phase,                        # 1    29

    # --- Task Command ---
    target_pelvis_height = obs_buf[:, -1] 
    target_ang_vel_z = obs_buf[:, -2]
    target_vel_x = obs_buf[:, -3]

    # --- Termination handling ---
    alive_reward_coef = torch.where(target_vel_x < 0.1, 4*torch.ones_like(target_vel_x), torch.ones_like(target_vel_x))
    alive_reward = alive_reward_coef * torch.ones_like(potentials)
    progress_reward_coef = torch.where(target_vel_x < 0.1, torch.zeros_like(target_vel_x), torch.ones_like(target_vel_x))
    progress_reward = progress_reward_coef * (potentials - prev_potentials)

    # --- Pelvis velocity ---
    root_vel = obs_buf[:, 1:4] 
    root_target_vel = torch.zeros((root_vel.shape[0], 3), device=root_vel.device)
    root_target_vel[:, 0] = target_vel_x   
    vel_error_vec = root_vel - root_target_vel
    vel_tracking_reward =  vel_tracking_reward_scale *  torch.exp(-4 * torch.norm(vel_error_vec, dim=-1))  # α = 1.5

    # --- Torques cost ---
    torques_cost = 0 * torch.sum(actions ** 2, dim=-1)

    # --- DOF deviation cost ---
    dof_pos = obs_buf[:, 10:16]
    dof_pos[:,0] = dof_pos[:,0] * 3 # 
    dof_pos[:,3] = dof_pos[:,0] * 3
    dof_pos_cost = torch.sum(dof_pos ** 2, dim=-1)

    # --- DOF velocity cost ---
    dof_vel = obs_buf[:, 16:16+actions.shape[1]]
    dof_vel_cost = torch.sum(dof_vel ** 2, dim=-1)

    # --- DOF acceleration cost ---
    dof_vel_prev = obs_buf[:, 16+30:16+30+actions.shape[1]]  # 前一帧速度
    dof_acc = dof_vel - dof_vel_prev  # 关节加速度
    dof_acc_magnitude_sq = torch.sum(dof_acc ** 2, dim=-1)
    dof_acc_reward = torch.exp(- 2 * dof_acc_magnitude_sq)

    # --- Action Smooth ---
    actions_prev = obs_buf[:, 22+30:22+30+ actions.shape[1]]
    actions_prev_prev = obs_buf[:, 22+60:22+60+ actions.shape[1]]
    actions_rate = torch.sum((actions - actions_prev) ** 2, dim=-1)
    actions_smoothness = torch.sum((actions - 2*actions_prev + actions_prev_prev) ** 2, dim=-1)

    # --- Pelvis Orientation ---
    euler_err = obs_buf[:,7:10] 
    angle_diff = ((euler_err + math.pi) % (2 * math.pi)) - math.pi
    cos_angle = torch.cos(2 * angle_diff)
    ori_error = 1 - torch.mean(cos_angle, dim=-1)
    orientation_reward = torch.exp(-20 * ori_error   ) 

    # --- Pelvis height ---
    pelvis_height = obs_buf[:,0]
    # target_pelvis_height = torch.full((pelvis_height.shape[0],), 0.88, device=pelvis_height.device)
    pelvis_height_error = pelvis_height - target_pelvis_height
    pelvis_height_reward =  torch.exp(-12 * (10* pelvis_height_error) **2 ) 
    pelvis_height_penalty =  (10* pelvis_height_error) **2 

    # --- Pelvis angular rate ---
    root_ang_vel = obs_buf[:, 4:7]
    root_target_ang_vel = torch.zeros((root_ang_vel.shape[0], 3), device=root_ang_vel.device)
    root_target_ang_vel[:, 2] = target_ang_vel_z   
    root_ang_vel[:,2] = root_ang_vel[:,2] 
    ang_vel_error_vec = root_ang_vel - root_target_ang_vel
    ang_vel_tracking_reward = tracking_ang_vel_reward_scale * torch.exp(-3 * torch.norm(ang_vel_error_vec, dim=-1))   
   
    # --- No fly --- 
    contact_threshold = 0.095  
    left_foot_height = srl_end_body_pos[:, 0, 2]  # 获取左脚的位置 
    right_foot_height = srl_end_body_pos[:, 1, 2]  # 获取右脚的位置 
    no_feet_on_ground = (left_foot_height > contact_threshold) & (right_foot_height > contact_threshold)
    no_fly_penalty_coef = torch.where(target_vel_x < 0.1, 5*torch.ones_like(target_vel_x), torch.ones_like(target_vel_x))
    no_fly_penalty_scale = no_fly_penalty_scale * no_fly_penalty_coef
    # 如果两只脚同时离地，给予惩罚
    no_fly_penalty = torch.where(no_feet_on_ground, torch.ones_like(no_feet_on_ground) * no_fly_penalty_scale, torch.zeros_like(no_feet_on_ground))
    

    # --- Contact force cost ---
    # Assuming contact force encoded in obs_buf at some index e.g. 36+num_dof+num_dof: adjust as needed
    contact_force = obs_buf[:, 21:21+actions.shape[1]]  # You may need to adjust slice
    contact_force_magnitude = torch.norm(contact_force, dim=-1)
    contact_force_cost = contact_force_cost_scale * contact_force_magnitude

    # --- Feet Lateral Distance ---
    local_srl_end_body_pos = srl_end_body_pos - srl_root_pos.unsqueeze(-2)
    lateral_distance = torch.abs(local_srl_end_body_pos[:,0,1] - local_srl_end_body_pos[:,1,1])
    min_d, max_d = 0.21, 0.32 # FIXME: Lateral Distance 
    below_violation = torch.clamp(min_d - lateral_distance, min=0.0)
    above_violation = torch.clamp(lateral_distance - max_d, min=0.0)
    feet_lateral_penalty = below_violation + above_violation  

    # --- Foot Phase ---
    phase_t = (2 * math.pi / gait_period) * phase_buf.float()
    phase_left = phase_t
    phase_right = (phase_t + math.pi) % (2 * math.pi)
    expect_stancing_left  = (torch.sin(phase_left) > -0.2).float()  # stance phase left
    expect_stancing_right = (torch.sin(phase_right) > -0.2).float()   # stance phase right
    expect_flying_left  = (torch.sin(phase_left) < -0.7).float()  # swing phase left
    expect_flying_right = (torch.sin(phase_right) < -0.7).float()   # swing phase right
    is_contact_left = (left_foot_height < contact_threshold).float()
    is_contact_right = (right_foot_height < contact_threshold).float()
    # walking
    # 1. 期望接触但未接触 → 惩罚
    stance_miss_left  = expect_stancing_left  * (1.0 - is_contact_left)
    stance_miss_right = expect_stancing_right * (1.0 - is_contact_right)
    # 2. 期望摆动但发生接触 → 惩罚
    flying_miss_left  = expect_flying_left  * is_contact_left
    flying_miss_right = expect_flying_right * is_contact_right
    gait_phase_penalty = gait_similarity_penalty_scale * (stance_miss_left + stance_miss_right + flying_miss_left + flying_miss_right)
    # standing
    gait_phase_penalty_coef = torch.where(target_vel_x < 0.1, torch.zeros_like(target_vel_x), torch.ones_like(target_vel_x))
    gait_phase_penalty =  gait_phase_penalty*gait_phase_penalty_coef
   
    # --- Total reward ---
    total_reward = alive_reward_scale * alive_reward  \
        + progress_reward_scale * progress_reward \
        + vel_tracking_reward \
        + ang_vel_tracking_reward \
        + orientation_reward_scale * orientation_reward  \
        + pelvis_height_reward_scale * pelvis_height_reward \
        - torques_cost_scale * torques_cost \
        - dof_pos_cost_sacle * dof_pos_cost \
        - dof_vel_cost_scale * dof_vel_cost \
        + dof_acc_cost_scale * dof_acc_reward \
        - actions_rate_scale * actions_rate \
        - actions_smoothness_scale * actions_smoothness \
        - no_fly_penalty \
        - gait_phase_penalty \
        - clearance_penalty * clearance_penalty_scale \
        - lateral_distance_penalty_scale * feet_lateral_penalty

    # --- Handle termination ---
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # --- Reset agents ---
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    terminated = torch.zeros_like(reset_buf)
    terminated = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), terminated)
    return total_reward, reset, terminated


@torch.jit.script
def compute_srl_bot_observations(
    progress_buf,
    phase_buf,
    default_joint_pos,
    root_states ,
    dof_pos ,
    dof_vel ,
    target_yaw,
    dof_force_tensor ,
    gravity_vec ,
    actions,
    obs_scales,
    targets,
    potentials,
    dt,
    target_vel_x,
    gait_period,
)  :
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    # root state 分解
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    # base 高度
    root_h = root_pos[:, 2:3]

    euler = quat_to_euler_xyz(root_rot)
    target_euler = torch.zeros_like(euler,device=euler.device)
    target_euler[:,0] = target_yaw
    euler_err = target_euler - euler

    # 将线速度/角速度旋转到局部坐标
    local_root_vel     = quat_rotate_inverse(root_rot, root_vel)
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    projected_gravity  = quat_rotate_inverse(root_rot, gravity_vec)

    # SRL loadcell 是负载力传感器（向下为正）
    # load_cell_force = -load_cell

    # 主体关节位置编码（humanoid + SRL）
    # dof_obs = dof_to_obs(dof_pos)  
    srl_dof_obs   = dof_pos - default_joint_pos
    srl_dof_vel   = dof_vel
    srl_dof_force = dof_force_tensor 

    torso_position = root_states[:, 0:3]
    to_target = targets - torso_position
    to_target[:, 2] = 0

    # potentials
    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt


    # 假设周期为 T=60 步，则频率为 1/T，每一步是 2π/T 相位步长
    phase_t = (2 * math.pi / gait_period) * (phase_buf-1).float()  # shape: [num_envs]
    sin_phase = torch.sin(phase_t).unsqueeze(-1)
    cos_phase = torch.cos(phase_t).unsqueeze(-1)

    # standing phase mask
    standing_phase_mask = target_vel_x <= 0.1
    sin_phase = torch.where(standing_phase_mask.unsqueeze(-1), sin_phase*0, sin_phase)
    cos_phase = torch.where(standing_phase_mask.unsqueeze(-1), cos_phase*0, cos_phase)

    obs = torch.cat((root_h,                             # 1    0
                     local_root_vel ,                    # 3    1:3
                     local_root_ang_vel ,                # 3    4:6
                     euler_err,                          # 3    7:9
                     srl_dof_obs * obs_scales[2],        # 6    10:15
                     srl_dof_vel * obs_scales[3],        # 6    16:21
                     actions ,                           # 6    22:27
                     sin_phase,                          # 1    28
                     cos_phase,                        # 1    29
                    ), dim=-1)
    return obs , potentials, prev_potentials_new



@torch.jit.script
def compute_srl_bot_observations_mirrored(
    progress_buf,
    phase_buf,
    mirror_mat,
    default_joint_pos,
    root_states ,
    dof_pos ,
    dof_vel ,
    target_yaw,
    dof_force_tensor ,
    gravity_vec ,
    actions,
    obs_scales,
    targets,
    potentials,
    dt,
    target_vel_x,
    gait_period,
)  :
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float) ->  Tensor 
    
    
    # root state 分解
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    # base 高度
    root_h = root_pos[:, 2:3]

    # 将线速度/角速度旋转到局部坐标
    local_root_vel     = quat_rotate_inverse(root_rot, root_vel)
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    projected_gravity  = quat_rotate_inverse(root_rot, gravity_vec)

    # SRL loadcell 是负载力传感器（向下为正）
    # load_cell_force = -load_cell

    # 主体关节位置编码（humanoid + SRL）
    # dof_obs = dof_to_obs(dof_pos)  
    srl_dof_obs   = dof_pos - default_joint_pos  
    srl_dof_vel   = dof_vel
     
    torso_position = root_states[:, 0:3]
    to_target = targets - torso_position
    to_target[:, 2] = 0
 
    euler = quat_to_euler_xyz(root_rot)
    target_euler = torch.zeros_like(euler,device=euler.device)
    target_euler[:,0] = target_yaw
    euler_err = target_euler - euler


    # Mirrored
    local_root_vel[:,1] = -local_root_vel[:,1] # y方向速度
    local_root_ang_vel[:,0] = -local_root_ang_vel[:,0] # x轴角速度
    local_root_ang_vel[:,2] = -local_root_ang_vel[:,2] # z轴角速度
    projected_gravity[:,1] = -projected_gravity[:,1] # y方向投影
    srl_dof_obs = torch.matmul(srl_dof_obs, mirror_mat) # Perform the matrix multiplication to get mirrored dof_pos
    srl_dof_vel = torch.matmul(srl_dof_vel, mirror_mat)
    actions = torch.matmul(actions, mirror_mat)
    euler_err[:,0] = - euler_err[:,0] # yaw
    euler_err[:,2] = - euler_err[:,2] # roll

    # heading_proj[:,1] = -heading_proj[:,1]

    # 假设周期为 T=60 步，则频率为 1/T，每一步是 2π/T 相位步长
    phase_t = (2 * math.pi / gait_period) * (phase_buf-1).float()  # shape: [num_envs]
    sin_phase = torch.sin(phase_t).unsqueeze(-1)
    cos_phase = torch.cos(phase_t).unsqueeze(-1)

    # standing phase mask
    standing_phase_mask = target_vel_x <= 0.1
    sin_phase = torch.where(standing_phase_mask.unsqueeze(-1), sin_phase*0, sin_phase)
    cos_phase = torch.where(standing_phase_mask.unsqueeze(-1), cos_phase*0, cos_phase)

    # phase mirrored
    sin_phase = - sin_phase
    cos_phase = - cos_phase

    obs = torch.cat((root_h,                         # 1
                     local_root_vel  ,               # 3
                     local_root_ang_vel  ,           # 3
                     euler_err,                      # 3 
                     srl_dof_obs * obs_scales[2],    # 6
                     srl_dof_vel * obs_scales[3],        # 6
                     actions ,
                     sin_phase,    # TODO: mirrored
                     cos_phase,     
                    ), dim=-1)
    return obs  

# TODO: Task Setting
@torch.jit.script
def set_task_target(
    cur_target_vel_x,
    cur_target_pelvis_height,
    cur_target_ang_vel_z,
    cur_target_yaw,
    progress_buf,
    max_episode_length=1000,
)  :
# type: (Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    step_period = 240
    velocity_change_period = 200
    height_change_period = 180

    target_vel_x = cur_target_vel_x.clone().float() 
    target_pelvis_height = cur_target_pelvis_height.clone().float() 
    target_ang_vel_z = cur_target_ang_vel_z.clone().float() 
    target_yaw = cur_target_yaw.clone().float()

    # 候选集
    unit_vel_x = torch.ones_like(target_vel_x, device=target_vel_x.device)
    ang_vel_choices = torch.tensor([0.0, -0.785/2,  0.785/2], device=cur_target_vel_x.device)
    yaw_choices = torch.tensor([0.0,-0.785, 0.785],device=cur_target_vel_x.device)
    height_choices = torch.tensor([ 0.81, 0.83, 0.85, 0.87,], device=cur_target_vel_x.device)
    vel_x_choices = torch.tensor([0.0, 0.8, 1.0, 1.2, 1.4, 1.6], device=cur_target_vel_x.device)

    # 高度变化
    for i in range(max_episode_length//height_change_period):
        mask = progress_buf == height_change_period * i+1
        height_indices = torch.randint(0, len(height_choices), (len(target_pelvis_height),), device=target_pelvis_height.device)
        target_pelvis_height =  torch.where(mask, unit_vel_x*height_choices[height_indices], target_pelvis_height)
    mask = progress_buf == 1  # reset
    target_pelvis_height =  torch.where(mask, unit_vel_x*height_choices[2], target_pelvis_height)

    # # 单纯速度变化
    vel_x_choices = torch.tensor([0.6, 0.8, 1.0, 1.2], device=cur_target_vel_x.device)
    for i in range(max_episode_length//velocity_change_period):
        mask = progress_buf == velocity_change_period * i+1
        vel_indices = torch.randint(0, len(vel_x_choices), (len(target_vel_x),), device=target_vel_x.device)
        target_vel_x =  torch.where(mask, unit_vel_x*vel_x_choices[vel_indices], target_vel_x)

    # 速度变化 Delta Vx
    # delta_vel_x_choices = torch.tensor([-0.2, 0.0, 0.2], device=cur_target_vel_x.device)
    # delta_vel_x = torch.zeros_like(target_vel_x)
    # for i in range(max_episode_length//velocity_change_period):
    #     mask = progress_buf == velocity_change_period * i+1
    #     vel_indices =  torch.randint(0, len(delta_vel_x_choices), (len(target_vel_x),), device=delta_vel_x_choices.device)
    #     delta_vel_x =  torch.where(mask, unit_vel_x*delta_vel_x_choices[vel_indices], delta_vel_x)
    # target_vel_x =  target_vel_x + delta_vel_x
    # target_vel_x =  target_vel_x.clamp(min=0.8, max=1.6)

    # # progress buf 小于velocity_change_period时，固定站立
    # mask = progress_buf <= velocity_change_period
    # target_vel_x =  torch.where(mask, unit_vel_x*0.0, target_vel_x)

    # 速度变化+站立交替进行
    # for i in range(max_episode_length//step_period):
    #     mask = progress_buf == step_period * i+1
    #     vel_indices = torch.randint(1, len(vel_x_choices), (len(target_vel_x),), device=target_vel_x.device)
    #     if i%2 == 1:
    #         vel_indices = vel_indices * 0
    #     target_vel_x =  torch.where(mask, unit_vel_x*vel_x_choices[vel_indices], target_vel_x)

    # 随机朝向变化 以45为单位量
    delta_yaw = torch.zeros_like(target_yaw)
    ang_vel_indices = torch.randint(0, len(ang_vel_choices), (len(target_ang_vel_z),), device=target_vel_x.device)
    yaw_indices = torch.randint(0, len(yaw_choices), (len(target_yaw),), device=target_vel_x.device)
    for i in range(max_episode_length//step_period ):
        mask = progress_buf == step_period * i+1
        if i%2 == 0:
            ang_vel_indices = torch.full_like(ang_vel_indices, 0)
            yaw_indices = torch.full_like(yaw_indices, 0)
            if i  == 0:
                target_yaw = torch.where(mask, unit_vel_x*yaw_choices[yaw_indices], target_yaw)
        # turn left
        elif i%2 == 1:
            ang_vel_indices = torch.randint(1, len(ang_vel_choices), (len(target_ang_vel_z),), device=target_vel_x.device)
            yaw_indices = ang_vel_indices
        # set angular rate
        target_ang_vel_z =  torch.where(mask, unit_vel_x*ang_vel_choices[ang_vel_indices], target_ang_vel_z)
        # set yaw
        delta_yaw  = torch.where(mask, unit_vel_x*yaw_choices[yaw_indices], delta_yaw)
        # reset angular rate
        reset_mask = progress_buf == step_period * i+61
        ang_vel_indices = torch.full_like(ang_vel_indices, 0)
        target_ang_vel_z =  torch.where(reset_mask, unit_vel_x*ang_vel_choices[ang_vel_indices], target_ang_vel_z)
    target_yaw = target_yaw + delta_yaw

    return target_vel_x, target_pelvis_height, target_ang_vel_z, target_yaw

