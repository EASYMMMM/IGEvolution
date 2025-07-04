'''
SRL-Gym marl(v3)
Humanoid-SRL 协作环境代码
定义环境基础内容
相比第二版新增内容：
1. 使用v3系列的仿真模型
2. Humanoid和外肢体使用完全分开的观测空间 (对应的训练程序为srl_continuous_marl)
3. 分段训练
'''

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

from isaacgym import gymtorch
from isaacgym import gymapi
import math
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, get_axis_params, calc_heading_quat_inv, \
     exp_map_to_quat, quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv, quat_rotate_inverse

from ..base.vec_task import VecTask

DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
NUM_OBS = 131 + 111 # humanoid: 111, SRL 38
NUM_HUMANOID_OBS = 13 + 52 + 28 + 12 + 6  # Humanoid基本观测 + 人机交互力
NUM_ACTIONS = 28 + 6 + 1  # Actions humanoid (Dof=28) + SRL + freejoint-Y
NUM_HUMANOID_ACTIONS = 28

SRL_ROOT_BODY_NAMES = ["SRL", "SRL_root"]
UPPER_BODY_NAMES = ["pelvis", "torso"]
KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]  # body end + SRL end
SRL_END_BODY_NAMES = ["SRL_right_end","SRL_left_end"] 
SRL_CONTACT_BODY_NAMES = ['SRL_root', 'SRL_leg2', 'SRL_shin11', 'SRL_right_end', 'SRL_leg1', 'SRL_shin1', 'SRL_left_end']

class HumanoidAMPSRLmarlBase(VecTask):

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
        # self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]  
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        # --- SRL-Gym Defined ---
        self._torque_threshold = self.cfg["env"]["torque_threshold"]
        self._upper_reward_w = self.cfg["env"]["upper_reward_w"]
        self._srl_torque_reward_w = self.cfg["env"]["srl_torque_reward_w"]
        self._srl_load_cell_w = self.cfg["env"]["srl_load_cell_w"]
        self._srl_root_force_reward_w = self.cfg["env"]["srl_root_force_reward_w"]
        self._srl_feet_slip_w = self.cfg["env"]["srl_feet_slip_w"]
        self._srl_endpos_obs = self.cfg["env"]["srl_endpos_obs"]
        self._target_v_task = self.cfg["env"]["target_v_task"]
        self._autogen_model = self.cfg["env"].get("autogen_model", False)
        self._design_param_obs = self.cfg["env"].get("design_param_obs", False)
        self._load_cell_activate = self.cfg["env"].get("load_cell",False)
        self._humanoid_load_cell_obs = self.cfg["env"].get("humanoid_load_cell_obs", False)
        self._srl_partial_obs = self.cfg["env"].get("srl_partial_obs", False)
        self.srl_default_joint_angles = [0*np.pi,
                                     0*np.pi, 
                                     0.15*np.pi,
                                     0.25*np.pi,
                                     0*np.pi,
                                     0.15*np.pi,
                                     0.25*np.pi]
        self.initial_dof_pos = torch.tensor(self.default_joint_angles, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.obs_scales={
            "lin_vel" : 2.0,
            "ang_vel" : 0.25,
            "dof_pos" : 1.0,
            "dof_vel" : 0.05,
            "height_measurements" : 5.0 }
        self.obs_scales_tensor = torch.tensor([
        self.obs_scales["lin_vel"],
        self.obs_scales["ang_vel"],
        self.obs_scales["dof_pos"],
        self.obs_scales["dof_vel"],
        ], device=self.device)
        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # --- SRL-Gym Defined End ---

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()
   
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        # --- srl defined ---
        # 28 为被动链接关节
        self._srl_joint_ids = to_torch([ 29, 30, 31, 32, 33, 34], device=self.device, dtype=torch.long)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # --- srl defined end ---

        # mirror matrix
        self.mirror_idx_humanoid = np.array([-0.0001, 1, -2, -3, 4, -5, -10, 11, -12,  13, -6,
                                                 7, -8, 9, -21, 22, -23, 24, -25, 26, -27, -14, 
                                                 15, -16, 17, -18, 19, -20,])
        self.mirror_idx_srl = np.array([28,-32,33,34,-29,30,31])
        self.mirror_idx = np.concatenate((self.mirror_idx_humanoid, self.mirror_idx_srl))
        obs_dim = self.mirror_idx.shape[0]
        self.mirror_mat = torch.zeros((obs_dim, obs_dim), dtype=torch.float32, device=self.device)
        for i, perm in enumerate(self.mirror_idx):
            self.mirror_mat[i, int(abs(perm))] = np.sign(perm)
        self.mirror_idx_act_srl = np.array([0.01, -4, 5, 6, -1, 2, 3, ])
        self.mirror_act_srl_mat = torch.zeros((self.mirror_idx_act_srl.shape[0], self.mirror_idx_act_srl.shape[0]), dtype=torch.float32, device=self.device)
        for i, perm in enumerate(self.mirror_idx_act_srl):
            self.mirror_act_srl_mat[i, int(abs(perm))] = np.sign(perm)
         
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) #  State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 4
        if self._load_cell_activate:
            sensors_per_env += 1
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
        self._initial_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        self._initial_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi

        # MLY: set SRL init pos
        self.srl_joint_r1_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0],'SRL_joint_right_hipjoint_y')
        self.srl_joint_r3_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0],'SRL_joint_right_kneejoint')
        self.srl_joint_l1_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0],'SRL_joint_left_hip_y')
        self.srl_joint_l3_idx = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0],'SRL_joint_left_kneejoint')

        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self.srl_rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.rew_joint_cost_buf = torch.zeros(      # 关节力矩惩罚
            self.num_envs, device=self.device, dtype=torch.float)
        self.rew_v_pen_buf = torch.zeros(           # 速度惩罚
            self.num_envs, device=self.device, dtype=torch.float)
        self.rew_upper_buf = torch.zeros(           # 直立惩罚
            self.num_envs, device=self.device, dtype=torch.float)
        self.obs_mirrored_buf = torch.zeros(
            (self.num_envs, self.get_srl_obs_size()), device=self.device, dtype=torch.float)
        if self._design_param_obs:
            design_param = self._get_design_param()
        self.observation_space

        if self.viewer != None:
            self._init_camera()
            
        return
    
    def _get_design_param(self,):
        first_leg_lenth = self.cfg["env"]["design_params"]["first_leg_lenth"]
        first_leg_size = self.cfg["env"]["design_params"]["first_leg_size"]
        second_leg_lenth = self.cfg["env"]["design_params"]["second_leg_lenth"]
        second_leg_size = self.cfg["env"]["design_params"]["second_leg_size"]
        end_size = self.cfg["env"]["design_params"]["third_leg_size"]
        self.design_param = torch.tensor([first_leg_lenth,first_leg_size,second_leg_lenth,second_leg_size,end_size],
                                         device=self.device,
                                         dtype = torch.float32,
                                         )
        return self.design_param 

    def get_obs_size(self):
        obs_size = NUM_OBS
        if self._srl_partial_obs:
            obs_size = NUM_HUMANOID_OBS + 38
        if self._srl_endpos_obs:
            obs_size = obs_size + 6
        if self._target_v_task:
            obs_size = obs_size + 2
        if self._design_param_obs:
            if self.cfg['env']['design_params']['mode'] == 'mode1':
                design_param_num = 5
            obs_size = obs_size + design_param_num
        return obs_size

    def get_action_size(self):
        return NUM_ACTIONS
    
    def get_humanoid_action_size(self):
        return NUM_HUMANOID_ACTIONS

    def get_srl_action_size(self):
        return NUM_ACTIONS - NUM_HUMANOID_ACTIONS

    def get_humanoid_obs_size(self):
        return NUM_HUMANOID_OBS
    
    def get_srl_obs_size(self):
        if self._srl_partial_obs:
            return 38
        else:
            return NUM_OBS - NUM_HUMANOID_OBS

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
        asset_file = "mjcf/amp_humanoid_srl.xml"

        if "asset" in self.cfg["env"]:
            #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)
            print('Asset file name:'+asset_file)
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Add actuator list
        self._dof_names = self.gym.get_asset_dof_names(humanoid_asset)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        right_srl_end_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL_right_end")
        left_srl_end_idx  = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL_left_end")
        sensor_pose = gymapi.Transform()

        self.right_foot_ssidx = self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.left_foot_ssidx = self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        if self._load_cell_activate:
            # 添加背板处Load Cell传感器
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = False
            sensor_props.enable_constraint_solver_forces = True
            sensor_props.use_world_frame = False
            load_cell_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL")
            self.load_cell_ssidx = self.gym.create_asset_force_sensor(humanoid_asset, load_cell_idx, sensor_pose, sensor_props)
        
        srl_end_sensor_props = gymapi.ForceSensorProperties()
        srl_end_sensor_props.enable_forward_dynamics_forces = True
        srl_end_sensor_props.enable_constraint_solver_forces = True
        srl_end_sensor_props.use_world_frame = True
        self.right_srl_end_ssidx = self.gym.create_asset_force_sensor(humanoid_asset, right_srl_end_idx, sensor_pose, srl_end_sensor_props)
        self.left_srl_end_ssidx = self.gym.create_asset_force_sensor(humanoid_asset, left_srl_end_idx, sensor_pose, srl_end_sensor_props)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof    = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose   = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
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
            
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, contact_filter, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            if (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
            #print(self.gym.get_actor_dof_names(env_ptr,handle))

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
        self._upper_body_ids = self._build_upper_body_ids_tensor(env_ptr, handle)
        self._srl_root_body_ids = self._build_srl_root_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)
        
        self._srl_end_ids = self._build_srl_end_body_ids_tensor(env_ptr, handle)

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return

    def get_task_target_v(self, env_ids=None):
        # 0:X轴正方向, 1:Y轴正方向
        task_v = torch.tensor([[0,1]] * 100 + [[1,0]] * 100 +  [[0,1]] * 101,device=self.device)
        target_velocity = task_v[self.progress_buf,:]
        if env_ids is None:
            return target_velocity
        else:
            return target_velocity[env_ids]

        
    def reset_done(self):
        _, done_env_ids = super().reset_done()
        # 添加镜像OBS
        self.obs_dict["obs_mirrored"] = torch.clamp(self.obs_mirrored_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        return self.obs_dict, done_env_ids
        
    def _build_pd_action_offset_scale(self):
        Dof_offsets = DOF_OFFSETS
        for i in range(DOF_OFFSETS[-1], self.dof_limits_lower.shape[0]):  # 补齐
            Dof_offsets.append(i+1)
            
        num_joints = len(Dof_offsets) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = Dof_offsets[j]
            dof_size = Dof_offsets[j + 1] - Dof_offsets[j]

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale  = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale  = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):
        upper_body_pos = self._rigid_body_pos[:, self._upper_body_ids, :]
        srl_root_body_pos = self._rigid_body_pos[:, self._srl_root_body_ids, :]
        load_cell_sensor = self.vec_sensor_tensor[:,self.load_cell_ssidx,:]
        srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :]
        srl_end_body_vel = self._rigid_body_vel[:, self._srl_end_ids, :]
        srl_feet_slip = compute_srl_feet_slip(srl_end_body_pos, srl_end_body_vel)
        self.rew_buf[:], self.rew_v_pen_buf[:], self.rew_joint_cost_buf[:], self.rew_upper_buf[:] = compute_humanoid_reward(self.obs_buf, 
                                                                                                     self.dof_force_tensor, 
                                                                                                     self._contact_forces,
                                                                                                     actions,
                                                                                                     self._torque_threshold,
                                                                                                     srl_root_body_pos,
                                                                                                     self._upper_reward_w,
                                                                                                     self._srl_joint_ids,
                                                                                                     load_cell_sensor,
                                                                                                     srl_feet_slip,
                                                                                                     target_v_task = self._target_v_task,
                                                                                                     srl_torque_w  = self._srl_torque_reward_w, 
                                                                                                     srl_load_cell_w = self._srl_load_cell_w,
                                                                                                     srl_feet_slip_w = self._srl_feet_slip_w)
        self.srl_rew_buf[:] = compute_srl_reward(self.obs_buf, self.dof_force_tensor, actions)
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
        obs, obs_mirrored = self._compute_humanoid_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
            self.obs_mirrored_buf[:] = obs_mirrored
        else:
            self.obs_buf[env_ids] = obs  
            self.obs_mirrored_buf[env_ids] = obs_mirrored

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            load_cell_sensor = self.vec_sensor_tensor[:,self.load_cell_ssidx,:]
            if self._srl_endpos_obs: # Add cartisian pos of SRL-end to OBS
                srl_end_body_pos = self._rigid_body_pos[:,self._srl_end_ids, :]
                key_body_pos = torch.cat((key_body_pos, srl_end_body_pos), dim=1)
            target_v = self.get_task_target_v() # Target speed
            dof_force_tensor = self.dof_force_tensor
            progress_buf = self.progress_buf
            initial_dof_pos = self.initial_dof_pos
            actions = self.actions
            gravity_vec = self.gravity_vec 
            targets = self.targets
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
            load_cell_sensor = self.vec_sensor_tensor[env_ids,self.load_cell_ssidx,:]
            if self._srl_endpos_obs:
                srl_end_body_pos = self._rigid_body_pos[env_ids][:,self._srl_end_ids, :]
                key_body_pos = torch.cat((key_body_pos, srl_end_body_pos), dim=1)
            target_v = self.get_task_target_v(env_ids) # Target speed
            dof_force_tensor = self.dof_force_tensor[env_ids]
            progress_buf = self.progress_buf[env_ids]
            initial_dof_pos = self.initial_dof_pos[env_ids]
            actions = self.actions[env_ids]
            gravity_vec = self.gravity_vec[env_ids]
            targets = self.targets[env_ids]

        humanoid_obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs,
                                            load_cell_sensor, self._humanoid_load_cell_obs)
        srl_obs = compute_srl_observations(progress_buf, initial_dof_pos, root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs,
                                            load_cell_sensor, dof_force_tensor, gravity_vec, actions, 
                                            self.obs_scales_tensor, targets, self._srl_partial_obs)
        obs = torch.cat((humanoid_obs, srl_obs),dim=1)
        obs_mirrored = compute_srl_observations_mirrored(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs, 
                                            load_cell_sensor, self.mirror_mat, dof_force_tensor, self._srl_partial_obs)
        if self._target_v_task:
            #target_v= target_v.unsqueeze(1)
            obs = torch.cat((obs, target_v),dim=1)
            obs_mirrored = torch.cat((obs_mirrored, target_v),dim=1)
        if self._design_param_obs:
            design_param = self.design_param
            design_param = design_param.unsqueeze(0).repeat(obs.shape[0], 1)
            obs = torch.cat([obs, design_param], dim=1)
            obs_mirrored = torch.cat([obs_mirrored, design_param], dim=1)
        return obs, obs_mirrored

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

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions) # pd_tar.shape: [num_actors, num_dofs]
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
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
        
        self.extras["terminate"] = self._terminate_buf

        # SRL reward
        self.extras["srl_rewards"] = self.srl_rew_buf.to(self.rl_device)
        self.extras["v_penalty"] = self.rew_v_pen_buf.to(self.rl_device)         # Reward: velocity penalty
        self.extras["torque_cost"] = self.rew_joint_cost_buf.to(self.rl_device)  # Reward: torque cost
        self.extras["upper_reward"] = self.rew_upper_buf.to(self.rl_device)      # Reward: upper reward
        self.extras["x_velocity"] = self.obs_buf[:,7]                            
        self.extras["dof_forces"] = self.dof_force_tensor.to(self.rl_device)
        self.extras["obs_mirrored"] = self.obs_mirrored_buf.to(self.rl_device)  # 镜像观测
        
        srl_torque_cost = self.SRL_joint_torque_cost()
        self.extras["srl_torque_cost"] = srl_torque_cost.to(self.rl_device)
        # plotting
        self.extras["root_pos"] = self._root_states[0, 0:3].to(self.rl_device)
        srl_end_body_pos = self._rigid_body_pos[0, self._srl_end_ids, :]
        srl_end_body_vel = self._rigid_body_vel[0, self._srl_end_ids, :]
        self.extras['srl_end_pos'] = srl_end_body_pos
        self.extras['srl_end_vel'] = srl_end_body_vel
        key_body_pos = self._rigid_body_pos[0, self._key_body_ids, :]
        self.extras['key_body_pos'] = key_body_pos
        self.extras['dof_pos'] = self._dof_pos[0].to(self.rl_device)
        self.extras['load_cell'] = self.vec_sensor_tensor[0,self.load_cell_ssidx,:].to(self.rl_device)
        self.extras['right_srl_end_sensor'] = self.vec_sensor_tensor[0,self.right_srl_end_ssidx,:].to(self.rl_device)

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def SRL_joint_torque_cost(self):
        joint_forces = self.dof_force_tensor[:, self._srl_joint_ids]
        torque_sum = - torch.sum((joint_forces/100) ** 2, dim=1).to(self.rl_device)
        return torque_sum

    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render()
        return

    def _build_upper_body_ids_tensor(self, env_ptr, actor_handle):
        # get id of upper body 
        # used to calculate the balance reward
        body_ids = []
        for body_name in UPPER_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def _build_srl_root_body_ids_tensor(self, env_ptr, actor_handle):
        # get id of upper body 
        # used to calculate the balance reward
        body_ids = []
        for body_name in SRL_ROOT_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_srl_end_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        if self._autogen_model:
            srl_end_body_names = ["SRL_right_end","SRL_left_end"] 
        else:
            srl_end_body_names = SRL_END_BODY_NAMES
        
        for body_name in srl_end_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []      
        # 添加外肢体碰撞模型 
        contact_body = self._contact_bodies
        if self._autogen_model:
            contact_body = contact_body + ['SRL_root', 'SRL_right_leg1', 'SRL_right_leg2', 'SRL_right_end', 'SRL_left_leg1', 'SRL_left_leg2', 'SRL_left_end']
        else:
            contact_body = contact_body + SRL_CONTACT_BODY_NAMES
        for body_name in contact_body:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1, f'No agent-body named: {body_name}'
            body_ids.append(body_id)

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
    
    def restart_sim(self,):
        self.gym.destroy_sim(self.sim)
        if self.viewer != None:
            self.gym.destroy_viewer(self.viewer)
        # create envs, sim and viewer
        self.gym = gymapi.acquire_gym()
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True
        
        self.set_viewer()


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28] # humanoid 
    
    for i in range(dof_offsets[-1], pose.shape[-1]):  # 补齐SRL
        dof_offsets.append(i+1)
        dof_obs_size += 1

    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
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

    root_h = root_pos[:, 2:3] # root高度
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs) # root朝向

    local_root_vel = my_quat_rotate(heading_rot, root_vel) # 局部root速度
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel) # 局部根部角速度

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

    humanoid_dof_obs = dof_to_obs(dof_pos[:,0:28]) # 仅保留humanoid关节位置
    humanoid_dof_vel = dof_vel[:,0:28]
    # root_h 1; root_rot_obs 6; local_root_vel 3 ; local_root_ang_vel 3 ; dof_obs 58; dof_vel 36 ; load_cell_force 6, flat_local_key_pos 12
    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, humanoid_dof_obs, humanoid_dof_vel, load_cell_force, flat_local_key_pos), dim=-1)
    return obs


# @torch.jit.script
def compute_srl_observations(
            progress_buf,
            default_joint_pos,
            root_states, 
            dof_pos, 
            dof_vel, 
            key_body_pos, 
            local_root_obs, 
            load_cell, 
            dof_force_tensor, 
            gravity_vec ,
            actions,
            obs_scales,
            targets,
            srl_partial_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3] # root高度

    heading_rot = calc_heading_quat_inv(root_rot)
    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs) # root朝向

    # 将线速度/角速度旋转到局部坐标
    local_root_vel     = quat_rotate_inverse(root_rot, root_vel)
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    projected_gravity  = quat_rotate_inverse(root_rot, gravity_vec)

    #  humanoid key-body position
    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    load_cell_force = - load_cell

    dof_obs = dof_to_obs(dof_pos) # dof_pos 34
    
    dof_vel[:,28] = 0.00   # SRL连接处被动自由关节不考虑

    # Y轴free joint仅保留角度信息
    srl_dof_obs = dof_pos[:, 28:] - default_joint_pos # 7D 仅保留srl关节位置
    srl_dof_vel = dof_vel[:, 29:] # 6D 仅保留srl关节速度
    srl_dof_force = dof_force_tensor[:,29:]


    # 计算 heading_proj 默认机器人面朝 x 轴 [1, 0, 0]
    heading_vector = torch.tensor([1., 0., 0.], device=root_rot.device).repeat(root_rot.shape[0], 1)
    facing_dir = - my_quat_rotate(root_rot, heading_vector)   #  MLY: 在定义SRL时，对ROOT进行了180度旋转，故此处取负
    to_target = targets - root_pos
    to_target[:, 2] = 0
    to_target_dir = torch.nn.functional.normalize(to_target, dim=-1)
    cos_theta = torch.sum(facing_dir[:, :2] * to_target_dir[:, :2], dim=-1, keepdim=True)  # [-1, 1]
    cos_theta_clipped = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    angle_error = torch.acos(cos_theta_clipped)  # ∈ [0, π]
    
    # 假设周期为 T=60 步，则频率为 1/T，每一步是 2π/T 相位步长
    phase_t = (2 * math.pi / 15.0) * (progress_buf-1).float()  # shape: [num_envs]
    sin_phase = torch.sin(phase_t).unsqueeze(-1)
    cos_phase = torch.cos(phase_t).unsqueeze(-1)
    
    obs = torch.cat((root_h,                         # 1
                    local_root_vel * obs_scales[0], # 3
                    local_root_ang_vel * obs_scales[1], # 3
                    projected_gravity,                  # 3 
                    srl_dof_obs * obs_scales[2],        # 6
                    srl_dof_vel * obs_scales[3],        # 6
                    actions , # actions 通常不用 scale      6
                    angle_error,                       # 1       
                    sin_phase,                      # 1
                    cos_phase,                      # 1
                ), dim=-1)
    return obs

# @torch.jit.script
def compute_srl_observations_mirrored( 
            progress_buf,
            default_joint_pos,
            root_states, 
            dof_pos, 
            dof_vel, 
            key_body_pos, 
            local_root_obs, 
            load_cell, 
            dof_force_tensor, 
            gravity_vec ,
            actions,
            obs_scales,
            targets,
            srl_partial_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3] # root高度

    heading_rot = calc_heading_quat_inv(root_rot)
    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs) # root朝向

    # 将线速度/角速度旋转到局部坐标
    local_root_vel     = quat_rotate_inverse(root_rot, root_vel)
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)
    projected_gravity  = quat_rotate_inverse(root_rot, gravity_vec)

    #  humanoid key-body position
    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    load_cell_force = - load_cell

    dof_obs = dof_to_obs(dof_pos) # dof_pos 34
    
    dof_vel[:,28] = 0.00   # SRL连接处被动自由关节不考虑

    # Y轴free joint仅保留角度信息
    srl_dof_obs = dof_pos[:, 28:] - default_joint_pos # 7D 仅保留srl关节位置
    srl_dof_vel = dof_vel[:, 29:] # 6D 仅保留srl关节速度
    srl_dof_force = dof_force_tensor[:,29:]


    # 计算 heading_proj 默认机器人面朝 x 轴 [1, 0, 0]
    heading_vector = torch.tensor([1., 0., 0.], device=root_rot.device).repeat(root_rot.shape[0], 1)
    facing_dir = - my_quat_rotate(root_rot, heading_vector)   #  MLY: 在定义SRL时，对ROOT进行了180度旋转，故此处取负
    to_target = targets - root_pos
    to_target[:, 2] = 0
    to_target_dir = torch.nn.functional.normalize(to_target, dim=-1)
    cos_theta = torch.sum(facing_dir[:, :2] * to_target_dir[:, :2], dim=-1, keepdim=True)  # [-1, 1]
    cos_theta_clipped = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    angle_error = torch.acos(cos_theta_clipped)  # ∈ [0, π]
    
    # 假设周期为 T=60 步，则频率为 1/T，每一步是 2π/T 相位步长
    phase_t = (2 * math.pi / 15.0) * (progress_buf-1).float()  # shape: [num_envs]
    sin_phase = torch.sin(phase_t).unsqueeze(-1)
    cos_phase = torch.cos(phase_t).unsqueeze(-1)
    


    # Mirror
    root_rot_obs[:,1] =  -root_rot_obs[:,1]  # 切向量
    root_rot_obs[:,4] =  -root_rot_obs[:,4]  # 法向量
    local_root_vel[:,1] = -local_root_vel[:,1] # y方向速度
    local_root_ang_vel[:,0] = -local_root_ang_vel[:,0] # x轴角速度
    local_root_ang_vel[:,2] = -local_root_ang_vel[:,2] # z轴角速度
    mirrored_dof_pos = torch.matmul(dof_pos, mirror_mat) # Perform the matrix multiplication to get mirrored dof_pos
    dof_obs = dof_to_obs(mirrored_dof_pos) # dof_pos 36
    dof_vel = torch.matmul(dof_vel, mirror_mat)
    dof_force = torch.matmul(dof_force_tensor, mirror_mat)

    flat_local_key_pos_mirror = flat_local_key_pos.clone()
    # right/left hand
    flat_local_key_pos_mirror[:,0:3] = flat_local_key_pos[:,3:6]
    flat_local_key_pos_mirror[:,3:6] = flat_local_key_pos[:,0:3]
    # right/left foot
    flat_local_key_pos_mirror[:,6:9] = flat_local_key_pos[:,9:12]
    flat_local_key_pos_mirror[:,9:12] = flat_local_key_pos[:,6:9]
    # right/left SRL end
    flat_local_key_pos_mirror[:,12:15] = flat_local_key_pos[:,15:18]
    flat_local_key_pos_mirror[:,15:18] = flat_local_key_pos[:,12:15]
    
    load_cell_mirror = -load_cell
    load_cell_mirror[:,1] = -load_cell_mirror[:,1]  # y轴速度
    load_cell_mirror[:,3] = -load_cell_mirror[:,3]  # x轴角速度
    load_cell_mirror[:,5] = -load_cell_mirror[:,5]  # z轴角速度

    # Y轴free joint仅保留角度信息
    srl_dof_obs = dof_pos[:, 28:] # 7D 仅保留srl关节位置
    srl_dof_vel = dof_vel[:, 29:] # 6D 仅保留srl关节速度
    srl_dof_force = dof_force[:,29:]
    srl_dof_force = srl_dof_force[:, [3, 4, 5, 0, 1, 2]] # mirrored 

    obs = torch.cat((root_h,                         # 1
                    local_root_vel * obs_scales[0], # 3
                    local_root_ang_vel * obs_scales[1], # 3
                    projected_gravity,                  # 3 
                    srl_dof_obs * obs_scales[2],        # 6
                    srl_dof_vel * obs_scales[3],        # 6
                    actions , # actions 通常不用 scale      6
                    angle_error,                       # 1       
                    sin_phase,                      # 1
                    cos_phase,                      # 1
                ), dim=-1)
    return obs



# 计算任务奖励函数
@torch.jit.script
def compute_humanoid_reward(obs_buf, 
                            dof_force_tensor, 
                            contact_buf,  # body net contact force
                            action, 
                            _torque_threshold, 
                            upper_body_pos, 
                            upper_reward_w, 
                            srl_joint_ids,
                            srl_load_cell_sensor,
                            srl_feet_slip,
                            target_v_task = False,
                            srl_torque_w = 0.0,
                            srl_load_cell_w  = 0.0 ,
                            srl_feet_slip_w = 0.0):
    # type: (Tensor, Tensor, Tensor, Tensor, int, Tensor, int, Tensor, Tensor, Tensor, bool, float, float , float ) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
    # TODO: 目标速度跟随
    velocity_threshold = 1.4
    if not target_v_task:  # 速度惩罚
        velocity  = obs_buf[:,7]  # vx
        vy  = obs_buf[:,8]  # vy
        velocity_penalty = - 20 * (vy**2) - torch.where(velocity < velocity_threshold, (velocity_threshold - velocity)**2, torch.zeros_like(velocity))
    else:                  # 运动方向
        # velocity_x  = obs_buf[:,7]  # vx
        # velocity_y  = obs_buf[:,8]  # vy
        # velocity = torch.sqrt(velocity_x**2 + velocity_y**2)
        velocity  = obs_buf[:,7]  # vx
        v_penalty = - torch.where(velocity < velocity_threshold, (velocity_threshold - velocity)**2, torch.zeros_like(velocity))
        direction = obs_buf[:,1:3]  # [x,y]
        norm_direction = direction / torch.norm(direction, p=2, dim=1, keepdim=True)
        target_direction = obs_buf[:,-2:] # [x,y]
        # d_penalty = -1+torch.sum(norm_direction * target_direction, dim=1)
        d_penalty = -(-1+torch.sum(norm_direction * target_direction, dim=1))**2

        velocity_penalty = d_penalty

    
    
    # v1.5.12 比例惩罚，humanoid力矩绝对值超过100
    torque_threshold = _torque_threshold
    torque_usage   = dof_force_tensor[:, 14:28]
    torque_penalty = torch.where(torch.abs(torque_usage) > torque_threshold, 
                                 (torch.abs(torque_usage) - torque_threshold) / torque_threshold, 
                                 torch.zeros_like(torque_usage))
    torque_reward  = - torch.sum(torque_penalty, dim=1)
    # MLY: 暂时关闭HUMANOID受力惩罚
    torque_reward = torque_reward * 0

    # 外肢体水平奖励项
    board_pos = upper_body_pos[:, 0, :]  # (4096, 3)
    root_pos = upper_body_pos[:, 1, :]  # (4096, 3)
    upper_body_direction = board_pos - root_pos  # 维度 (4096, 3)
    norm_upper_body_direction = upper_body_direction / torch.norm(upper_body_direction, dim=1, keepdim=True)
    # upper_reward = upper_reward_w * (norm_upper_body_direction[:,2] - 1 )
    upper_reward = - upper_reward_w * (torch.abs(norm_upper_body_direction[:,2]) )

    # SRL 关节力矩惩罚
    srl_joint_forces = dof_force_tensor[:,  srl_joint_ids]
    srl_torque_sum = - torch.sum((srl_joint_forces/100) ** 2, dim=1)
    srl_torque_reward = srl_torque_sum * srl_torque_w
    # MLY: 暂时关闭SRL受力惩罚
    srl_torque_reward = 0


    # SRL Root受力惩罚
    load_cell_z = srl_load_cell_sensor[:,2] # 原始数据为正
    load_cell_y = srl_load_cell_sensor[:,1]
    load_cell_x = srl_load_cell_sensor[:,0] 
    scaled_z    = torch.clamp(torch.abs(load_cell_z), min=50, max=2500)  # 限制受力范围
    scaled_y    = torch.clamp(torch.abs(load_cell_y), min=50, max=2500) 
    scaled_x    = torch.clamp(torch.abs(load_cell_x), min=50, max=2500)
    z_penalty   = ((scaled_z - 50) / 50) ** 2 # 平方
    y_penalty   = ((scaled_y - 50) / 50) ** 2  
    x_penalty   = ((scaled_x - 50) / 50) ** 2  
    # z_penalty =  torch.log(1.0 + (scaled_z - 100) / 100.0) / 3.0  # 对数
    srl_load_cell_reward = -srl_load_cell_w * (z_penalty + y_penalty + x_penalty)

    # 末端滑动惩罚 feet slip
    srl_feet_slip_reward = - srl_feet_slip_w * srl_feet_slip.squeeze(1)


    # scaled_x = torch.clamp(load_cell_x ,min=0)
    # x_penalty = scaled_x / 1000.0  # ~1 if x=1000
    # load_cell_penalty =  z_penalty    
    # srl_load_cell_reward = -load_cell_penalty * srl_load_cell_w       

    # reward = -velocity_penalty + torque_reward
    reward = velocity_penalty + torque_reward + upper_reward + srl_torque_reward + srl_load_cell_reward + srl_feet_slip_reward
    
    # return reward, velocity_penalty, torque_reward, upper_reward
    return reward, velocity_penalty, srl_load_cell_reward, upper_reward

# 计算外肢体奖励函数
@torch.jit.script
def compute_srl_reward(obs_buf, dof_force_tensor, action):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    # 14-28 包括髋关节+膝关节+踝关节
    torque_usage =  torch.sum(action[:,14:28] ** 2, dim=1)
    # v1.2.1力矩使用惩罚（假设action代表施加的力矩）
    torque_reward = - 0.1 *  torque_usage # 惩罚力矩的平方和
    # v1.2.2指数衰减
    # torque_reward = torch.exp(-0.1 * torque_usage)  # 指数衰减，0.1为衰减系数
    #r = 0*reward + velocity_reward - torque_cost
    r =  torque_reward
    return r

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    # 提前终止
    if (enable_early_termination):
        # contact_buf: shape[self.num_envs, self.num_bodies, 3](acquire_net_contact_force_tensor)
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1) # 在每个env中，判断是否有body发生碰撞
        fall_contact = torch.any(fall_contact, dim=-1) # 找出发生碰撞的env

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height # 在每个env中，判断是否有body低于阈值高度
        fall_height[:, contact_body_ids] = False 
        fall_height = torch.any(fall_height, dim=-1) # 找出低于阈值高度的env

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def compute_srl_feet_slip(srl_end_pos, srl_end_vel):
    # type: (Tensor, Tensor ) -> Tensor

    # srl_end_pos: [num_envs, 2, 3]
    # srl_end_vel: [num_envs, 2, 3]
    num_envs = srl_end_pos.size(0)

    # 判断是否接触地面（z < 0.9）
    foot_contact = srl_end_pos[..., 2] < 0.095  # [num_envs, 2]，bool tensor

    # 计算xy平面速度的合速度
    foot_vel_mag = torch.sqrt(srl_end_vel[..., 0] ** 2 + srl_end_vel[..., 1] ** 2)  # [num_envs, 2]

    # 判断是否滑动（速度 > 0.05 且接触地面）
    foot_slip = (foot_contact & (foot_vel_mag > 0.05))  # [num_envs, 2]

    # 若任一脚滑动，则该env为滑动状态
    slip_flag = foot_slip.any(dim=1).float().unsqueeze(-1)  # [num_envs, 1]

    return slip_flag