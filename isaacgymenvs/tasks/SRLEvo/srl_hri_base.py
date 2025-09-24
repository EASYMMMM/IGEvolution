import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
import math
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, get_axis_params, calc_heading_quat_inv, \
     exp_map_to_quat, quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv, quat_rotate_inverse

from ..base.vec_task import VecTask
from isaacgymenvs.learning.SRLEvo.srl_models import ModelSRLContinuous 
from isaacgymenvs.learning.SRLEvo.srl_network_builder import SRLBuilder 

DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]

SRL_ROOT_BODY_NAMES = ["SRL", "SRL_root"]
UPPER_BODY_NAMES = ["pelvis", "torso"]
KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]  # body end + SRL end
SRL_END_BODY_NAMES = ["SRL_right_end","SRL_left_end"] 
SRL_CONTACT_BODY_NAMES = ['SRL_root', 'SRL_leg2', 'SRL_shin11', 'SRL_right_end', 'SRL_leg1', 'SRL_shin1', 'SRL_left_end']

class SRL_HRIBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self._force_control = self.cfg["env"]["forceControl"]
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
        self._srl_termination_height = self.cfg["env"]["SRLterminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        # --- env  ---
        self.humanoid_obs_num = self.cfg["env"]["humanoid_obs_num"]
        self.humanoid_actions_num = self.cfg["env"]["humanoid_actions_num"]
        self.srl_obs_num = self.cfg["env"]["srl_obs_num"]
        self.srl_actions_num = self.cfg["env"]["srl_actions_num"]
        self.srl_obs_frame_stack = self.cfg["env"].get("srl_obs_frame_stack", 5)
        self.srl_command_num = self.cfg["env"]["srl_command_num"]
        self.gait_period = self.cfg["env"]["gait_period"]
        self.foot_clearance = self.cfg["env"]["foot_clearance"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.srl_termination_height = self.cfg["env"]["srlTerminationHeight"]

        # --- reward ---
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

        self._torque_threshold = self.cfg["env"]["torque_threshold"]
        self._upper_reward_w = self.cfg["env"]["upper_reward_w"]
        self._srl_torque_reward_w = self.cfg["env"]["srl_torque_reward_w"]
        self._srl_load_cell_w = self.cfg["env"]["srl_load_cell_w"]
        self._srl_root_force_reward_w = self.cfg["env"]["srl_root_force_reward_w"]
        self._srl_feet_slip_w = self.cfg["env"]["srl_feet_slip_w"]
        self._srl_endpos_obs = self.cfg["env"]["srl_endpos_obs"]
        self._autogen_model = self.cfg["env"].get("autogen_model", False)
        self._design_param_obs = self.cfg["env"].get("design_param_obs", False)
        self._load_cell_activate = self.cfg["env"].get("load_cell",False)
        self._humanoid_load_cell_obs = self.cfg["env"].get("humanoid_load_cell_obs", False)
        self._srl_partial_obs = self.cfg["env"].get("srl_partial_obs", False)
       
        # self.initial_dof_pos = torch.tensor(self.srl_default_joint_angles, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.obs_scales={
            "lin_vel" : 1,
            "ang_vel" : 1,
            "dof_pos" : 1.0,
            "dof_vel" : 0.05,
            "height_measurements" : 5.0 }
     
        # --- SRL-Gym Defined End ---

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.default_srl_joint_angles = [0*np.pi, 
                                         0.28*np.pi,
                                         0*np.pi,
                                         0*np.pi,
                                         0.28*np.pi,
                                         0*np.pi]
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        # --- srl defined ---
        # 28 为被动链接关节
        self._srl_joint_ids = to_torch([ 29, 30, 31, 32, 33, 34], device=self.device, dtype=torch.long)
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
        self.phase_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.target_ang_vel_z = torch.zeros(self.num_envs, device=self.device)
        self.target_pelvis_height = torch.full((self.num_envs,), 0.83, device=self.device)
        self.target_vel_x = torch.full((self.num_envs,), 0.0, device=self.device)

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "right_shoulder_x")
        left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "left_shoulder_x")
        self._initial_dof_pos[:, right_shoulder_x_handle] = 0 * np.pi
        self._initial_dof_pos[:, left_shoulder_x_handle] = 0 * np.pi

        self.initial_srl_dof_pos = torch.tensor(self.default_srl_joint_angles, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self._initial_dof_pos[:, -6:] = self.initial_srl_dof_pos
        
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
        
        self.srl_root_states = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.srl_root_index ,  :]
        self.srl_rotate_root_states = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.srl_rotate_root_index ,  :]
        self.prev_srl_end_body_pos = torch.zeros((self.num_envs,2,3), device=self.device)

        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self.srl_obs_buf = torch.zeros(
            (self.num_envs, self.get_srl_obs_size()), device=self.device, dtype=torch.float)

        self.srl_rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.rew_joint_cost_buf = torch.zeros(      # 关节力矩惩罚
            self.num_envs, device=self.device, dtype=torch.float)
        self.rew_v_pen_buf = torch.zeros(           # 速度惩罚
            self.num_envs, device=self.device, dtype=torch.float)
        self.rew_upper_buf = torch.zeros(           # 直立惩罚
            self.num_envs, device=self.device, dtype=torch.float)
        self.srl_obs_mirrored_buf = torch.zeros(
            (self.num_envs, self.get_srl_obs_size()), device=self.device, dtype=torch.float)
        if self._design_param_obs:
            design_param = self._get_design_param()

        self.srl_obs_buffer = torch.zeros((self.num_envs, self.srl_obs_frame_stack, self.srl_obs_num), device=self.device)
        self.srl_obs_mirrored_buffer = torch.zeros((self.num_envs, self.srl_obs_frame_stack, self.srl_obs_num), device=self.device)

        # self.observation_space
        self.obs_scales_tensor = torch.tensor([
        self.obs_scales["lin_vel"],
        self.obs_scales["ang_vel"],
        self.obs_scales["dof_pos"],
        self.obs_scales["dof_vel"],
        ], device=self.device)
        
        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

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
        obs_size = self.get_humanoid_obs_size() + self.get_srl_obs_size()
        return obs_size

    def get_action_size(self):
        action_size = self.humanoid_actions_num + self.srl_actions_num
        return action_size
    
    def get_humanoid_action_size(self):
        return self.humanoid_actions_num

    def get_srl_action_size(self):
        return self.srl_actions_num

    def get_teacher_srl_action_size(self):
        return self.srl_actions_num - 1

    def get_humanoid_obs_size(self):
        return self.humanoid_obs_num
    
    def get_srl_obs_size(self):
        return self.srl_obs_num*self.srl_obs_frame_stack + self.srl_command_num

    def get_teacher_srl_obs_size(self):
        return 153

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
        
        self.srl_root_index = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL_root", )
        self.srl_rotate_root_index = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL", )

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
        start_pose.p = gymapi.Vec3(*get_axis_params(0.86, self.up_axis_idx))
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
            
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            if (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            if self._force_control:
                srl_start_id = self.get_humanoid_action_size()
                dof_prop[srl_start_id:]["stiffness"].fill(0.0)
                dof_prop[srl_start_id:]["damping"].fill(0.0)
                dof_prop[srl_start_id:]["velocity"].fill(14.0)
                dof_prop[srl_start_id:]["effort"].fill(200.0)
                dof_prop[srl_start_id:]["driveMode"] = gymapi.DOF_MODE_EFFORT
                dof_prop[srl_start_id]["driveMode"] = gymapi.DOF_MODE_NONE
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
        #FIXME: srl dof range
        if self._force_control:
            self.dof_limits_lower[srl_start_id+1+1] = self.default_srl_joint_angles[1] - 45/180*np.pi
            self.dof_limits_lower[srl_start_id+1+2] = self.default_srl_joint_angles[2] - 45/180*np.pi
            self.dof_limits_lower[srl_start_id+1+4] = self.default_srl_joint_angles[4] - 45/180*np.pi
            self.dof_limits_lower[srl_start_id+1+5] = self.default_srl_joint_angles[5] - 45/180*np.pi
            self.dof_limits_upper[srl_start_id+1+1] = self.default_srl_joint_angles[1] + 45/180*np.pi
            self.dof_limits_upper[srl_start_id+1+2] = self.default_srl_joint_angles[2] + 45/180*np.pi
            self.dof_limits_upper[srl_start_id+1+4] = self.default_srl_joint_angles[4] + 45/180*np.pi
            self.dof_limits_upper[srl_start_id+1+5] = self.default_srl_joint_angles[5] + 45/180*np.pi
                
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._upper_body_ids = self._build_upper_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)
        
        self._srl_end_ids = self._build_srl_end_body_ids_tensor(env_ptr, handle)


        self._build_pd_action_offset_scale()

        return
        
    def reset_done(self):
        _, done_env_ids = super().reset_done()
        # 添加镜像OBS
        self.obs_dict["obs_mirrored"] = torch.clamp(self.srl_obs_mirrored_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
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
                if dof_offset >= self.get_humanoid_action_size():  # srl
                    curr_scale = 0.45 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_offset[self.get_humanoid_action_size()+1:]=self.default_srl_joint_angles
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)

        self._pd_action_scale  = 0.5 * (lim_high - lim_low)
        self._pd_action_scale  = to_torch(self._pd_action_scale, device=self.device)


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
    
    def _compute_reward(self, actions):
        # srl_root_body_pos = self._rigid_body_pos[:, self._srl_root_body_ids, :]
        load_cell_sensor = self.vec_sensor_tensor[:,self.load_cell_ssidx,:]
        srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :]
        srl_end_body_vel = self._rigid_body_vel[:, self._srl_end_ids, :]
        to_target = self.targets - self._initial_root_states[:, 0:3]
        srl_root_pos = self.srl_root_states[:, 0:3]
        clearance_reward = self.compute_foot_clearance_reward()
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        self.srl_rew_buf[:]  = compute_srl_reward(self.srl_obs_buf[:],
                                            clearance_reward,
                                            to_target,
                                            self.progress_buf,
                                            self.phase_buf,
                                            self.actions[:,-self.srl_actions_num:],
                                            srl_end_body_pos,
                                            srl_root_pos,
                                            self.potentials,
                                            self.prev_potentials,
                                            self.srl_termination_height,
                                            self.death_cost,
                                            self.max_episode_length,
                                            self.gait_period,
                                            srl_obs_num = self.srl_obs_num,
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
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf, self.srl_obs_buf,
                                                   self._dof_pos[:,-self.srl_actions_num:], self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height, self._srl_termination_height)
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
        humanoid_obs, srl_obs, srl_obs_mirrored, potentials, prev_potentials, = self._compute_env_obs(env_ids)

        if (env_ids is None):
            self.srl_obs_buffer[:, 1:, :] = self.srl_obs_buffer[:, :-1, :]  # 向后移动数据
            self.srl_obs_buffer[:, 0, :] = srl_obs  # 将新的观测数据放到队列的开头
            self.srl_obs_mirrored_buffer[:, 1:, :] = self.srl_obs_mirrored_buffer[:, :-1, :]  # 向后移动数据
            self.srl_obs_mirrored_buffer[:, 0, :] = srl_obs_mirrored  # 将新的观测数据放到队列的开头

            srl_base_obs = self.srl_obs_buffer.reshape(self.num_envs, -1)  
            srl_base_mirrored_obs = self.srl_obs_mirrored_buffer.reshape(self.num_envs, -1)
            
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
            srl_stacked_obs = torch.cat([srl_base_obs, task_params], dim=-1)
            srl_mirrored_stacked_obs = torch.cat([srl_base_mirrored_obs, mirrored_task_params], dim=-1)
           
            total_obs = torch.cat((humanoid_obs, srl_stacked_obs),dim=1)
            self.obs_buf[:] = total_obs
            self.srl_obs_buf[:] = srl_stacked_obs
            self.srl_obs_mirrored_buf[:] = srl_mirrored_stacked_obs      

            self.potentials[:] = potentials
            self.prev_potentials[:] = prev_potentials

        else:
            # 对指定环境进行更新 
            self.srl_obs_buffer[env_ids, 1:, :] = self.srl_obs_buffer[env_ids, :-1, :]  
            self.srl_obs_buffer[env_ids, 0, :] = srl_obs  
            self.srl_obs_mirrored_buffer[env_ids, 1:, :] = self.srl_obs_mirrored_buffer[env_ids, :-1, :]  # 向后移动数据
            self.srl_obs_mirrored_buffer[env_ids, 0, :] = srl_obs_mirrored  

            srl_base_obs = self.srl_obs_buffer[env_ids, :, :].reshape(len(env_ids), -1)  
            srl_base_mirrored_obs = self.srl_obs_mirrored_buffer[env_ids, :, :].reshape(len(env_ids), -1)

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
            srl_stacked_obs = torch.cat([srl_base_obs, task_params], dim=-1)
            srl_mirrored_stacked_obs = torch.cat([srl_base_mirrored_obs, mirrored_task_params], dim=-1)

            total_obs = torch.cat((humanoid_obs, srl_stacked_obs), dim=1)
            self.obs_buf[env_ids] = total_obs
            self.srl_obs_buf[env_ids] = srl_stacked_obs
            self.srl_obs_mirrored_buf[env_ids] = srl_mirrored_stacked_obs

            self.potentials[env_ids] = potentials
            self.prev_potentials[env_ids] = prev_potentials
        return

    def _compute_env_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            load_cell_sensor = self.vec_sensor_tensor[:,self.load_cell_ssidx,:]
            if self._srl_endpos_obs: # Add cartisian pos of SRL-end to OBS
                srl_end_body_pos = self._rigid_body_pos[:,self._srl_end_ids, :]
                key_body_pos = torch.cat((key_body_pos, srl_end_body_pos), dim=1)
            dof_force_tensor = self.dof_force_tensor
            srl_root_states = self.srl_root_states.clone()
            srl_root_states[:,3:7]  = self.srl_rotate_root_states[:,3:7]
            srl_root_states[:,10:13]= self.srl_rotate_root_states[:,10:13]
            progress_buf = self.progress_buf
            phase_buf = self.phase_buf
            initial_dof_pos = self.initial_srl_dof_pos
            actions = self.actions
            targets = self.targets
            potentials = self.potentials
            target_vel_x = self.target_vel_x
            target_yaw   = self.target_yaw
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
            load_cell_sensor = self.vec_sensor_tensor[env_ids,self.load_cell_ssidx,:]
            if self._srl_endpos_obs:
                srl_end_body_pos = self._rigid_body_pos[env_ids][:,self._srl_end_ids, :]
                key_body_pos = torch.cat((key_body_pos, srl_end_body_pos), dim=1)
            dof_force_tensor = self.dof_force_tensor[env_ids]
            srl_root_states = self.srl_root_states[env_ids,:].clone()
            srl_root_states[:,3:7]  = self.srl_rotate_root_states[env_ids,3:7]
            srl_root_states[:,10:13]= self.srl_rotate_root_states[env_ids,10:13]
            progress_buf = self.progress_buf[env_ids]
            phase_buf = self.phase_buf[env_ids]
            initial_dof_pos = self.initial_srl_dof_pos[env_ids]
            actions = self.actions[env_ids]
            targets = self.targets[env_ids]
            potentials = self.potentials[env_ids]
            target_vel_x = self.target_vel_x[env_ids]
            target_yaw  = self.target_yaw[env_ids]
 
        humanoid_obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs,
                                            load_cell_sensor, self._humanoid_load_cell_obs)
        
        srl_obs, potentials, prev_potentials,  = compute_srl_observations(phase_buf, initial_dof_pos, srl_root_states, 
                                                                          dof_pos[:, -(self.srl_actions_num):], dof_vel[:, -(self.srl_actions_num):],
                                                                          load_cell_sensor, target_yaw, dof_force_tensor[:, -(self.srl_actions_num):], 
                                                                          actions[:, -(self.srl_actions_num):], self.obs_scales_tensor, 
                                                                          targets, potentials, self.dt, target_vel_x, self.gait_period )
        srl_obs_mirrored = compute_srl_observations_mirrored(phase_buf, self.mirror_act_srl_mat, initial_dof_pos, srl_root_states, 
                                                             dof_pos[:, -(self.srl_actions_num):], dof_vel[:, -(self.srl_actions_num):],
                                                            load_cell_sensor,  target_yaw, dof_force_tensor[:, -(self.srl_actions_num):], 
                                                            actions[:, -(self.srl_actions_num):], self.obs_scales_tensor, 
                                                            targets, target_vel_x, self.gait_period )
        if self._design_param_obs:
            design_param = self.design_param
            design_param = design_param.unsqueeze(0).repeat(obs.shape[0], 1)
            obs = torch.cat([obs, design_param], dim=1)
            obs_mirrored = torch.cat([obs_mirrored, design_param], dim=1)
        return humanoid_obs, srl_obs, srl_obs_mirrored, potentials, prev_potentials,

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
        self.phase_buf[env_ids] = torch.randint(0, int(self.gait_period), (len(env_ids),), device=self.device, dtype=torch.long)
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if self._force_control:
            pd_tar = self._action_to_pd_targets(self.actions)
            torques = self.p_gains*(pd_tar - self._dof_pos) - self.d_gains*self._dof_vel
            self.torques = torch.clip(torques, -self.torque_limits, self.torque_limits).view(self.torques.shape)
            self.torques[:,:self.get_humanoid_action_size()+1] = 0.00
            # self.torques[:,self.get_humanoid_action_size()+1:] = self.torques[:,self.get_humanoid_action_size()+1:]*0.000001
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        if self._pd_control:
            pd_tar = self._action_to_pd_targets(self.actions) # pd_tar.shape: [num_actors, num_dofs]
            if self._force_control:
                pd_tar[:,self.get_humanoid_action_size():] = 0.00
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

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
        self.extras["x_velocity"] = self.obs_buf[:,7]                            
        self.extras["obs_mirrored"] = self.srl_obs_mirrored_buf.to(self.rl_device)  # 镜像观测
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
        self.extras["target_yaw"] = self.target_yaw
        
        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def get_srl_teacher_obs_mask(self):
        obs_dim = self.srl_obs_num  # 单帧观测维度
        frame_stack = self.srl_obs_frame_stack  # 堆叠帧数
        cmd_dim = 3      # 命令输入维度
        total_dim = obs_dim * frame_stack + cmd_dim
        masked_idx = torch.ones(total_dim, dtype=torch.float32)
        mask_num_list = [22, 31]  # 需要mask的单帧观测索引，后续可直接修改
        for i in range(frame_stack):
            for idx in mask_num_list:
                masked_idx[i * obs_dim + idx] = 0.0
        return masked_idx.bool()

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
        margin = 0.0 * math.pi / 180.0 
        low  = (self.dof_limits_lower + margin).unsqueeze(0)
        high = (self.dof_limits_upper - margin).unsqueeze(0)
        return torch.max(torch.min(pd_tar, high), low)

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


# @torch.jit.script
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
    obs = torch.cat((root_h, 
                     root_rot_obs, 
                     local_root_vel,
                     local_root_ang_vel, 
                     humanoid_dof_obs, 
                     humanoid_dof_vel, 
                     load_cell_force,
                     flat_local_key_pos), dim=-1)
    return obs


# @torch.jit.script
def compute_srl_observations(
    phase_buf,
    default_joint_pos,
    root_states ,
    dof_pos ,
    dof_vel ,
    load_cell,
    target_yaw,
    dof_force_tensor ,
    actions,
    obs_scales,
    targets,
    potentials,
    dt,
    target_vel_x,
    gait_period,
)  :
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
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

    # SRL loadcell 是负载力传感器（向下为正）
    load_cell_force = -load_cell

    # 主体关节位置编码（humanoid + SRL）
    # dof_obs = dof_to_obs(dof_pos)  
    srl_dof_obs   = dof_pos 
    srl_dof_obs[:,1:] = srl_dof_obs[:,1:] - default_joint_pos
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
                     srl_dof_obs[:,1:] * obs_scales[2],  # 6    10:15
                     srl_dof_vel[:,1:] * obs_scales[3],  # 6    16:21
                     actions ,                           # 7    22:28
                     sin_phase,                          # 1    29
                     cos_phase,                          # 1    30
                     srl_dof_obs[:,0:1]                    # 1    31
                    ), dim=-1)
    return obs , potentials, prev_potentials_new

# @torch.jit.script
def compute_srl_observations_mirrored(
    phase_buf,
    mirror_mat,
    default_joint_pos,
    root_states ,
    dof_pos ,
    dof_vel ,
    load_cell ,
    target_yaw,
    dof_force_tensor ,
    actions,
    obs_scales,
    targets,
    target_vel_x,
    gait_period,
)  :
    # type: ( Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) ->  Tensor 
    
    
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

    # SRL loadcell 是负载力传感器（向下为正）
    load_cell_force = -load_cell

    # 主体关节位置编码（humanoid + SRL）
    # dof_obs = dof_to_obs(dof_pos)  
    srl_dof_obs   = dof_pos 
    srl_dof_obs[:,1:] = srl_dof_obs[:,1:] - default_joint_pos
    srl_dof_vel   = dof_vel
    srl_dof_force = dof_force_tensor



    euler = quat_to_euler_xyz(root_rot)
    target_euler = torch.zeros_like(euler,device=euler.device)
    target_euler[:,0] = target_yaw
    euler_err = target_euler - euler


    # Mirrored
    local_root_vel[:,1] = -local_root_vel[:,1] # y方向速度
    local_root_ang_vel[:,0] = -local_root_ang_vel[:,0] # x轴角速度
    local_root_ang_vel[:,2] = -local_root_ang_vel[:,2] # z轴角速度
    srl_dof_obs = torch.matmul(srl_dof_obs, mirror_mat) # Perform the matrix multiplication to get mirrored dof_pos
    srl_dof_vel = torch.matmul(srl_dof_vel, mirror_mat)
    srl_dof_force = torch.matmul(srl_dof_force, mirror_mat)
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
                     srl_dof_obs[:,:] * obs_scales[2],    # 6
                     srl_dof_vel[:,1:] * obs_scales[3],        # 6
                     actions ,
                     sin_phase,    
                     cos_phase,     
                    ), dim=-1)
    return obs  

def compute_humanoid_reward(obs_buf):
    target_vel_x = obs_buf[:, -3]

    # --- Target Velocity ---
    root_vel = obs_buf[:, 7:10] 
    root_target_vel = torch.zeros((root_vel.shape[0], 3), device=root_vel.device)
    root_target_vel[:, 0] = target_vel_x  
    vel_error_vec = root_vel - root_target_vel
    vel_tracking_reward =  1 *  torch.exp(-4 * torch.norm(vel_error_vec, dim=-1))  # α = 1.5

    total_reward = vel_tracking_reward
    return total_reward



# @torch.jit.script
def compute_srl_reward(
    obs_buf,
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
    srl_obs_num: int = 0,
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
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float,  float, float, float, float, float, float, float, float, float, float, float, float, float) -> Tuple[Tensor, ]

    # obs = root_h,                             # 1    0
    #       local_root_vel ,                    # 3    1:3
    #       local_root_ang_vel ,                # 3    4:6
    #       euler_err,                          # 3    7:9
    #       srl_dof_obs * obs_scales[2],        # 6    10:15
    #       srl_dof_vel * obs_scales[3],        # 6    16:21
    #       actions ,                           # 6    22:28
    #       sin_phase,                          # 1    29
    #       cos_phase,                          # 1    30   total: 31

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
    dof_vel = obs_buf[:, 16:16+(actions.shape[1]-1)]
    dof_vel_cost = torch.sum(dof_vel ** 2, dim=-1)

    # --- DOF acceleration cost ---
    dof_vel_prev = obs_buf[:, 16+srl_obs_num:16+srl_obs_num+(actions.shape[1]-1)]  # 前一帧速度
    dof_acc = dof_vel - dof_vel_prev  # 关节加速度
    dof_acc_magnitude_sq = torch.sum(dof_acc ** 2, dim=-1)
    dof_acc_reward = torch.exp(- 2 * dof_acc_magnitude_sq)

    # --- Action Smooth ---
    actions_prev = obs_buf[:, 22+srl_obs_num:22+srl_obs_num+actions.shape[1]]
    actions_prev_prev = obs_buf[:, 22+srl_obs_num*2:22+srl_obs_num*2+actions.shape[1]]
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
    # walking
    no_feet_on_ground = (left_foot_height > contact_threshold) & (right_foot_height > contact_threshold)
    # standing
    no_both_feet_on_ground = (left_foot_height > contact_threshold) | (right_foot_height > contact_threshold)
    fly_idx = torch.where(target_vel_x < 0.1, no_both_feet_on_ground, no_feet_on_ground)
    no_fly_penalty = torch.where(fly_idx, torch.ones_like(no_feet_on_ground) * no_fly_penalty_scale, torch.zeros_like(no_feet_on_ground))
    

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
   
    # Termination penalty
    srl_unstable = torch.where(torch.abs(obs_buf[:,31]) > 0.38, torch.ones_like(alive_reward), torch.zeros_like(alive_reward))
    srl_termination_penalty = srl_unstable*10

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
        - lateral_distance_penalty_scale * feet_lateral_penalty \
        - srl_termination_penalty

    # --- Handle termination ---
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    return total_reward

# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, srl_obs_buf, srl_freejoint_pos, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height, srl_termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float, float) -> Tuple[Tensor, Tensor]
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

        # srl fallen        
        srl_fallen = torch.where(srl_obs_buf[:, 0] < srl_termination_height, torch.ones_like(reset_buf), terminated)
        # srl unstable
        srl_unstable = torch.where(torch.abs(srl_freejoint_pos[:,0]) > 0.490, torch.ones_like(reset_buf), terminated)
        srl_failed = torch.logical_or(srl_fallen, srl_unstable)

        has_fallen = torch.logical_or(has_fallen, srl_failed)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

# # 计算任务奖励函数
# @torch.jit.script
# def compute_humanoid_reward(obs_buf, 
#                             dof_force_tensor, 
#                             contact_buf,  # body net contact force
#                             action, 
#                             _torque_threshold, 
#                             upper_body_pos, 
#                             upper_reward_w, 
#                             srl_joint_ids,
#                             srl_load_cell_sensor,
#                             srl_feet_slip,
#                             srl_torque_w = 0.0,
#                             srl_load_cell_w  = 0.0 ,
#                             srl_feet_slip_w = 0.0):
#     # type: (Tensor, Tensor, Tensor, Tensor, int, Tensor, int, Tensor, Tensor, Tensor, float, float , float ) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
#     # TODO: 目标速度跟随
#     velocity_threshold = 1.4
    
#     velocity  = obs_buf[:,7]  # vx
#     vy  = obs_buf[:,8]  # vy
#     velocity_penalty = - 20 * (vy**2) - torch.where(velocity < velocity_threshold, (velocity_threshold - velocity)**2, torch.zeros_like(velocity))

    
    
#     # v1.5.12 比例惩罚，humanoid力矩绝对值超过100
#     torque_threshold = _torque_threshold
#     torque_usage   = dof_force_tensor[:, 14:28]
#     torque_penalty = torch.where(torch.abs(torque_usage) > torque_threshold, 
#                                  (torch.abs(torque_usage) - torque_threshold) / torque_threshold, 
#                                  torch.zeros_like(torque_usage))
#     torque_reward  = - torch.sum(torque_penalty, dim=1)
#     # MLY: 暂时关闭HUMANOID受力惩罚
#     torque_reward = torque_reward * 0

#     # 外肢体水平奖励项
#     board_pos = upper_body_pos[:, 0, :]  # (4096, 3)
#     root_pos = upper_body_pos[:, 1, :]  # (4096, 3)
#     upper_body_direction = board_pos - root_pos  # 维度 (4096, 3)
#     norm_upper_body_direction = upper_body_direction / torch.norm(upper_body_direction, dim=1, keepdim=True)
#     # upper_reward = upper_reward_w * (norm_upper_body_direction[:,2] - 1 )
#     upper_reward = - upper_reward_w * (torch.abs(norm_upper_body_direction[:,2]) )

#     # SRL 关节力矩惩罚
#     srl_joint_forces = dof_force_tensor[:,  srl_joint_ids]
#     srl_torque_sum = - torch.sum((srl_joint_forces/100) ** 2, dim=1)
#     srl_torque_reward = srl_torque_sum * srl_torque_w
#     # MLY: 暂时关闭SRL受力惩罚
#     srl_torque_reward = 0


#     # SRL Root受力惩罚
#     load_cell_z = srl_load_cell_sensor[:,2] # 原始数据为正
#     load_cell_y = srl_load_cell_sensor[:,1]
#     load_cell_x = srl_load_cell_sensor[:,0] 
#     scaled_z    = torch.clamp(torch.abs(load_cell_z), min=50, max=2500)  # 限制受力范围
#     scaled_y    = torch.clamp(torch.abs(load_cell_y), min=50, max=2500) 
#     scaled_x    = torch.clamp(torch.abs(load_cell_x), min=50, max=2500)
#     z_penalty   = ((scaled_z - 50) / 50) ** 2 # 平方
#     y_penalty   = ((scaled_y - 50) / 50) ** 2  
#     x_penalty   = ((scaled_x - 50) / 50) ** 2  
#     # z_penalty =  torch.log(1.0 + (scaled_z - 100) / 100.0) / 3.0  # 对数
#     srl_load_cell_reward = -srl_load_cell_w * (z_penalty + y_penalty + x_penalty)

#     # 末端滑动惩罚 feet slip
#     srl_feet_slip_reward = - srl_feet_slip_w * srl_feet_slip.squeeze(1)


#     # scaled_x = torch.clamp(load_cell_x ,min=0)
#     # x_penalty = scaled_x / 1000.0  # ~1 if x=1000
#     # load_cell_penalty =  z_penalty    
#     # srl_load_cell_reward = -load_cell_penalty * srl_load_cell_w       

#     # reward = -velocity_penalty + torque_reward
#     reward = velocity_penalty + torque_reward + upper_reward + srl_torque_reward + srl_load_cell_reward + srl_feet_slip_reward
    
#     # return reward, velocity_penalty, torque_reward, upper_reward
#     return reward, velocity_penalty, srl_load_cell_reward, upper_reward

