# srl_real_hri_base.py
# 真实SRL模型+Humanoid 人机协同实验
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
import math
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, get_axis_params, calc_heading_quat_inv, \
     exp_map_to_quat, quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv, quat_rotate_inverse, quat_conjugate

from ..base.vec_task import VecTask
from isaacgymenvs.learning.SRLEvo.srl_models import ModelSRLContinuous 
from isaacgymenvs.learning.SRLEvo.srl_network_builder import SRLBuilder 
from isaacgymenvs.tasks.SRLEvo.traj_generator import SimpleCurveGenerator as TrajGenerator

DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]

SRL_ROOT_BODY_NAMES = ["SRL", "SRL_root"]
UPPER_BODY_NAMES = ["pelvis", "torso"]
KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]  # body end + SRL end
SRL_END_BODY_NAMES = ["SRL_left_end","SRL_right_end"] 
SRL_CONTACT_BODY_NAMES = ['SRL_root', 'right_knee_link', 'SRL_right_end', 'left_knee_link', 'SRL_left_end']


class SRL_Real_HRI_Base(VecTask):

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
        self.teacher_srl_obs_num = self.cfg["env"]["teacher_srl_obs_num"]
        self.srl_free_actions_num = self.cfg["env"]["srl_free_actions_num"]
        self.srl_obs_frame_stack = self.cfg["env"].get("srl_obs_frame_stack", 5)
        self.srl_command_num = self.cfg["env"]["srl_command_num"]
        self.gait_period = self.cfg["env"]["gait_period"]
        self.foot_clearance = self.cfg["env"]["foot_clearance"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.srl_termination_height = self.cfg["env"]["srlTerminationHeight"]
        self.hri_virtual_stiffness = self.cfg["env"].get("hri_virtual_stiffness", 150000.0)
        self.hri_virtual_damping = self.cfg["env"].get("hri_virtual_damping", 3000.0)
        self.hri_virtual_force_max = self.cfg["env"].get("hri_virtual_force_max", 1000.0)

        # --- reward ---
        self.alive_reward_scale = self.cfg["env"]["alive_reward_scale"]
        self.humanoid_share_reward_scale = self.cfg["env"]["humanoid_share_reward_scale"]
        self.progress_reward_scale = self.cfg["env"]["progress_reward_scale"]
        self.torques_cost_scale = self.cfg["env"]["torques_cost_scale"]
        self.dof_acc_cost_scale = self.cfg["env"]["dof_acc_cost_scale"]
        self.dof_vel_cost_scale = self.cfg["env"]["dof_vel_cost_scale"]
        self.dof_pos_cost_sacle = self.cfg["env"]["dof_pos_cost_sacle"]
        self.contact_force_cost_scale = self.cfg["env"]["contact_force_cost_scale"]
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
        self._design_param_obs = self.cfg["env"].get("design_param_obs", False)
        self._load_cell_activate = self.cfg["env"].get("load_cell",False)
        self._humanoid_load_cell_obs = self.cfg["env"].get("humanoid_load_cell_obs", False)
        self._srl_partial_obs = self.cfg["env"].get("srl_partial_obs", False)
        motor_opt_cfg = self.cfg["env"].get("srl_motor_opt", {})
        self._srl_max_effort = motor_opt_cfg.get("srl_max_effort", 300.0)

        # self.initial_dof_pos = torch.tensor(self.srl_default_joint_angles, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.obs_scales={
            "lin_vel" : 1,
            "ang_vel" : 1,
            "dof_pos" : 1.0,
            "dof_vel" : 0.05,
            "load_cell": 0.01,
            "height_measurements" : 5.0 }
     
        # --- SRL-Gym Defined End ---

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.default_srl_joint_angles = [0*np.pi, 
                                            -0.1,
                                            0.2,
                                            0*np.pi,
                                            -0.1,
                                            0.2,]
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        dt = self.cfg["sim"]["dt"]
        self.control_dt = self.control_freq_inv * dt

        # --- srl defined ---
        # 28 为被动链接关节
        self._srl_joint_ids = to_torch([ 29, 30, 31, 32, 33, 34], device=self.device, dtype=torch.long)
        # --- srl defined end ---
        # ==========================================================
        # SRL motor peak / thermal optimization (based on applied torques)
        #   - Peak penalty: discourage operating close to actuator limits (instant + decayed window)
        #   - Thermal proxy: EMA of mean normalized torque^2 (approx. copper loss / heating)
        #   - Optional: EMA of mean normalized |tau * omega| (mechanical power)
        # NOTE: We use self.torques (after clip) as the actuator output. dof_force_tensor contains
        #       constraint/contact reactions and is better suited for mechanical-load safety checks.
        # ==========================================================
        self.srl_peak_start_ratio = float(motor_opt_cfg.get("peak_start_ratio", 0.65))   # start penalizing from this ratio
        self.srl_peak_ratio = float(motor_opt_cfg.get("peak_ratio", 0.85))               # recommended "soft peak" ratio (<=1.0)
        self.srl_peak_cost_scale = float(motor_opt_cfg.get("peak_cost_scale", 0.30))     # reward weight
        self.srl_thermal_tau_s = float(motor_opt_cfg.get("thermal_tau_s", 1.5))          # seconds (EMA time constant)
        self.srl_thermal_cost_scale = float(motor_opt_cfg.get("thermal_cost_scale", 0.05))
        self.srl_power_cost_scale = float(motor_opt_cfg.get("power_cost_scale", 0.00))   # optional
        self.srl_peak_window_tau_s = float(motor_opt_cfg.get("peak_window_tau_s", 0.30)) # seconds for decayed peak window
        self.srl_omega_ref = float(motor_opt_cfg.get("omega_ref", 14.0))                 # rad/s for power normalization
        self.srl_rated_ratio = float(motor_opt_cfg.get("rated_ratio", 0.80)) 
     
        # Per-motor torque limits (already filled after super().__init__() -> create_sim -> _create_envs)
        self.srl_tau_limits = self.torque_limits[ -self.srl_actions_num:].clamp(min=1e-6)

        # clamp ratios to sane ranges
        self.srl_peak_start_ratio = float(np.clip(self.srl_peak_start_ratio, 0.0, 0.99))
        self.srl_peak_ratio = float(np.clip(self.srl_peak_ratio, self.srl_peak_start_ratio + 1e-3, 1.0))

        # EMA decay factors
        self._srl_thermal_gamma = float(np.exp(-self.control_dt / max(self.srl_thermal_tau_s, 1e-6)))
        self._srl_peak_decay = float(np.exp(-self.control_dt / max(self.srl_peak_window_tau_s, 1e-6)))

        # Running stats (per-env)
        self.srl_tau2_ema = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.srl_power_ema = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.srl_peak_ratio_window = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)


        # mirror matrix
        self.mirror_idx_humanoid = np.array([-0.0001, 1, -2, -3, 4, -5, -10, 11, -12,  13, -6,
                                                 7, -8, 9, -21, 22, -23, 24, -25, 26, -27, -14, 
                                                 15, -16, 17, -18, 19, -20,])
        self.mirror_idx_srl = np.array([-31,32,33, -28,29,30,])
        self.mirror_idx = np.concatenate((self.mirror_idx_humanoid, self.mirror_idx_srl))
        obs_dim = self.mirror_idx.shape[0]
        self.mirror_mat = torch.zeros((obs_dim, obs_dim), dtype=torch.float32, device=self.device)
        for i, perm in enumerate(self.mirror_idx):
            self.mirror_mat[i, int(abs(perm))] = np.sign(perm)
        self.mirror_idx_act_srl = np.array([3, 4, 5, 0.01, 1, 2,  ])
        self.mirror_act_srl_mat = torch.zeros((self.mirror_idx_act_srl.shape[0], self.mirror_idx_act_srl.shape[0]), dtype=torch.float32, device=self.device)
        for i, perm in enumerate(self.mirror_idx_act_srl):
            self.mirror_act_srl_mat[i, int(abs(perm))] = np.sign(perm)
         
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) #  State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13])
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 5
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
        self._dof_vel_prev = self._dof_vel.clone()

        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.target_ang_vel_z = torch.zeros(self.num_envs, device=self.device)
        self.target_pelvis_height = torch.full((self.num_envs,), 0.95, device=self.device)
        self.target_vel_x = torch.full((self.num_envs,), 1.0, device=self.device)

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
        
        
        self.right_thigh_states = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.right_thigh_index ,  :]
        self.right_shin_states = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.right_shin_index ,  :]
        self.left_thigh_states = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.left_thigh_index ,  :]
        self.left_shin_states = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.left_shin_index ,  :]
        self.srl_root_states = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.srl_root_index ,  :]
        self.prev_srl_end_body_pos = torch.zeros((self.num_envs,2,3), device=self.device)

        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self.srl_obs_buf = torch.zeros(
            (self.num_envs, self.get_srl_obs_size()), device=self.device, dtype=torch.float)
        self.srl_obs_mirrored_buf = torch.zeros(
            (self.num_envs, self.get_srl_obs_size()), device=self.device, dtype=torch.float)

        self.srl_priv_extra_obs_buf = torch.zeros(
            (self.num_envs, self.get_srl_priv_obs_size()-self.get_srl_obs_size()), device=self.device, dtype=torch.float)
        self.srl_priv_extra_mirrored_obs_buf = torch.zeros(
            (self.num_envs, self.get_srl_priv_obs_size()-self.get_srl_obs_size()), device=self.device, dtype=torch.float)

        self.teacher_srl_obs_buf = torch.zeros(
            (self.num_envs, self.get_teacher_srl_obs_size()), device=self.device, dtype=torch.float)

        # frame stack buffer
        self.srl_obs_buffer = torch.zeros((self.num_envs, self.srl_obs_frame_stack, self.srl_obs_num), device=self.device)
        self.srl_obs_mirrored_buffer = torch.zeros((self.num_envs, self.srl_obs_frame_stack, self.srl_obs_num), device=self.device)
        self.teacher_srl_obs_buffer = torch.zeros((self.num_envs, self.srl_obs_frame_stack, self.teacher_srl_obs_num), device=self.device)


        self.humanoid_task_rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.srl_rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.rew_joint_cost_buf = torch.zeros(      # 关节力矩惩罚
            self.num_envs, device=self.device, dtype=torch.float)
        self.rew_v_pen_buf = torch.zeros(           # 速度惩罚
            self.num_envs, device=self.device, dtype=torch.float)
        self.rew_upper_buf = torch.zeros(           # 直立惩罚
            self.num_envs, device=self.device, dtype=torch.float)
        if self._design_param_obs:
            design_param = self._get_design_param()

        # self.observation_space
        self.obs_scales_tensor = torch.tensor([
        self.obs_scales["lin_vel"],
        self.obs_scales["ang_vel"],
        self.obs_scales["dof_pos"],
        self.obs_scales["dof_vel"],
        self.obs_scales["load_cell"],
        ], device=self.device)
        
        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.potentials = to_torch([-1000./self.control_dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        # humanoid trajectory generator
        self._traj_sample_times = [0.5, 1.0, 1.5] # 轨迹参数
        self._num_traj_points = len(self._traj_sample_times)
        self._traj_obs_dim = self._num_traj_points * 2 
        episode_duration = self.max_episode_length * self.control_dt
        self._traj_gen = TrajGenerator(self.num_envs, self.device, self.control_dt, episode_duration,
                                speed_mean=1.0,      
                                turn_speed_max=0.0,  # 不要太大，先设为0.5
                                num_turns=2)         # 转向次数
        
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
    
    def get_srl_free_action_size(self):
        return self.srl_free_actions_num

    def get_teacher_srl_obs_size(self):
        return self.teacher_srl_obs_num*self.srl_obs_frame_stack + self.srl_command_num
    
    def get_teacher_srl_action_size(self):
        return self.srl_actions_num 

    def get_humanoid_obs_size(self):
        return self.humanoid_obs_num
    
    def get_srl_obs_size(self):
        return self.srl_obs_num*self.srl_obs_frame_stack + self.srl_command_num

    def get_srl_priv_obs_size(self):
        return 12 + self.srl_obs_num*self.srl_obs_frame_stack + self.srl_command_num

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        total_dof_nums = self.humanoid_actions_num + self.srl_free_actions_num + self.srl_actions_num
        self.torques = torch.zeros(self.num_envs, total_dof_nums, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(total_dof_nums, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(total_dof_nums, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_limits = torch.zeros(total_dof_nums, dtype=torch.float, device=self.device, requires_grad=False)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        for env_id in env_ids:
            self.srl_obs_buffer[env_id] = 0
            self.srl_obs_mirrored_buffer[env_id] = 0
            self.teacher_srl_obs_buffer[env_id] = 0
        # reset SRL motor running stats (peak/thermal)
        self.srl_tau2_ema[env_ids] = 0.0
        self.srl_power_ema[env_ids] = 0.0
        self.srl_peak_ratio_window[env_ids] = 0.0
        root_pos = self._root_states[env_ids, 0:3]
        root_rot = self._root_states[env_ids, 3:7]
        self._traj_gen.reset(env_ids, root_pos, init_rot=root_rot)
        self._refresh_sim_tensors()
        # 重置 foot clearance 的历史
        srl_end_body_pos = self._rigid_body_pos[env_ids][:, self._srl_end_ids, :]
        self.prev_srl_end_body_pos[env_ids] = srl_end_body_pos.clone()
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
            print('Asset file name: '+asset_file)
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        self.srl_root_index = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL_root", )

        self.right_thigh_index = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_thigh", )
        self.right_shin_index = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_shin", )
        self.left_thigh_index = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_thigh", )
        self.left_shin_index = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_shin", )
        # Add actuator list
        self._dof_names = self.gym.get_asset_dof_names(humanoid_asset)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        srl_free_joint_x_idx = self.gym.find_asset_dof_index(humanoid_asset, 'SRL_freejoint_x')
        srl_free_joint_y_idx = self.gym.find_asset_dof_index(humanoid_asset, 'SRL_freejoint_y')
        srl_free_joint_z_idx = self.gym.find_asset_dof_index(humanoid_asset, 'SRL_freejoint_z')
        self.srl_virtual_damping_x_idx = self.gym.find_asset_dof_index(humanoid_asset, 'SRL_virtual_damping_x')
        self.srl_virtual_damping_y_idx = self.gym.find_asset_dof_index(humanoid_asset, 'SRL_virtual_damping_y')
        self.srl_virtual_damping_z_idx = self.gym.find_asset_dof_index(humanoid_asset, 'SRL_virtual_damping_z')
        self.srl_virtual_damping_ids = torch.as_tensor(
                [int(self.srl_virtual_damping_x_idx),
                int(self.srl_virtual_damping_y_idx),
                int(self.srl_virtual_damping_z_idx)],
                device=self.rl_device, dtype=torch.long
            )

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset,  "left_foot")
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
            # forward dynamics
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True
            sensor_props.enable_constraint_solver_forces = False
            sensor_props.use_world_frame = False
            load_cell_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "SRL")
            self.load_cell_ssidx_fd = self.gym.create_asset_force_sensor(humanoid_asset, load_cell_idx, sensor_pose, sensor_props)      
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
        start_pose.p = gymapi.Vec3(*get_axis_params(1.05, self.up_axis_idx))
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
                srl_start_id = self.get_srl_action_size() 
                dof_prop[-srl_start_id:]["stiffness"].fill(0.0)
                dof_prop[-srl_start_id:]["damping"].fill(0.0)
                dof_prop[-srl_start_id:]["velocity"].fill(14.0)
                dof_prop[-srl_start_id:]["effort"].fill(self._srl_max_effort)
                dof_prop[-srl_start_id:]["driveMode"] = gymapi.DOF_MODE_EFFORT
                if not srl_free_joint_x_idx == -1:
                    dof_prop[srl_free_joint_x_idx]["effort"] = 200.0
                    dof_prop[srl_free_joint_x_idx]["stiffness"] = 0.0
                    dof_prop[srl_free_joint_x_idx]["damping"] = 0.0
                    dof_prop[srl_free_joint_x_idx]["driveMode"] = gymapi.DOF_MODE_EFFORT
                if not srl_free_joint_y_idx == -1:
                    dof_prop[srl_free_joint_y_idx]["effort"] = 200.0
                    dof_prop[srl_free_joint_y_idx]["stiffness"] = 0.0
                    dof_prop[srl_free_joint_y_idx]["damping"] = 0.0
                    dof_prop[srl_free_joint_y_idx]["driveMode"] = gymapi.DOF_MODE_EFFORT
                if not srl_free_joint_z_idx == -1:
                    dof_prop[srl_free_joint_z_idx]["effort"] = 200.0
                    dof_prop[srl_free_joint_z_idx]["stiffness"] = 0.0
                    dof_prop[srl_free_joint_z_idx]["damping"] = 0.0
                    dof_prop[srl_free_joint_z_idx]["driveMode"] = gymapi.DOF_MODE_EFFORT
                if not self.srl_virtual_damping_x_idx == -1:
                    dof_prop[self.srl_virtual_damping_x_idx]["effort"] = 5000
                    dof_prop[self.srl_virtual_damping_x_idx]["stiffness"] = self.hri_virtual_stiffness
                    dof_prop[self.srl_virtual_damping_x_idx]["damping"] = self.hri_virtual_damping
                    dof_prop[self.srl_virtual_damping_x_idx]["driveMode"] = gymapi.DOF_MODE_POS
                if not self.srl_virtual_damping_y_idx == -1:
                    dof_prop[self.srl_virtual_damping_y_idx]["effort"] = 5000
                    dof_prop[self.srl_virtual_damping_y_idx]["stiffness"] = self.hri_virtual_stiffness
                    dof_prop[self.srl_virtual_damping_y_idx]["damping"] = self.hri_virtual_damping
                    dof_prop[self.srl_virtual_damping_y_idx]["driveMode"] = gymapi.DOF_MODE_POS
                if not self.srl_virtual_damping_z_idx == -1:
                    dof_prop[self.srl_virtual_damping_z_idx]["effort"] = 5000
                    dof_prop[self.srl_virtual_damping_z_idx]["stiffness"] = self.hri_virtual_stiffness
                    dof_prop[self.srl_virtual_damping_z_idx]["damping"] = self.hri_virtual_damping
                    dof_prop[self.srl_virtual_damping_z_idx]["driveMode"] = gymapi.DOF_MODE_POS
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
            srl_start_id = self.get_srl_action_size() 
            indices_to_limit = [1, 2, 4, 5] 
            for i in indices_to_limit:
                self.dof_limits_lower[i] = self.default_srl_joint_angles[i] - 45/180*np.pi
                self.dof_limits_upper[i] = self.default_srl_joint_angles[i] + 45/180*np.pi
                
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
        self.obs_dict["srl_priv_extra_obs"] = self.srl_priv_extra_obs_buf.to(self.rl_device)
        self.obs_dict["srl_priv_extra_mirrored_obs"] = self.srl_priv_extra_mirrored_obs_buf.to(self.rl_device)
        self.obs_dict["teacher_srl_obs"] = torch.clamp(self.teacher_srl_obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        return self.obs_dict, done_env_ids
        
    def _build_pd_action_offset_scale(self):
        Dof_offsets = DOF_OFFSETS.copy()
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
                    # FIXME: Limited Range (previous value: 0.45)
                    curr_scale = 0.45 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_offset[self.get_humanoid_action_size()+self.get_srl_free_action_size():]=self.default_srl_joint_angles
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
        v_xy = torch.sqrt(dx*dx + dy*dy) / self.control_dt             # [num_envs,2]
        zeros = torch.zeros_like(v_xy,device=v_xy.device)
        v_xy = torch.where(v_xy<0.8, zeros, v_xy)
        # 参数设定
        pz_target = self.foot_clearance

        this_term = (pz_target - pz) ** 2 * v_xy                 
        clearance_reward = torch.sum(this_term, dim=1)

        return clearance_reward
    
    def _compute_reward(self, actions):
        # srl_root_body_pos = self._rigid_body_pos[:, self._srl_root_body_ids, :]
        load_cell_sensor = self._virtual_load_cell_from_dof(self._dof_pos, self._dof_vel)
        srl_end_body_pos = self._rigid_body_pos[:, self._srl_end_ids, :]
        # srl_end_body_vel = self._rigid_body_vel[:, self._srl_end_ids, :]
        to_target = self.targets - self._initial_root_states[:, 0:3]
        srl_root_pos = self.srl_root_states[:, 0:3]
        clearance_reward = self.compute_foot_clearance_reward()

        # --- SRL motor near-peak + thermal / power costs (based on applied torques) ---
        srl_peak_cost, srl_thermal_cost, srl_power_cost = self._compute_srl_motor_load_costs()
        srl_motor_cost = (self.srl_peak_cost_scale * srl_peak_cost
                          + self.srl_thermal_cost_scale * srl_thermal_cost
                          + self.srl_power_cost_scale * srl_power_cost)

        # log for debugging (optional; available in self.extras)
        if self.viewer != None:
            self.extras["srl_motor_peak_cost"] = srl_peak_cost.to(self.rl_device)
            self.extras["srl_motor_thermal_cost"] = srl_thermal_cost.to(self.rl_device)
            self.extras["srl_motor_power_cost"] = srl_power_cost.to(self.rl_device)
            peak0 = float(srl_peak_cost[0].item())
            therm0 = float(srl_thermal_cost[0].item())
            pow0 = float(srl_power_cost[0].item())
            total0 = float(srl_motor_cost[0].item())
            print(f"[SRL motor][env0][step {self.progress_buf[0]}] peak={peak0:.6f} thermal={therm0:.6f} power={pow0:.6f} total={total0:.6f}")

        env_ids = torch.arange(self.num_envs, device=self.device)
        current_time = self.progress_buf * self.control_dt
        humanoid_target_point = self._traj_gen.get_position(env_ids, current_time)
        humanoid_root_pos = self._root_states[..., 0:3]
        self.rew_buf[:], self.humanoid_task_rew_buf[:] = compute_humanoid_reward(self.obs_buf, self._dof_pos, 
                                                  self._dof_vel, self._dof_vel_prev, 
                                                  humanoid_target_point, humanoid_root_pos, load_cell_sensor*self.obs_scales["load_cell"],)
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
                                            self.humanoid_task_rew_buf,
                                            srl_obs_num = self.srl_obs_num,
                                            alive_reward_scale = self.alive_reward_scale,
                                            humanoid_share_reward_scale = self.humanoid_share_reward_scale,
                                            progress_reward_scale = self.progress_reward_scale,
                                            torques_cost_scale = self.torques_cost_scale,
                                            dof_acc_cost_scale = self.dof_acc_cost_scale,
                                            dof_vel_cost_scale = self.dof_vel_cost_scale,
                                            dof_pos_cost_sacle = self.dof_pos_cost_sacle,
                                            contact_force_cost_scale = self.contact_force_cost_scale,
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
                                            srl_motor_cost = srl_motor_cost,
                                        )         
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf, self.srl_obs_buf,
                                                   self._dof_pos[:,-self.srl_actions_num:], self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height, self._srl_termination_height)
        # 检测是否偏离轨迹太远
        current_time = self.progress_buf * self.control_dt
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        root_pos = self._root_states[..., 0:3]

        dist_sq = torch.sum((target_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
        
        # 如果偏离超过 1 米，强制重置
        max_dist_sq = 100.0 * 100.0
        has_strayed = dist_sq > max_dist_sq
        
        # 刚开始的前几帧不检测（反应时间）
        has_strayed = has_strayed & (self.progress_buf > 10)

        strayed_tensor = has_strayed.to(dtype=torch.long)

        # 更新重置 Buffer
        self.reset_buf = self.reset_buf | strayed_tensor
        self._terminate_buf = self._terminate_buf | strayed_tensor
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _virtual_load_cell_from_dof(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
        """
        用 3 个虚拟 slide DOF 的 PD(刚度/阻尼)计算交互力:
            F = k*(q_tar - q) - c*qdot, 这里 q_tar=0 -> F = -k*q - c*qdot
        返回 shape [N, 6]，前3维是力(N)，后3维力矩置0。
        """

        q  = dof_pos.index_select(1, self.srl_virtual_damping_ids)   # [N,3] (m)
        qd = dof_vel.index_select(1, self.srl_virtual_damping_ids)   # [N,3] (m/s)
        # stiffness/damping 
        k = self.p_gains.index_select(0, self.srl_virtual_damping_ids)  # [3] (N/m)
        c = self.d_gains.index_select(0, self.srl_virtual_damping_ids)  # [3] (N*s/m)
        # PD force: q_tar = 0
        F = -k * q - c * qd     # [N,3] (N)
        # effort 上限（你设为 50000）
        F = torch.clamp(F,-self.hri_virtual_force_max, self.hri_virtual_force_max)

        wrench = torch.zeros((dof_pos.shape[0], 6), device=dof_pos.device, dtype=dof_pos.dtype)
        wrench[:, 0:3] = - F
        return wrench


    def _compute_observations(self, env_ids=None):
        humanoid_obs, srl_obs, srl_obs_mirrored, priv_extra_obs, priv_extra_obs_mirrored, teacher_srl_obs, potentials, prev_potentials, = self._compute_env_obs(env_ids)

        if (env_ids is None):
            # frame rolling
            self.srl_obs_buffer = torch.roll(self.srl_obs_buffer, shifts=1, dims=1)  # 向后移动数据
            self.srl_obs_buffer[:, 0, :] = srl_obs.clone()  # 将新的观测数据放到队列的开头
            self.srl_obs_mirrored_buffer = torch.roll(self.srl_obs_mirrored_buffer, shifts=1, dims=1)  
            self.srl_obs_mirrored_buffer[:, 0, :] = srl_obs_mirrored.clone()   
            self.teacher_srl_obs_buffer = torch.roll(self.teacher_srl_obs_buffer, shifts=1, dims=1) 
            self.teacher_srl_obs_buffer[:, 0, :] = teacher_srl_obs.clone()   

            srl_base_obs = self.srl_obs_buffer.reshape(self.num_envs, -1)  
            srl_base_mirrored_obs = self.srl_obs_mirrored_buffer.reshape(self.num_envs, -1)
            teacher_srl_base_obs = self.teacher_srl_obs_buffer.reshape(self.num_envs, -1)

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
            teacher_srl_stacked_obs = torch.cat([teacher_srl_base_obs, task_params], dim=-1)

            total_obs = torch.cat((humanoid_obs, srl_stacked_obs),dim=1)
            self.obs_buf[:] = total_obs
            self.srl_obs_buf[:] = srl_stacked_obs
            self.srl_obs_mirrored_buf[:] = srl_mirrored_stacked_obs      
            self.srl_priv_extra_obs_buf[:] = priv_extra_obs
            self.srl_priv_extra_mirrored_obs_buf[:] = priv_extra_obs_mirrored
            self.teacher_srl_obs_buf[:] = teacher_srl_stacked_obs

            self.potentials[:] = potentials
            self.prev_potentials[:] = prev_potentials

        else:
            # 对指定环境进行更新 
            self.srl_obs_buffer[env_ids] = torch.roll(self.srl_obs_buffer[env_ids], shifts=1, dims=1)  
            self.srl_obs_buffer[env_ids, 0, :] = srl_obs.clone()  
            self.srl_obs_mirrored_buffer[env_ids] = torch.roll(self.srl_obs_mirrored_buffer[env_ids], shifts=1, dims=1)  
            self.srl_obs_mirrored_buffer[env_ids, 0, :] = srl_obs_mirrored.clone()  
            self.teacher_srl_obs_buffer[env_ids] = torch.roll(self.teacher_srl_obs_buffer[env_ids], shifts=1, dims=1) 
            self.teacher_srl_obs_buffer[env_ids, 0, :] = teacher_srl_obs.clone()   

            srl_base_obs = self.srl_obs_buffer[env_ids, :, :].reshape(len(env_ids), -1)  
            srl_base_mirrored_obs = self.srl_obs_mirrored_buffer[env_ids, :, :].reshape(len(env_ids), -1)
            teacher_srl_base_obs = self.teacher_srl_obs_buffer[env_ids, :, :].reshape(len(env_ids), -1)

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
            teacher_srl_stacked_obs = torch.cat([teacher_srl_base_obs, task_params], dim=-1)

            total_obs = torch.cat((humanoid_obs, srl_stacked_obs), dim=1)
            self.obs_buf[env_ids] = total_obs
            self.srl_obs_buf[env_ids] = srl_stacked_obs
            self.srl_obs_mirrored_buf[env_ids] = srl_mirrored_stacked_obs
            self.srl_priv_extra_obs_buf[env_ids] = priv_extra_obs
            self.srl_priv_extra_mirrored_obs_buf[env_ids] = priv_extra_obs_mirrored
            self.teacher_srl_obs_buf[env_ids] = teacher_srl_stacked_obs

            self.potentials[env_ids] = potentials
            self.prev_potentials[env_ids] = prev_potentials
        return

    def _compute_env_obs(self, env_ids=None):
        virtual_load_cell = self._virtual_load_cell_from_dof(self._dof_pos, self._dof_vel)
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            load_cell_sensor = virtual_load_cell
            dof_force_tensor = self.dof_force_tensor
            srl_root_states = self.srl_root_states.clone()
            progress_buf = self.progress_buf
            phase_buf = self.phase_buf
            initial_dof_pos = self.initial_srl_dof_pos
            actions = self.actions
            targets = self.targets
            potentials = self.potentials
            target_vel_x = self.target_vel_x
            target_yaw   = self.target_yaw
            right_thigh_rot = self.right_thigh_states[:,3:7]
            right_shin_rot = self.right_shin_states[:,3:7]
            left_thigh_rot = self.left_thigh_states[:,3:7]
            left_shin_rot = self.left_shin_states[:,3:7]
            humanoid_legs_rot = torch.stack([right_thigh_rot, right_shin_rot, left_thigh_rot, left_shin_rot], dim=1)
            current_time = self.progress_buf * self.control_dt
            env_ids_flat = torch.arange(self.num_envs, device=self.device)
            traj_points_world = self._traj_gen.get_observation_points(env_ids_flat, current_time, self._traj_sample_times)
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
            load_cell_sensor = virtual_load_cell[env_ids]
            dof_force_tensor = self.dof_force_tensor[env_ids]
            srl_root_states = self.srl_root_states[env_ids,:].clone()
            progress_buf = self.progress_buf[env_ids]
            phase_buf = self.phase_buf[env_ids]
            initial_dof_pos = self.initial_srl_dof_pos[env_ids]
            actions = self.actions[env_ids]
            targets = self.targets[env_ids]
            potentials = self.potentials[env_ids]
            target_vel_x = self.target_vel_x[env_ids]
            target_yaw  = self.target_yaw[env_ids]
            right_thigh_rot = self.right_thigh_states[env_ids][:,3:7]
            right_shin_rot = self.right_shin_states[env_ids][:,3:7]
            left_thigh_rot = self.left_thigh_states[env_ids][:,3:7]
            left_shin_rot = self.left_shin_states[env_ids][:,3:7]
            humanoid_legs_rot = torch.stack([right_thigh_rot, right_shin_rot, left_thigh_rot, left_shin_rot], dim=1)
            current_time = self.progress_buf[env_ids] * self.control_dt
            traj_points_world = self._traj_gen.get_observation_points(env_ids, current_time, self._traj_sample_times)
        humanoid_obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs,
                                            load_cell_sensor, self._humanoid_load_cell_obs, traj_points_world)
        srl_dof_num = self.srl_actions_num + self.srl_free_actions_num
        srl_obs, teacher_srl_obs, potentials, prev_potentials,  = compute_srl_observations(phase_buf, initial_dof_pos, srl_root_states, root_states,
                                                                          humanoid_legs_rot, dof_pos[:, -srl_dof_num:], dof_vel[:, -srl_dof_num:],
                                                                          load_cell_sensor, target_yaw, dof_force_tensor[:, -srl_dof_num:], 
                                                                          actions[:, -(self.srl_actions_num):], self.obs_scales_tensor, 
                                                                          targets, potentials, self.control_dt, target_vel_x, self.gait_period )
        srl_obs_mirrored = compute_srl_observations_mirrored(phase_buf, self.mirror_act_srl_mat, initial_dof_pos, srl_root_states, root_states,
                                                             humanoid_legs_rot, dof_pos[:, -srl_dof_num:], dof_vel[:, -srl_dof_num:],
                                                            load_cell_sensor,  target_yaw, dof_force_tensor[:, -srl_dof_num:], 
                                                            actions[:, -(self.srl_actions_num):], self.obs_scales_tensor, 
                                                            targets, target_vel_x, self.gait_period )
        priv_extra_obs = compute_priv_extra_observations(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs,)
        priv_extra_obs_mirrored = compute_priv_extra_observations_mirrored(root_states, dof_pos, dof_vel,
                                            key_body_pos, self._local_root_obs,)

        if self._design_param_obs:
            design_param = self.design_param
            design_param = design_param.unsqueeze(0).repeat(obs.shape[0], 1)
            obs = torch.cat([obs, design_param], dim=1)
            obs_mirrored = torch.cat([obs_mirrored, design_param], dim=1)
        return humanoid_obs, srl_obs, srl_obs_mirrored, priv_extra_obs, priv_extra_obs_mirrored, teacher_srl_obs, potentials, prev_potentials,

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
        # [humanoid_joint, free_joint, srl_joint]
        total_action_num = self.humanoid_actions_num+self.srl_free_actions_num+self.srl_actions_num
        _action = torch.zeros([self.num_envs, total_action_num ], device=self.device)
        _action[:, :self.humanoid_actions_num] = self.actions[:, :self.humanoid_actions_num]
        _action[:, -self.srl_actions_num:] = self.actions[:, -self.srl_actions_num:]
        if self._force_control:
            pd_tar = self._action_to_pd_targets(_action)
            torques = self.p_gains*(pd_tar - self._dof_pos) - self.d_gains*self._dof_vel
            self.torques = torch.clip(torques, -self.torque_limits, self.torque_limits).view(self.torques.shape)
            self.torques[:,:self.get_humanoid_action_size()+self.get_srl_free_action_size()] = 0.00
            # --- virtual interaction force: F = -k*q - c*qdot, clamp to max ---
            # q  = self._dof_pos[:, self.srl_virtual_damping_ids]
            # qd = self._dof_vel[:, self.srl_virtual_damping_ids]
            # Fv = - self.hri_virtual_stiffness * q - self.hri_virtual_damping * qd
            # Fv = torch.clamp(Fv, -self.hri_virtual_force_max, self.hri_virtual_force_max)
            # self.torques[:, self.srl_virtual_damping_ids] = Fv
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        if self._pd_control:
            pd_tar = self._action_to_pd_targets(_action) # pd_tar.shape: [num_actors, num_dofs]
            if self._force_control:
                pd_tar[:,self.get_humanoid_action_size():] = 0.00
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        return

    def post_physics_step(self):
        self.progress_buf += 1
        self.phase_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        # TODO: Task Randomization
        # self.set_task_target()

        self._dof_vel_prev = self._dof_vel.clone()
        self.extras["terminate"] = self._terminate_buf

        # SRL reward & Obs
        self.extras["srl_rewards"] = self.srl_rew_buf.to(self.rl_device)
        self.extras["x_velocity"] = self.obs_buf[:,7]                            

        # plotting
        if self.viewer != None:
            self.extras["srl_torques"] = self.torques[0, -self.srl_actions_num:].to(self.rl_device)
            self.extras["root_pos"] = self._root_states[0, 0:3].to(self.rl_device)
            srl_end_body_pos = self._rigid_body_pos[0, self._srl_end_ids, :]
            srl_end_body_vel = self._rigid_body_vel[0, self._srl_end_ids, :]
            self.extras['srl_end_pos'] = srl_end_body_pos
            self.extras['srl_end_vel'] = srl_end_body_vel
            key_body_pos = self._rigid_body_pos[0, self._key_body_ids, :]
            self.extras['key_body_pos'] = key_body_pos
            self.extras['dof_pos'] = self._dof_pos[0].to(self.rl_device)
            self.extras['load_cell'] = self.vec_sensor_tensor[0,self.load_cell_ssidx,:].to(self.rl_device)
            self.extras['load_cell_fd'] = self.vec_sensor_tensor[0,self.load_cell_ssidx_fd,:].to(self.rl_device)
            self.extras['right_srl_end_sensor'] = self.vec_sensor_tensor[0,self.right_srl_end_ssidx,:].to(self.rl_device)
            self.extras["target_yaw"] = self.target_yaw
            # virtual_ids = [
            #     int(self.srl_virtual_damping_x_idx),
            #     int(self.srl_virtual_damping_y_idx),
            #     int(self.srl_virtual_damping_z_idx),
            # ]
            vpos0 = self._dof_pos[0, self.srl_virtual_damping_ids]
            self.extras["srl_virtual_passive_pos"] = vpos0.to(self.rl_device)
            virtual_load_cell = self._virtual_load_cell_from_dof(self._dof_pos, self._dof_vel)
            self.extras["srl_virtual_load_cell"] = virtual_load_cell[0].to(self.rl_device)
            self.extras["dof_force"] = self.dof_force_tensor[0].to(self.rl_device)
        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return


    def set_task_target(self):
        self.target_vel_x[:], self.target_pelvis_height[:], self.target_ang_vel_z[:], self.target_yaw[:] = set_task_target(self.target_vel_x,
                                                                                                       self.target_pelvis_height,
                                                                                                       self.target_ang_vel_z,
                                                                                                       self.target_yaw,
                                                                                                       self.progress_buf,
                                                                                                       max_episode_length=self.max_episode_length)        

    def _compute_srl_motor_load_costs(self):
        """Compute SRL motor near-peak penalty and rated/thermal/power proxies.

        Key ideas
        - Peak torque (e.g., 320 Nm): short-duration limit -> penalize being near peak (instant + short window).
        - Rated torque (e.g., 85 Nm): “continuous” / thermal-limited operating point -> penalize sustained high torque
        via a slow thermal proxy built from (tau / tau_rated)^2.

        Expected attributes (already in your class, or add in __init__):
        - self.srl_actions_num: int
        - self.torques: (N, num_dof) applied torques after clipping
        - self._dof_vel: (N, num_dof) dof velocities (rad/s)
        - self.srl_tau_limits: (6,) peak torque limits used for clipping (per SRL joint)
        - self.srl_peak_start_ratio: float in (0,1) (start penalizing near peak)
        - self._srl_peak_decay: float (window decay, computed from peak_window_tau_s)
        - self.srl_peak_ratio_window: (N,) running window state
        - self.srl_tau2_ema: (N,) thermal state
        - self.srl_power_ema: (N,) power state
        - self._srl_thermal_gamma: float (thermal EMA decay, computed from thermal_tau_s)
        - self.srl_omega_ref: float (rad/s) for power normalization

        Optional (recommended) attributes to represent *rated* torque:
        - self.srl_rated_ratio: float, default = 85/320 (rated/peak)
        - self.srl_thermal_start: float, default ~0.7 (start penalizing when thermal state is high)
            NOTE: self.srl_tau2_ema tends to 1.0 if you keep tau≈tau_rated for long enough.
        """
        # Applied actuator torques for SRL motors
        tau = self.torques[:, -self.srl_actions_num:]     # (N, 6)
        omega = self._dof_vel[:, -self.srl_actions_num:]  # (N, 6)
        tau_abs = torch.abs(tau)

        # ------------------------------------------------------------
        # 1) Peak cost (normalized by peak limit)
        # ------------------------------------------------------------
        tau_peak = self.srl_tau_limits.unsqueeze(0).clamp(min=1e-6)  # (1, 6)
        tau_ratio_peak = tau_abs / tau_peak                           # (N, 6) in [0,1] after clip

        peak_ratio_inst = torch.max(tau_ratio_peak, dim=1).values      # (N,)
        self.srl_peak_ratio_window = torch.maximum(
            peak_ratio_inst, self.srl_peak_ratio_window * self._srl_peak_decay
        )

        start = float(self.srl_peak_start_ratio)
        peak_cost = torch.clamp((self.srl_peak_ratio_window - start) / (1.0 - start), min=0.0) ** 2

        # ------------------------------------------------------------
        # 2) Thermal cost (normalized by *rated* torque, slow EMA)
        #    thermal_state ~ EMA(mean((tau/tau_rated)^2))
        #    - If tau ~= tau_rated steadily, thermal_state -> 1
        # ------------------------------------------------------------
        rated_ratio = getattr(self, "srl_rated_ratio", None)
        if rated_ratio is None:
            rated_ratio = 85.0 / 320.0  # default based on your vendor numbers

        tau_rated = (tau_peak * float(rated_ratio)).clamp(min=1e-6)    # (1, 6)
        tau_ratio_rated = tau_abs / tau_rated                          # (N, 6) can be >1

        tau2_mean = torch.mean(tau_ratio_rated ** 2, dim=1)            # (N,) = 1 at rated (on average)
        g = float(self._srl_thermal_gamma)
        self.srl_tau2_ema = g * self.srl_tau2_ema + (1.0 - g) * tau2_mean

        # Map thermal state -> penalty (don’t punish small values too much)
        thermal_start = getattr(self, "srl_thermal_start", 0.7)
        thermal_cost = torch.clamp((self.srl_tau2_ema - float(thermal_start)) / (1.0 - float(thermal_start)), min=0.0) ** 2

        # ------------------------------------------------------------
        # 3) Optional power proxy (normalized by peak*omega_ref), slow EMA
        # ------------------------------------------------------------
        denom = (tau_peak * float(self.srl_omega_ref)).clamp(min=1e-6)  # (1, 6)
        power_mean = torch.mean(torch.abs(tau * omega) / denom, dim=1)  # (N,)
        self.srl_power_ema = g * self.srl_power_ema + (1.0 - g) * power_mean
        power_cost = self.srl_power_ema

        return peak_cost, thermal_cost, power_cost

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
def quat_to_euler_ypr(q):
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
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, 
                                  local_root_obs, load_cell, humanoid_load_cell_obs, 
                                  traj_points_world):
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


    num_samples = 3
    heading_inv_expand = heading_rot.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 4)
    delta_pos = traj_points_world - root_pos.unsqueeze(1)
    delta_pos_flat = delta_pos.reshape(-1, 3)
    local_traj_points = my_quat_rotate(heading_inv_expand, delta_pos_flat)
    local_traj_points = local_traj_points.view(root_pos.shape[0], num_samples, 3)
    traj_obs = local_traj_points[..., 0:2].reshape(root_pos.shape[0], -1)

    # root_h 1; root_rot_obs 6; local_root_vel 3 ; local_root_ang_vel 3 ; dof_obs 58; dof_vel 36 ; load_cell_force 6, flat_local_key_pos 12
    obs = torch.cat((root_h,              # 1
                     root_rot_obs,        # 6
                     local_root_vel,      # 3
                     local_root_ang_vel,  # 3
                     humanoid_dof_obs,    # 52
                     humanoid_dof_vel,    # 28
                     load_cell_force,     # 6
                     flat_local_key_pos,  # 12
                     traj_obs,            # 6 command, 
                     ), dim=-1)
    return obs


# @torch.jit.script
def compute_srl_observations(
    phase_buf,
    default_joint_pos,
    root_states ,
    humanoid_root_states,
    humanoid_legs_rot,
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
    humanoid_root_rot = humanoid_root_states[:, 3:7]

    # base 高度
    root_h = root_pos[:, 2:3]
    euler = quat_to_euler_ypr(root_rot)
    
    # Humanoid腿部运动
    humanoid_root_rot_inv = quat_conjugate(humanoid_root_rot)
    humanoid_root_rot_inv_expanded = humanoid_root_rot_inv.unsqueeze(1).expand(-1, 4, -1)
    humanoid_legs_rel_rot = quat_mul(humanoid_root_rot_inv_expanded, humanoid_legs_rot)
    right_thigh_rel_euler = quat_to_euler_ypr(humanoid_legs_rel_rot[ :, 0, :])
    right_shin_rel_euler  = quat_to_euler_ypr(humanoid_legs_rel_rot[ :, 1, :])
    left_thigh_rel_euler  = quat_to_euler_ypr(humanoid_legs_rel_rot[ :, 2, :])
    left_shin_rel_euler   = quat_to_euler_ypr(humanoid_legs_rel_rot[ :, 3, :])

    right_thigh_rel_pitch = right_thigh_rel_euler[:,1:2] 
    right_shin_rel_pitch  = right_shin_rel_euler[:,1:2]
    left_thigh_rel_pitch  = left_thigh_rel_euler[:,1:2]
    left_shin_rel_pitch   = left_shin_rel_euler[:,1:2]

    yaw   = euler[:, 0]
    delta   = target_yaw - yaw                 
    yaw_err = torch.atan2(torch.sin(delta), torch.cos(delta))  # (-pi, pi]
    euler_err = torch.zeros_like(euler)
    euler_err[:, 0] = yaw_err        # yaw 误差（wrap 后）
    euler_err[:, 1] = -euler[:, 1]   
    euler_err[:, 2] = -euler[:, 2]   
    
    srl_rel_rot = quat_mul(humanoid_root_rot_inv, root_rot)
    humanoid_euler_err = quat_to_euler_ypr(srl_rel_rot)
    
    # 将线速度/角速度旋转到局部坐标
    local_root_vel     = quat_rotate_inverse(root_rot, root_vel)
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)

   
    # SRL loadcell 是负载力传感器 
    load_cell_force = load_cell

    # 主体关节位置编码（humanoid + SRL）
    # dof_obs = dof_to_obs(dof_pos)  
    srl_dof_obs   = dof_pos[:,-6:] 
    srl_dof_obs   = srl_dof_obs - default_joint_pos
    srl_dof_vel   = dof_vel[:,-6:]
    srl_dof_force = dof_force_tensor[:,-6:] 

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
                     srl_dof_obs  * obs_scales[2],       # 6    10:15
                     srl_dof_vel  * obs_scales[3],       # 6    16:21
                     actions  ,                          # 6    22:27
                     sin_phase,                          # 1    28
                     cos_phase,                          # 1    29
                     humanoid_euler_err,                 # 3    30:32
                     load_cell_force * obs_scales[4],    # 6    33:38
                     right_thigh_rel_pitch,              # 3    39
                     right_shin_rel_pitch,               # 3    40
                     left_thigh_rel_pitch,               # 3    41
                     left_shin_rel_pitch,                # 3    42
                    ), dim=-1)
    
    teacher_obs = torch.cat((root_h,                             # 1    0
                     local_root_vel ,                    # 3    1:3
                     local_root_ang_vel ,                # 3    4:6
                     euler_err,                          # 3    7:9
                     srl_dof_obs  * obs_scales[2],       # 6    10:15
                     srl_dof_vel  * obs_scales[3],       # 6    16:21
                     actions  ,                          # 6    22:27
                     sin_phase,                          # 1    28
                     cos_phase,                          # 1    29
                    ), dim=-1)
    return obs , teacher_obs, potentials, prev_potentials_new

# @torch.jit.script
def compute_srl_observations_mirrored(
    phase_buf,
    mirror_mat,
    default_joint_pos,
    root_states ,
    humanoid_root_states,
    humanoid_legs_rot,
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
    humanoid_root_rot = humanoid_root_states[:, 3:7]

    # base 高度
    root_h = root_pos[:, 2:3]

    # 将线速度/角速度旋转到局部坐标
    local_root_vel     = quat_rotate_inverse(root_rot, root_vel)
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)

    # SRL loadcell 是负载力传感器 
    load_cell_force = load_cell

    # 主体关节位置编码（humanoid + SRL）
    # dof_obs = dof_to_obs(dof_pos)  
    srl_dof_obs   = dof_pos[:,-6:] 
    srl_dof_obs   = srl_dof_obs - default_joint_pos
    srl_dof_vel   = dof_vel[:,-6:]
    srl_dof_force = dof_force_tensor[:,-6:]

    euler = quat_to_euler_ypr(root_rot)
    humanoid_euler = quat_to_euler_ypr(humanoid_root_rot)

    # Humanoid腿部运动
    humanoid_root_rot_inv = quat_conjugate(humanoid_root_rot)
    humanoid_root_rot_inv_expanded = humanoid_root_rot_inv.unsqueeze(1).expand(-1, 4, -1)
    humanoid_legs_rel_rot = quat_mul(humanoid_root_rot_inv_expanded, humanoid_legs_rot)
    right_thigh_rel_euler = quat_to_euler_ypr(humanoid_legs_rel_rot[ :, 0, :])
    right_shin_rel_euler  = quat_to_euler_ypr(humanoid_legs_rel_rot[ :, 1, :])
    left_thigh_rel_euler  = quat_to_euler_ypr(humanoid_legs_rel_rot[ :, 2, :])
    left_shin_rel_euler   = quat_to_euler_ypr(humanoid_legs_rel_rot[ :, 3, :])

    right_thigh_rel_pitch = right_thigh_rel_euler[:,1:2] 
    right_shin_rel_pitch  = right_shin_rel_euler[:,1:2]
    left_thigh_rel_pitch  = left_thigh_rel_euler[:,1:2]
    left_shin_rel_pitch   = left_shin_rel_euler[:,1:2]

    yaw   = euler[:, 0]
    delta   = target_yaw - yaw                 
    yaw_err = torch.atan2(torch.sin(delta), torch.cos(delta))  # (-pi, pi]
    euler_err = torch.zeros_like(euler)
    euler_err[:, 0] = yaw_err        # yaw 误差（wrap 后）
    euler_err[:, 1] = -euler[:, 1]   
    euler_err[:, 2] = -euler[:, 2]  

    srl_rel_rot = quat_mul(humanoid_root_rot_inv, root_rot)
    humanoid_euler_err = quat_to_euler_ypr(srl_rel_rot)

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
    load_cell_force[:,1] =  -load_cell_force[:,1]
    load_cell_force[:,3] =  -load_cell_force[:,3]
    load_cell_force[:,5] =  -load_cell_force[:,5]
    humanoid_euler_err[:,0] = -humanoid_euler_err[:,0]
    humanoid_euler_err[:,2] = -humanoid_euler_err[:,2]

    # right_thigh_rel_euler[:,0] = -right_thigh_rel_euler[:,0]
    # right_thigh_rel_euler[:,2] = -right_thigh_rel_euler[:,2]
    # right_shin_rel_euler[:,0]  = -right_shin_rel_euler[:,0]
    # right_shin_rel_euler[:,2]  = -right_shin_rel_euler[:,2]
    # left_thigh_rel_euler[:,0]  = -left_thigh_rel_euler[:,0]
    # left_thigh_rel_euler[:,2]  = -left_thigh_rel_euler[:,2]
    # left_shin_rel_euler[:,0]  = -left_shin_rel_euler[:,0]
    # left_shin_rel_euler[:,2]  = -left_shin_rel_euler[:,2]

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
                     srl_dof_obs[:,-6:] * obs_scales[2],    # 6
                     srl_dof_vel[:,-6:] * obs_scales[3],        # 6
                     actions[:,-6:] ,
                     sin_phase,    
                     cos_phase,     
                     humanoid_euler_err,                  
                     load_cell_force * obs_scales[4],       # 6   
                     right_thigh_rel_pitch,              
                     right_shin_rel_pitch,      
                     left_thigh_rel_pitch,               
                     left_shin_rel_pitch,                      
                    ), dim=-1)
    return obs  



def compute_priv_extra_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs) # root朝向

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    # root_h 1; root_rot_obs 6; local_root_vel 3 ; local_root_ang_vel 3 ; dof_obs 58; dof_vel 36 ; load_cell_force 6, flat_local_key_pos 12
    obs = torch.cat((flat_local_key_pos,  # 12
                     ), dim=-1)
    return obs

def compute_priv_extra_observations_mirrored(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs) # root朝向

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    # Mirrored
    flat_local_key_pos_mirrored = flat_local_key_pos.clone()
    flat_local_key_pos_mirrored[:,0:3] = flat_local_key_pos[:,3:6]
    flat_local_key_pos_mirrored[:,3:6] = flat_local_key_pos[:,0:3]
    flat_local_key_pos_mirrored[:,6:9] = flat_local_key_pos[:,9:12]
    flat_local_key_pos_mirrored[:,9:12] = flat_local_key_pos[:,6:9]

    # root_h 1; root_rot_obs 6; local_root_vel 3 ; local_root_ang_vel 3 ; dof_obs 58; dof_vel 36 ; load_cell_force 6, flat_local_key_pos 12
    obs = torch.cat((flat_local_key_pos_mirrored,  # 12
                     ), dim=-1)
    return obs

def compute_humanoid_reward(obs_buf, dof_pos, dof_vel, dof_vel_prev, humanoid_target_point, humanoid_root_pos, load_cell_sensor):
    # --- Task Command ---
    target_pelvis_height = obs_buf[:, -1] 
    target_ang_vel_z = obs_buf[:, -2]
    target_vel_x = obs_buf[:, -3]

    # --- Target Point Tracking ---
    dist_sq = torch.sum((humanoid_target_point[..., 0:2] - humanoid_root_pos[..., 0:2]) ** 2, dim=-1)
    pos_reward = 1 * torch.exp(-1.0 * dist_sq)

    # --- Alive Reward ---
    alive_reward_coef = torch.where(target_vel_x < 0.1, 4*torch.ones_like(target_vel_x), torch.ones_like(target_vel_x))
    alive_reward = 1 * torch.ones_like(dist_sq)


    # --- Target Velocity ---
    root_vel = obs_buf[:, 7:10] 
    root_target_vel = torch.zeros((root_vel.shape[0], 3), device=root_vel.device)
    root_target_vel[:, 0] = target_vel_x  
    vel_error_vec = root_vel - root_target_vel
    vel_tracking_reward =  1 *  torch.exp(-3 * torch.norm(vel_error_vec, dim=-1))  # α = 1.5

    # --- Pelvis Orientation ---
    pelvis_upright_z = obs_buf[:,6]
    pelvis_penalty = (1.0 - pelvis_upright_z).clamp(min=0.0)
    pelvis_penalty = 5 * pelvis_penalty

    # --- Standing Joint Pose ---
    humanoid_dof_pos = dof_pos[:,0:28]
    dof_pos_cost = 0.25*torch.sum(humanoid_dof_pos ** 2, dim=-1)
    # standing
    gait_phase_penalty_coef = torch.where(target_vel_x > 0.1, torch.zeros_like(target_vel_x), torch.ones_like(target_vel_x))
    dof_pos_cost = gait_phase_penalty_coef * dof_pos_cost

    # --- Torso DOF velocity cost ---
    torso_dof_vel = dof_vel[:,0:6]
    dof_vel_cost = 0.01 * torch.sum(torso_dof_vel ** 2, dim=-1)

    # --- Dof Acc cost ---
    humanoid_dof_acc = dof_vel[:,0:6] - dof_vel_prev[:,0:6]
    dof_acc_cost = 0.01 * torch.sum(humanoid_dof_acc ** 2, dim=-1)

    # --- Load Cell Force ---
    load_cell_force = load_cell_sensor[:, 0:3]
    Fx, Fy, Fz = load_cell_force[:,0], load_cell_force[:,1], load_cell_force[:,2]
    # 支撑：至少 30N，超过 100N 不再额外追求（可按你想要的力度改）
    Fz_min  = 0.30   # 30N
    Fz_cap  = 1.00   # 100N
    Fz_tol  = 0.20   # 20N
    # 低于 Fz_min 才罚（鼓励“给够就行”）
    lack = torch.clamp(Fz_min - Fz, min=0.0)
    support_cost = torch.tanh(lack / Fz_tol) ** 2
    # 反向力（强罚）
    neg_cost = torch.tanh(torch.clamp(-Fz, min=0.0) / 0.10) ** 2   # 10N
    # 剪切：稍微放宽，并可用剪切比（推荐）
    shear = torch.sqrt(Fx**2 + Fy**2)
    shear_dead  = 0.20   # 20N
    shear_scale = 0.20   # 20N
    shear_cost = torch.tanh(torch.clamp(shear - shear_dead, min=0.0) / shear_scale) ** 2
    # 过大力（硬约束按 400N）
    f_norm = torch.sqrt(Fx**2 + Fy**2 + Fz**2)
    F_hard = 4.00   # 400N
    Fhs    = 0.50   # 50N ramp
    hard_cost = torch.tanh(torch.clamp(f_norm - F_hard, min=0.0) / Fhs) ** 2
    # 人机交互力惩罚项
    contact_force_cost = (
        1.0 * support_cost +
        0.8 * shear_cost +
        2.0 * neg_cost +
        2.0 * hard_cost
    )

    # FIXME: HUMANOID奖励函数设定
    total_reward = vel_tracking_reward \
                   + 0.5*alive_reward \
                   - dof_pos_cost \
                   - dof_vel_cost \
                   - pelvis_penalty \
                   - dof_acc_cost \
                   - 0.2*contact_force_cost
    humanoid_task_reward = vel_tracking_reward
    return total_reward, humanoid_task_reward



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
    humanoid_task_reward,
    srl_obs_num: int = 0,
    alive_reward_scale: float = 0,
    humanoid_share_reward_scale: float = 0,
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
    srl_motor_cost: float = 0,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float,  float, float, float, float, float, float, float, float, float, float, float, float, float) -> Tuple[Tensor, ]

    # obs = torch.cat((root_h,                             # 1    0
    #                  local_root_vel ,                    # 3    1:3
    #                  local_root_ang_vel ,                # 3    4:6
    #                  euler_err,                          # 3    7:9
    #                  srl_dof_obs  * obs_scales[2],       # 6    10:15
    #                  srl_dof_vel  * obs_scales[3],       # 6    16:21
    #                  actions  ,                          # 6    22:27
    #                  sin_phase,                          # 1    28
    #                  cos_phase,                          # 1    29
    #                  humanoid_euler_err,                 # 3    30:32
    #                  load_cell_force * obs_scales[4],    # 6    33:38
    #                  right_thigh_rel_euler,              # 3    39
    #                  right_shin_rel_euler,               # 3    40
    #                  left_thigh_rel_euler,               # 3    41
    #                  left_shin_rel_euler,                # 3    42
    #                 ), dim=-1)

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
    vel_tracking_reward =  torch.exp(-4 * torch.norm(vel_error_vec, dim=-1))  # α = 1.5

    # --- Torques cost ---
    torques_cost = 0 * torch.sum(actions ** 2, dim=-1)

    # --- DOF deviation cost ---
    srl_joint_pos = obs_buf[:, 10:16].clone()  
    srl_joint_pos[:,0] *= 3 
    srl_joint_pos[:,3] *= 3
    dof_pos_cost = torch.sum(srl_joint_pos ** 2, dim=-1)

    # --- DOF velocity cost ---
    dof_vel = obs_buf[:, 16:16+(actions.shape[1])]
    dof_vel_cost = torch.sum(dof_vel ** 2, dim=-1)

    # --- DOF acceleration cost ---
    dof_vel_prev = obs_buf[:, 16+srl_obs_num:16+srl_obs_num+actions.shape[1]]  # 前一帧速度
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
    # orientation_reward = torch.exp(-20 * ori_error   ) 
    # 分轴权重
    w_roll  = 6.0
    w_pitch = 6.0
    w_yaw   = 3.0
    orientation_reward = - (w_yaw  * angle_diff[:, 0]**2 + w_pitch * angle_diff[:, 1]**2 + w_roll * angle_diff[:, 2]**2 )
    # TODO: 将朝向奖励改为惩罚
    # orientation_reward = - 3*torch.sum((angle_diff) ** 2, dim=-1)

    # --- Pelvis height ---
    pelvis_height = obs_buf[:,0]
    pelvis_height_error = pelvis_height - target_pelvis_height
    pelvis_height_reward =  torch.exp(-12 * (10* pelvis_height_error) **2 ) 
    pelvis_height_penalty =  (10* pelvis_height_error) **2 

    # --- Pelvis angular rate ---
    root_ang_vel = obs_buf[:, 4:7]
    root_target_ang_vel = torch.zeros((root_ang_vel.shape[0], 3), device=root_ang_vel.device)
    root_target_ang_vel[:, 2] = target_ang_vel_z   
    root_ang_vel[:,2] = root_ang_vel[:,2] 
    ang_vel_error_vec = root_ang_vel - root_target_ang_vel
    ang_vel_tracking_reward = torch.exp(-3 * torch.norm(ang_vel_error_vec, dim=-1))   
   
    # --- Interaction Force ---
    # load_cell_force = obs_buf[:, 33:36]
    # f = torch.norm(load_cell_force, dim=-1)          # kN
    # F0 = 0.30   # deadband: <=50N basically no penalty (conservative)
    # Fs = 0.30   # scale: 50N scale of growth
    # F_hard = 1.50  # hard safety threshold: 300N
    # Fhs    = 0.5   # 100N  hard penalty ramp scale
    # x = torch.clamp((f - F0) / Fs, min=0.0)
    # soft = torch.tanh(x) ** 2
    # xh = torch.clamp((f - F_hard) / Fhs, min=0.0)
    # hard = torch.tanh(xh) ** 2
    # w_soft = 0.5   # conservative but not too dominating; can try 0.2~0.8
    # w_hard = 2.0   # makes big-force episodes unattractive
    # contact_force_cost = w_soft * soft + w_hard * hard

    # --- Interaction Force (support upward Z around +50N) ---
    load_cell_force = obs_buf[:, 33:36]
    Fx, Fy, Fz = load_cell_force[:,0], load_cell_force[:,1], load_cell_force[:,2]
    # 支撑：至少 30N，超过 100N 不再额外追求（可按你想要的力度改）
    Fz_min  = 0.30   # 30N
    Fz_cap  = 1.00   # 100N
    Fz_tol  = 0.20   # 20N
    # 低于 Fz_min 才罚（鼓励“给够就行”）
    lack = torch.clamp(Fz_min - Fz, min=0.0)
    support_cost = torch.tanh(lack / Fz_tol) ** 2
    # 反向力（强罚）
    neg_cost = torch.tanh(torch.clamp(-Fz, min=0.0) / 0.10) ** 2   # 10N
    # 剪切：稍微放宽，并可用剪切比（推荐）
    shear = torch.sqrt(Fx**2 + Fy**2)
    shear_dead  = 0.20   # 20N
    shear_scale = 0.20   # 20N
    shear_cost = torch.tanh(torch.clamp(shear - shear_dead, min=0.0) / shear_scale) ** 2
    # 过大力（硬约束按 400N）
    f_norm = torch.sqrt(Fx**2 + Fy**2 + Fz**2)
    F_hard = 4.00   # 400N
    Fhs    = 0.50   # 50N ramp
    hard_cost = torch.tanh(torch.clamp(f_norm - F_hard, min=0.0) / Fhs) ** 2
    # 人机交互力惩罚项
    contact_force_cost = (
        1.0 * support_cost +
        0.8 * shear_cost +
        2.0 * neg_cost +
        2.0 * hard_cost
    )

    # --- No fly --- 
    contact_threshold = 0.040  
    left_foot_height = srl_end_body_pos[:, 0, 2]  # 获取左脚的位置 
    right_foot_height = srl_end_body_pos[:, 1, 2]  # 获取右脚的位置 
    # walking
    no_feet_on_ground = (left_foot_height > contact_threshold) & (right_foot_height > contact_threshold)
    # standing
    no_both_feet_on_ground = (left_foot_height > contact_threshold) | (right_foot_height > contact_threshold)
    both_feet_on_ground = (left_foot_height < contact_threshold) & (right_foot_height < contact_threshold)
    fly_idx = torch.where(target_vel_x < 0.1, no_both_feet_on_ground, no_feet_on_ground)
    standing_idx = torch.where(target_vel_x < 0.1, both_feet_on_ground, torch.zeros_like(no_feet_on_ground))
    no_fly_penalty = torch.where(fly_idx, torch.ones_like(no_feet_on_ground) * no_fly_penalty_scale, torch.zeros_like(no_feet_on_ground))
    standing_reward = torch.where(standing_idx, -0.5*torch.ones_like(no_feet_on_ground) * no_fly_penalty_scale, torch.zeros_like(no_feet_on_ground))
    no_fly_penalty = no_fly_penalty + standing_reward

    # --- Feet Lateral Distance ---
    local_srl_end_body_pos = srl_end_body_pos - srl_root_pos.unsqueeze(-2)
    lateral_distance = torch.abs(local_srl_end_body_pos[:,0,1] - local_srl_end_body_pos[:,1,1])
    min_d, max_d = 0.40, 0.85 # FIXME: Lateral Distance 
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
    
    # --- foot clearance penalty ---
    # TODO: 0.3也可以work
    clearance_penalty = torch.clamp(clearance_penalty, max=1.0) 

    # --- SRL Motor Cost ---
    srl_motor_cost = torch.clamp(srl_motor_cost, max=3.0)

    # --- Total reward ---
    total_reward = humanoid_share_reward_scale * humanoid_task_reward \
        + alive_reward_scale * alive_reward  \
        + progress_reward_scale * progress_reward \
        + vel_tracking_reward_scale *  vel_tracking_reward \
        + tracking_ang_vel_reward_scale * ang_vel_tracking_reward \
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
        - contact_force_cost_scale*contact_force_cost \
        - srl_motor_cost

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

        has_fallen = torch.logical_or(fall_contact, fall_height)

        # srl fallen        
        srl_fallen = torch.where(srl_obs_buf[:, 0] < srl_termination_height, torch.ones_like(reset_buf), terminated)
        # srl unstable
        # srl_unstable = torch.where(srl_freejoint_pos[:,0] > 0.260, torch.ones_like(reset_buf), terminated)
        # srl_fallen = torch.logical_or(srl_fallen, srl_unstable)

        has_fallen = torch.logical_or(has_fallen, srl_fallen)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

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
    # for i in range(max_episode_length//height_change_period):
    #     mask = progress_buf == height_change_period * i+1
    #     height_indices = torch.randint(0, len(height_choices), (len(target_pelvis_height),), device=target_pelvis_height.device)
    #     target_pelvis_height =  torch.where(mask, unit_vel_x*height_choices[height_indices], target_pelvis_height)
    # mask = progress_buf == 1  # reset
    # target_pelvis_height =  torch.where(mask, unit_vel_x*height_choices[2], target_pelvis_height)

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
    # delta_yaw = torch.zeros_like(target_yaw)
    # ang_vel_indices = torch.randint(0, len(ang_vel_choices), (len(target_ang_vel_z),), device=target_vel_x.device)
    # yaw_indices = torch.randint(0, len(yaw_choices), (len(target_yaw),), device=target_vel_x.device)
    # for i in range(max_episode_length//step_period ):
    #     mask = progress_buf == step_period * i+1
    #     if i%2 == 0:
    #         ang_vel_indices = torch.full_like(ang_vel_indices, 0)
    #         yaw_indices = torch.full_like(yaw_indices, 0)
    #         if i  == 0:
    #             target_yaw = torch.where(mask, unit_vel_x*yaw_choices[yaw_indices], target_yaw)
    #     # turn left
    #     elif i%2 == 1:
    #         ang_vel_indices = torch.randint(1, len(ang_vel_choices), (len(target_ang_vel_z),), device=target_vel_x.device)
    #         yaw_indices = ang_vel_indices
    #     # set angular rate
    #     target_ang_vel_z =  torch.where(mask, unit_vel_x*ang_vel_choices[ang_vel_indices], target_ang_vel_z)
    #     # set yaw
    #     delta_yaw  = torch.where(mask, unit_vel_x*yaw_choices[yaw_indices], delta_yaw)
    #     # reset angular rate
    #     reset_mask = progress_buf == step_period * i+61
    #     ang_vel_indices = torch.full_like(ang_vel_indices, 0)
    #     target_ang_vel_z =  torch.where(reset_mask, unit_vel_x*ang_vel_choices[ang_vel_indices], target_ang_vel_z)
    # target_yaw = target_yaw + delta_yaw

    return target_vel_x, target_pelvis_height, target_ang_vel_z, target_yaw



