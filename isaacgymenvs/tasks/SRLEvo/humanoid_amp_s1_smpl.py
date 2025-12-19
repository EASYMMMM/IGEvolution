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
from isaacgymenvs.tasks.amp.utils_amp.motion_lib_smpl import MotionLibSMPL as MotionLib

from isaacgymenvs.tasks.SRLEvo.traj_generator import SimpleCurveGenerator as TrajGenerator
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, calc_heading_quat_inv, quat_to_tan_norm, my_quat_rotate


# 维度计算: Root(13) + DofObs(19*6=114) + DofVel(57) + KeyBody(4*3=12) = 196
# NUM_AMP_OBS_PER_STEP = 196 
NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]


class HumanoidAMP_s1_Smpl(HumanoidAMP_s1_Smpl_Base):

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self._traj_sample_times = [0.5, 1.0, 1.5] # 轨迹参数
        self._num_traj_points = len(self._traj_sample_times)
        self._traj_obs_dim = self._num_traj_points * 2 
        
        self.cfg = cfg
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMP_s1_Smpl.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # 2轨迹生成器参数
        episode_duration = self.max_episode_length * self.dt
        self._traj_gen = TrajGenerator(self.num_envs, self.device, self.dt, episode_duration,
                                       speed_mean=1.0,      
                                       turn_speed_max=0.5,  # 不要太大，先设为0.5
                                       num_turns=2)         # 转向次数

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
        return super().get_obs_size() + self._traj_obs_dim

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        if self.viewer and self.debug_viz:
            self._draw_debug_traj()
        return

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
        current_time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        root_pos = self._root_states[..., 0:3]

        # 计算位置误差
        dist_sq = torch.sum((target_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
        pos_reward = torch.exp(-1.0 * dist_sq)
        self.rew_buf[:] = pos_reward
        return

    def _compute_reset(self):
        super()._compute_reset()

        # 检测是否偏离轨迹太远
        current_time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        root_pos = self._root_states[..., 0:3]

        dist_sq = torch.sum((target_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
        
        # 如果偏离超过 0.8 米，强制重置
        max_dist_sq = 0.8 * 0.8
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
        dt = self.dt
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
                                      self._local_root_obs)
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
        super().reset_idx(env_ids)
        root_pos = self._root_states[env_ids, 0:3]
        root_rot = self._root_states[env_ids, 3:7]
        self._traj_gen.reset(env_ids, root_pos, init_rot=root_rot)
        self._init_amp_obs(env_ids)
        return

    def _compute_humanoid_obs(self, env_ids=None):
        base_obs = super()._compute_humanoid_obs(env_ids)

        if env_ids is None:
            env_ids_flat = torch.arange(self.num_envs, device=self.device)
            current_time = self.progress_buf * self.dt
            root_pos = self._root_states[:, 0:3]
            root_rot = self._root_states[:, 3:7]
        else:
            env_ids_flat = env_ids
            current_time = self.progress_buf[env_ids] * self.dt
            root_pos = self._root_states[env_ids, 0:3]
            root_rot = self._root_states[env_ids, 3:7]

        traj_points_world = self._traj_gen.get_observation_points(env_ids_flat, current_time, self._traj_sample_times)

        heading_inv = calc_heading_quat_inv(root_rot)
        num_samples = self._num_traj_points
        heading_inv_expand = heading_inv.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 4)
        
        delta_pos = traj_points_world - root_pos.unsqueeze(1)
        delta_pos_flat = delta_pos.reshape(-1, 3)

        local_traj_points = my_quat_rotate(heading_inv_expand, delta_pos_flat)
        local_traj_points = local_traj_points.view(root_pos.shape[0], num_samples, 3)

        traj_obs = local_traj_points[..., 0:2].reshape(root_pos.shape[0], -1)
        obs = torch.cat([base_obs, traj_obs], dim=-1)
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
        dt = self.dt
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
                                      self._local_root_obs)
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
                                                                self._local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._root_states[env_ids], self._dof_pos[env_ids], 
                                                                    self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                    self._local_root_obs)
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