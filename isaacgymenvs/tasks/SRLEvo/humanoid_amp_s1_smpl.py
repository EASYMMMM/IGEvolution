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
        
        # 确定当前是否为测试模式 (用于控制 Stage 4 的日志打印)
        is_testing = (not headless) or force_render
        
        # 2轨迹生成器参数
        self.forward_vec_local = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        episode_duration = self.max_episode_length * self.control_dt
        self._traj_gen = TrajGenerator(self.num_envs, self.device, self.control_dt, episode_duration,
                                       speed_mean=1.0,      
                                       turn_speed_max=0.5,  # 不要太大，先设为0.5
                                       num_turns=2,
                                       train_stage=self.train_stage, # 转向次数
                                       is_test=is_testing)         #stage4日志打印标志

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
        current_time = self.progress_buf * self.control_dt
        env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 1. 获取当前目标点位置
        target_pos = self._traj_gen.get_position(env_ids, current_time)
        root_pos = self._root_states[..., 0:3]
        root_rot = self._root_states[..., 3:7]
        
        # 2. 从路径点中提取“隐式目标速度”
        # 获取 0.5s 后的未来目标点位置
        # 注意：self._traj_sample_times 在类初始化中定义为 [0.5, 1.0, 1.5]
        future_traj_points = self._traj_gen.get_observation_points(env_ids, current_time, self._traj_sample_times)
        target_pos_05s = future_traj_points[:, 0, 0:2] # 取 0.5s 后的点
        
        # 目标速度 = (未来点 - 当前目标点) 的距离 / 时间差 0.5s
        target_dist = torch.norm(target_pos_05s - target_pos[..., 0:2], dim=-1)
        is_standing_phase = target_dist < 1e-3 # 判断是否处于站立
        implied_target_speed = target_dist / 0.5 
        
        # ---------------------------------------------------------
        # Reward 项计算
        # ---------------------------------------------------------

        # (1) 位置奖励: 距离误差越小奖励越高
        dist_sq = torch.sum((target_pos[..., 0:2] - root_pos[..., 0:2]) ** 2, dim=-1)
        pos_reward = torch.exp(-2.0 * dist_sq)

        # (2) 速度方向与数值匹配奖励
        # 计算当前目标方向 (从机器人指向目标点)
        vel_dir = target_pos[..., 0:2] - root_pos[..., 0:2]
        vel_dir = torch.nn.functional.normalize(vel_dir, dim=-1)
        
        # 机器人实际水平速度
        root_vel = self._root_states[..., 7:9] 
        # 计算实际速度在目标方向上的投影
        current_vel_proj = torch.sum(root_vel * vel_dir, dim=-1)
        
        # 速度匹配奖励: 只有当速度的大小和方向都与轨迹规划一致时奖励最高
        # 这样能自动处理轨迹生成器中的变速逻辑
        vel_reward = torch.exp(-2.0 * (current_vel_proj - implied_target_speed)**2)

        # (3) 朝向奖励 (约束机器人面朝运动方向，解决螃蟹步)
        # 假设机器人正前方为本地坐标系 X 轴 [1, 0, 0]
        
        # 将本地正前方转换到世界坐标系
        forward_vec_world = my_quat_rotate(root_rot, self.forward_vec_local)
        # 计算朝向与目标方向的对齐程度 (cos theta)
        heading_alignment = torch.sum(forward_vec_world[..., 0:2] * vel_dir, dim=-1)
        heading_reward = torch.clamp(heading_alignment, min=0, max=1.0)

        # ---------------------------------------------------------
        # 阶段奖励组合
        # ---------------------------------------------------------
        if self.train_stage == 1:
            # 阶段 1: 静止站立约束 (保持原有的严苛惩罚逻辑)
            root_vel_xyz = self._root_states[..., 7:10]
            vel_penalty = torch.sum(root_vel_xyz[:, 0:2] ** 2, dim=-1)
            root_ang_vel = self._root_states[..., 10:13]
            ang_penalty = root_ang_vel[:, 2] ** 2
            dof_diff = self._dof_pos - self._initial_dof_pos
            joint_penalty = torch.sum(dof_diff ** 2, dim=-1)
            act_penalty = torch.sum(actions ** 2, dim=-1)

            self.rew_buf[:] = (
                pos_reward
                * torch.exp(-2.0 * vel_penalty)
                * torch.exp(-1.0 * ang_penalty)
                * torch.exp(-1.0 * joint_penalty)
                * torch.exp(-0.001 * act_penalty)
            )
            self.extras['amp_mask'] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # --- 分支 2: Stage 4 (混合训练) ---
        # 新增逻辑：根据 is_standing_phase 动态切换
        elif self.train_stage == 4:
            # 1. 计算站立/归位奖励 (Pose Matching)
            root_vel_s4 = self._root_states[..., 7:10]
            vel_penalty_s4 = torch.sum(root_vel_s4[:, 0:2] ** 2, dim=-1)
            
            root_ang_vel_s4 = self._root_states[..., 10:13]
            ang_penalty_s4 = root_ang_vel_s4[:, 2] ** 2 # 保持朝向
            
            dof_diff_s4 = self._dof_pos - self._initial_dof_pos
            joint_penalty_s4 = torch.sum(dof_diff_s4 ** 2, dim=-1) # 回归初始姿态
            
            act_penalty_s4 = torch.sum(actions ** 2, dim=-1)

            # 站立时的强力奖励 (乘法结构)
            stand_reward_s4 = (
                pos_reward
                * torch.exp(-2.0 * vel_penalty_s4)
                * torch.exp(-1.0 * ang_penalty_s4)
                * torch.exp(-1.0 * joint_penalty_s4)
                * torch.exp(-0.001 * act_penalty_s4)
            )

            # 2. 计算行走奖励 (Tracking)
            # 采用叠加结构
            walk_reward_s4 = pos_reward + vel_reward + heading_reward

            # 3. 动态切换
            self.rew_buf[:] = torch.where(is_standing_phase, stand_reward_s4, walk_reward_s4)
            self.extras['amp_mask'] = (~is_standing_phase).float()

        # --- 分支 3: Stage 2 & 3 (直线/曲线行走) ---
        # 推荐权重分配: 速度匹配最重要(0.4)，位置与朝向辅助(各0.3)
        # 这样既能保证“跟得上速度”，又能保证“走得准”且“面朝前”
        else:
            self.rew_buf[:] = pos_reward + vel_reward + heading_reward

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
            current_time = self.progress_buf * self.control_dt
            root_pos = self._root_states[:, 0:3]
            root_rot = self._root_states[:, 3:7]
        else:
            env_ids_flat = env_ids
            current_time = self.progress_buf[env_ids] * self.control_dt
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