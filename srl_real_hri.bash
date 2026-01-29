export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/Mrlgpu/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
# =========== SMPL Humanoid 模型 ==========
# 训练 15_03_cmu
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl wandb_project=SRL wandb_activate=True max_iterations=1500 sim_device=cuda:1 rl_device=cuda:1 num_envs=4096 headless=True seed=$RANDOM
# 测试：
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True num_envs=1 sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/HumanoidAMP_SMPL_25-17-17-03/nn/HumanoidAMP_SMPL_25-17-17-08.pth seed=$RANDOM
# 175测试 
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True num_envs=1 sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/HumanoidAMP_SMPL_28-20-38-10/nn/HumanoidAMP_SMPL_28-20-38-17_1000.pth seed=$RANDOM

# 训练 175cm 56_01_cmu
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl wandb_project=SRL wandb_activate=True max_iterations=1500 sim_device=cuda:1 rl_device=cuda:1 num_envs=4096 headless=True seed=$RANDOM task.env.motion_file=56_01_cmu_amp.npy
# 测试 175cm 56_01_cmu
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True num_envs=1 sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/HumanoidAMP_SMPL_28-18-30-08/nn/HumanoidAMP_SMPL_28-18-30-13.pth seed=$RANDOM

# ------ Humanoid 预训练流程 ------
# Humanoid Stage1: 直线 Random （175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s1 wandb_project=SRL wandb_activate=True max_iterations=1500 task.env.episodeLength=2000 sim_device=cuda:1 rl_device=cuda:1   headless=True seed=$RANDOM task.env.train_stage=1  
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True num_envs=1 task.env.train_stage=2  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s1_30-13-15-51/nn/Humanoid_175_Pretrain_s1_30-13-15-57.pth  seed=$RANDOM
# Humanoid Stage2: 直线 Hybrid（175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s2 task.env.stateInit=Hybrid wandb_project=SRL wandb_activate=True max_iterations=2000 task.env.episodeLength=2500 sim_device=cuda:1 rl_device=cuda:1 num_envs=4096 headless=True seed=$RANDOM task.env.train_stage=2  checkpoint=runs/Humanoid_175_Pretrain_s1_30-13-15-51/nn/Humanoid_175_Pretrain_s1_30-13-15-57.pth
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True task.env.stateInit=Default num_envs=1 task.env.train_stage=2  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth   seed=$RANDOM
# Humanoid Stage3: 曲线 Hybrid（175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s3 task.env.stateInit=Hybrid wandb_project=SRL wandb_activate=True max_iterations=2500 task.env.episodeLength=4000 sim_device=cuda:1 rl_device=cuda:1 num_envs=4096 headless=True seed=$RANDOM task.env.train_stage=3  checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth   
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True task.env.stateInit=Default num_envs=1 task.env.train_stage=3  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s3_23-17-13-25/nn/Humanoid_175_Pretrain_s3_23-17-13-30.pth  task.env.episodeLength=4000  seed=$RANDOM 
# Humanoid Stage4: 混合任务（175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s4 task.env.stateInit=Hybrid wandb_project=SRL wandb_activate=True max_iterations=3500 task.env.episodeLength=4000 sim_device=cuda:1 rl_device=cuda:1 num_envs=4096 headless=True seed=$RANDOM task.env.train_stage=4  checkpoint=runs/Humanoid_175_Pretrain_s3_26-17-10-45/nn/Humanoid_175_Pretrain_s3_26-17-10-50.pth
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True task.env.stateInit=Default num_envs=1 task.env.train_stage=4  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s4_26-20-11-17/nn/Humanoid_175_Pretrain_s4_26-20-11-23.pth task.env.episodeLength=4000  seed=$RANDOM 
runs/Humanoid_175_Pretrain_s4_28-17-55-13/nn/Humanoid_175_Pretrain_s4_28-17-55-19.pth
runs/Humanoid_175_Pretrain_s4_28-21-16-06/nn/Humanoid_175_Pretrain_s4_28-21-16-11.pth
runs/Humanoid_175_Pretrain_s1_29-15-13-35/nn/Humanoid_175_Pretrain_s1_29-15-13-42.pth
runs/Humanoid_175_Pretrain_s1_29-17-32-44/nn/Humanoid_175_Pretrain_s1_29-17-33-34.pth
# =========== Real HRI ==========
# (1.4) 关节力矩更大的humanoid  
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_Real_HRI   max_iterations=2000   train.params.config.humanoid_checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth   train.params.config.srl_teacher_checkpoint=runs/SRL_Real_s4_26-17-10-46/nn/SRL_Real_s4.pth   train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5  train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=2   task.env.clearance_penalty_scale=0 task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_powerup.xml" ;    
# check √ It works!
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4  checkpoint=runs/SRL_Real_HRI_04-18-15-35/nn/SRL_Real_HRI_04-18-15-41.pth    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=2  task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_powerup.xml"

# (1.6) Central Critic + Dagger Loss
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_Real_HRI   max_iterations=2000   train.params.config.humanoid_checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth   train.params.config.srl_teacher_checkpoint=runs/SRL_Real_s4_04-21-00-44/nn/SRL_Real_s4.pth   train.params.config.dagger_loss_coef=1 train.params.config.sym_a_loss_coef=1.0   train.params.config.dagger_anneal_k=1e-5  task.env.srl_free_actions_num=2   task.env.clearance_penalty_scale=0 task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_06dis.xml"  train.params.config.central_critic=True ;    
# check  
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4  checkpoint=runs/SRL_Real_HRI_06-15-58-05/nn/SRL_Real_HRI_06-15-58-11_best.pth    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=2  task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_06dis.xml"

# (1.6) reward 调试
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    experiment=SRL_Real_HRI   max_iterations=2000   train.params.config.humanoid_checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth \
    train.params.config.srl_teacher_checkpoint=runs/SRL_Real_s4_04-21-00-44/nn/SRL_Real_s4.pth \
    train.params.config.dagger_loss_coef=1 train.params.config.sym_a_loss_coef=1.0  \
    task.env.contact_force_cost_scale=2.0  task.env.pelvis_height_reward_scale=2.0 \
    task.env.no_fly_penalty_scale=5.0  task.env.gait_similarity_penalty_scale=5.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    train.params.config.dagger_anneal_k=1e-5  task.env.srl_free_actions_num=2   task.env.clearance_penalty_scale=10 \
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_mesh.xml"  train.params.config.central_critic=True ;    
# check 
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4  checkpoint=runs/SRL_Real_HRI_07-17-41-11/nn/SRL_Real_HRI_07-17-41-17.pth    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=2  task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_mesh.xml"

# (1.20) 添加虚拟阻尼传感
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    experiment=SRL_Real_HRI   max_iterations=2000   train.params.config.humanoid_checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth \
    train.params.config.srl_teacher_checkpoint=runs/SRL_Real_s4_04-21-00-44/nn/SRL_Real_s4.pth \
    train.params.config.dagger_loss_coef=1 train.params.config.sym_a_loss_coef=1.0  \
    task.env.contact_force_cost_scale=0.5  task.env.pelvis_height_reward_scale=2.0 \
    task.env.no_fly_penalty_scale=5.0  task.env.gait_similarity_penalty_scale=5.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    train.params.config.dagger_anneal_k=1e-5  task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_mesh.xml"   
# check 
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4  checkpoint=runs/SRL_Real_HRI_20-15-35-44/nn/SRL_Real_HRI_20-15-35-50.pth     task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_mesh.xml"

# (1.20) MARL 人机reward 分配
 
# check 
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4  checkpoint=runs/SRL_Real_HRI_20-20-52-00/nn/SRL_Real_HRI_20-20-52-08.pth     task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_mesh.xml"
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    experiment=SRL_Real_HRI   max_iterations=2000   train.params.config.humanoid_checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth \
    train.params.config.srl_teacher_checkpoint=runs/SRL_Real_s4_04-21-00-44/nn/SRL_Real_s4.pth \
    train.params.config.dagger_loss_coef=1 train.params.config.sym_a_loss_coef=1.0  \
    task.env.contact_force_cost_scale=0.5  task.env.pelvis_height_reward_scale=2.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    train.params.config.dagger_anneal_k=1e-5  task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=2.0 \
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_mesh.xml"  

Error executing job with overrides: ['task=HumanoidAMPSRLGym_s1_Smpl', 'experiment=Humanoid_175_Pretrain_s1', 'wandb_project=SRL', 'wandb_activate=True', 'max_iterations=500', 'task.env.episodeLength=200', 'task.env.stateInit=Random', 'sim_device=cuda:0', 'rl_device=cuda:0', 'headless=True', 'seed=26473', 'task.env.train_stage=1']
Traceback (most recent call last):
  File "SRL_Evo_train.py", line 228, in launch_rlg_hydra
    'sigma': cfg.sigma if cfg.sigma != '' else None
  File "/home/ps/anaconda3/envs/Mrlgpu/lib/python3.7/site-packages/rl_games/torch_runner.py", line 221, in run
    self.run_train(args)
  File "/home/ps/anaconda3/envs/Mrlgpu/lib/python3.7/site-packages/rl_games/torch_runner.py", line 166, in run_train
    agent.train()
  File "/home/ps/pan1/files/zjh/zdh/IGEvolution/isaacgymenvs/learning/common_agent.py", line 134, in train
    train_info = self.train_epoch()
  File "/home/ps/pan1/files/zjh/zdh/IGEvolution/isaacgymenvs/learning/amp_continuous.py", line 193, in train_epoch
    batch_dict = self.play_steps()
  File "/home/ps/pan1/files/zjh/zdh/IGEvolution/isaacgymenvs/learning/amp_continuous.py", line 163, in play_steps
    mb_rewards = self._combine_rewards(mb_rewards, amp_rewards, mb_amp_masks)
  File "/home/ps/pan1/files/zjh/zdh/IGEvolution/isaacgymenvs/learning/amp_continuous.py", line 512, in _combine_rewards
    disc_r = disc_r * amp_masks
RuntimeError: The size of tensor a (8192) must match the size of tensor b (64) at non-singleton dimension 1