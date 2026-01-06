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
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s1 wandb_project=SRL wandb_activate=True max_iterations=1500 task.env.episodeLength=2000 sim_device=cuda:1 rl_device=cuda:1   headless=True seed=$RANDOM task.env.train_stage=2  
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True num_envs=1 task.env.train_stage=2  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s1_30-13-15-51/nn/Humanoid_175_Pretrain_s1_30-13-15-57.pth  seed=$RANDOM
# Humanoid Stage2: 直线 Hybrid（175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s2 task.env.stateInit=Hybrid wandb_project=SRL wandb_activate=True max_iterations=2000 task.env.episodeLength=2500 sim_device=cuda:1 rl_device=cuda:1 num_envs=4096 headless=True seed=$RANDOM task.env.train_stage=2  checkpoint=runs/Humanoid_175_Pretrain_s1_30-13-15-51/nn/Humanoid_175_Pretrain_s1_30-13-15-57.pth
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True task.env.stateInit=Default num_envs=1 task.env.train_stage=2  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth   seed=$RANDOM
# Humanoid Stage3: 负重 直线 Hybrid（175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s3 task.env.stateInit=Hybrid wandb_project=SRL wandb_activate=True max_iterations=2500 task.env.episodeLength=2500 sim_device=cuda:1 rl_device=cuda:1 num_envs=4096 headless=True seed=$RANDOM task.env.train_stage=2  checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth  task.env.asset.assetFileName="mjcf/amp_humanoid_175_load.xml"
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True task.env.stateInit=Default num_envs=1 task.env.train_stage=2  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s3_04-14-44-35/nn/Humanoid_175_Pretrain_s3_04-14-44-40.pth    seed=$RANDOM  task.env.asset.assetFileName="mjcf/amp_humanoid_175_load.xml"


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
    task.env.contact_force_cost_scale=1.0  task.env.pelvis_height_reward_scale=2.0\
    train.params.config.dagger_anneal_k=1e-5  task.env.srl_free_actions_num=2   task.env.clearance_penalty_scale=10 \
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_06dis.xml"  train.params.config.central_critic=True ;    
# check 
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4  checkpoint=runs/SRL_Real_HRI_06-18-16-24/nn/SRL_Real_HRI_06-18-16-30.pth    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=2  task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_06dis.xml"
