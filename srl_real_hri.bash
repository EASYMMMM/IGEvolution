export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/Mrlgpu/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# =========== Humanoid 多任务预训练 ===========
# Humanoid Stage1: 直线 Hybrid（175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s1 task.env.stateInit=Hybrid wandb_project=Humanoid_Pretrain wandb_activate=True max_iterations=3000 task.env.episodeLength=2500   headless=True seed=$RANDOM task.env.train_stage=2  
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True task.env.stateInit=Default num_envs=1 task.env.train_stage=2  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s1_18-17-16-53/nn/Humanoid_175_Pretrain_s1_18-17-17-01.pth   seed=$RANDOM
# Humanoid Stage2: 曲线 Hybrid（175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s2 task.env.stateInit=Hybrid wandb_project=Humanoid_Pretrain wandb_activate=True max_iterations=4500 task.env.episodeLength=4000 sim_device=cuda:0 rl_device=cuda:0 num_envs=4096 headless=True seed=$RANDOM task.env.train_stage=3  checkpoint=runs/Humanoid_175_Pretrain_s1_19-16-27-14/nn/Humanoid_175_Pretrain_s1_19-16-27-21.pth  
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True task.env.stateInit=Default num_envs=1 task.env.train_stage=3  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s2_19-19-59-51/nn/Humanoid_175_Pretrain_s2_19-19-59-57.pth   task.env.episodeLength=4000  seed=$RANDOM 
# Humanoid Stage3: 全向行走 Default （175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s3 wandb_project=Humanoid_Pretrain wandb_activate=True max_iterations=6500 task.env.stateInit=Default checkpoint=runs/Humanoid_175_Pretrain_s2_19-19-59-51/nn/Humanoid_175_Pretrain_s2_19-19-59-57.pth    task.env.episodeLength=4000 sim_device=cuda:0 rl_device=cuda:0   headless=True seed=$RANDOM task.env.train_stage=4  
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True num_envs=1 task.env.train_stage=4  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s3_20-13-56-52/nn/Humanoid_175_Pretrain_s3_20-13-56-58.pth      task.env.stateInit=Default seed=$RANDOM  task.env.episodeLength=4000

# =========== Humanoid 多任务预训练 ===========
# =========== (6.17 SMP mocap data) ===========
# Humanoid Stage1: 直线 Hybrid（175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s1_smp task.env.stateInit=Hybrid wandb_project=Humanoid_Pretrain wandb_activate=True max_iterations=3000 task.env.episodeLength=2500   headless=True seed=$RANDOM task.env.train_stage=2 task.env.motion_file="smp_humanoid_walk_175.npy"    
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True task.env.stateInit=Default num_envs=1 task.env.train_stage=2  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s1_smp_18-17-18-36/nn/Humanoid_175_Pretrain_s1_smp_18-17-18-42.pth   seed=$RANDOM
# Humanoid Stage2: 曲线 Hybrid（175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s2_smp task.env.stateInit=Hybrid wandb_project=Humanoid_Pretrain wandb_activate=True max_iterations=4500 task.env.episodeLength=4000 sim_device=cuda:0 rl_device=cuda:0 num_envs=4096 headless=True seed=$RANDOM task.env.train_stage=3  checkpoint=runs/Humanoid_175_Pretrain_s1_smp_18-17-18-36/nn/Humanoid_175_Pretrain_s1_smp_18-17-18-42.pth  task.env.motion_file="smp_humanoid_walk_175.npy"
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True task.env.stateInit=Default num_envs=1 task.env.train_stage=3  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s2_smp_23-15-18-20/nn/Humanoid_175_Pretrain_s2_smp_23-15-18-26.pth   task.env.episodeLength=4000  seed=$RANDOM 
# Humanoid Stage3: 全向行走 Default （175cm amp_175）
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl experiment=Humanoid_175_Pretrain_s3_smp wandb_project=Humanoid_Pretrain wandb_activate=True max_iterations=6500 task.env.stateInit=Default checkpoint=runs/Humanoid_175_Pretrain_s2_smp_23-15-18-20/nn/Humanoid_175_Pretrain_s2_smp_23-15-18-26.pth   task.env.episodeLength=4000 sim_device=cuda:0 rl_device=cuda:0   headless=True seed=$RANDOM task.env.train_stage=4  
# check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True num_envs=1 task.env.train_stage=4  sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/Humanoid_175_Pretrain_s3_smp_23-16-56-25/nn/Humanoid_175_Pretrain_s3_smp_23-16-56-31.pth      task.env.stateInit=Default seed=$RANDOM  task.env.episodeLength=4000


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
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    experiment=SRL_Real_HRI   max_iterations=2000   train.params.config.humanoid_checkpoint=runs/Humanoid_175_Pretrain_s2_30-13-53-51/nn/Humanoid_175_Pretrain_s2_30-13-53-56.pth \
    train.params.config.srl_teacher_checkpoint=runs/SRL_Real_s4_04-21-00-44/nn/SRL_Real_s4.pth \
    train.params.config.dagger_loss_coef=1 train.params.config.sym_a_loss_coef=1.0  \
      task.env.pelvis_height_reward_scale=2.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    train.params.config.dagger_anneal_k=1e-5  task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=2.0 task.env.contact_force_cost_scale=0.5\
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_mesh.xml"  
# check 添加了交互力奖励
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4  checkpoint=runs/SRL_Real_HRI_20-17-27-33/nn/SRL_Real_HRI_20-17-27-40.pth     task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_mesh.xml"
# check 没添加交互力奖励
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4  checkpoint=runs/SRL_Real_HRI_23-20-42-58/nn/SRL_Real_HRI_23-20-43-06.pth     task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_HXYK_175_mesh.xml"

# ================ V1 =================
# 使用V1版本SRL模型：
# 1. 真实电机参数
# 2. 基于电机实际参数的动力学约束
# SRL+Humanoid 训练分为三个阶段

# (5.20) obs中去除高度，线速度
# 第一阶段训练:从s4 humanoid预训练结果开始, 训练S2直线
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    train.params.config.humanoid_checkpoint=runs/Humanoid_175_Pretrain_s3_20-13-56-52/nn/Humanoid_175_Pretrain_s3_20-13-56-58.pth \
    task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.0\
    task.env.train_stage=2\
    experiment=SRL_Real_HRI_v1   max_iterations=2000   \
    train.params.config.srl_teacher_checkpoint=runs/SRL_Real_s4_25-14-45-53/nn/SRL_Real_s4.pth \
    train.params.config.dagger_loss_coef=1 train.params.config.sym_a_loss_coef=1.0  \
    task.env.pelvis_height_reward_scale=2.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    train.params.config.dagger_anneal_k=1e-5  task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=2.0 task.env.contact_force_cost_scale=0.5\
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"  
# check
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
       checkpoint=runs/SRL_Real_HRI_v1_20-19-11-50/nn/SRL_Real_HRI_v1_20-19-11-57.pth  \
       task.env.train_stage=2\
       task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"
# 第二阶段训练: S3曲线 + 取消高度跟踪，提高水平跟踪权重。增加电机功率惩罚。
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v1_20-19-11-50/nn/SRL_Real_HRI_v1_20-19-11-57.pth \
    task.env.train_stage=3 \
    task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.3\
    experiment=SRL_Real_HRI_v1   max_iterations=4000   \
    train.params.config.sym_a_loss_coef=1.0  \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=0.5\
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"  
# check  
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
       checkpoint=runs/SRL_Real_HRI_v1_24-16-49-16/nn/SRL_Real_HRI_v1_24-16-49-22.pth  \
       task.env.train_stage=3  task.env.enableDebugVis=True num_envs=1\
       task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"
# 第三阶段训练: S4全向行走 + 取消高度跟踪，提高水平跟踪权重。增加电机功率惩罚。
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v1_21-11-20-16/nn/SRL_Real_HRI_v1_21-11-20-24.pth \
    task.env.train_stage=4 \
    task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.5\
    experiment=SRL_Real_HRI_v1   max_iterations=4000   \
    train.params.config.sym_a_loss_coef=1.0  \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=5.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=1.0\
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"  
# check  
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
       checkpoint=runs/SRL_Real_HRI_vf_filtered_16-17-13-53/nn/SRL_Real_HRI_vf_filtered_16-17-13-59.pth   \
       task.env.train_stage=4  task.env.enableDebugVis=True num_envs=1\
       task.env.episodeLength=4000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"



# =============== SMP人体模型+SRL ==============
# 【未完成】
# （SMP人体模型）第一阶段训练:从s4 humanoid预训练结果开始, 训练S2直线
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    task.env.motion_file="smp_humanoid_walk_175.npy" \
    train.params.config.humanoid_checkpoint=runs/Humanoid_175_Pretrain_s3_smp_23-16-56-25/nn/Humanoid_175_Pretrain_s3_smp_23-16-56-31.pth \
    task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.0\
    task.env.train_stage=2\
    experiment=SRL_Real_HRI_v1_smp   max_iterations=2000   \
    train.params.config.srl_teacher_checkpoint=runs/SRL_Real_s4_25-14-45-53/nn/SRL_Real_s4.pth \
    train.params.config.dagger_loss_coef=1 train.params.config.sym_a_loss_coef=1.0  \
    task.env.pelvis_height_reward_scale=2.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    train.params.config.dagger_anneal_k=1e-5  task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=2.0 task.env.contact_force_cost_scale=0.5\
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"  
# check
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
       checkpoint=runs/SRL_Real_HRI_v1_smp_23-19-28-11/nn/SRL_Real_HRI_v1_smp_23-19-28-18.pth  \
       task.env.train_stage=2\
       task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"
# （SMP人体模型）第二阶段训练: S2直线 + 取消高度跟踪，提高水平跟踪权重。增加电机功率惩罚。
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    task.env.motion_file="smp_humanoid_walk_175.npy" \
    train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v1_smp_23-19-28-11/nn/SRL_Real_HRI_v1_smp_23-19-28-18.pth \
    task.env.train_stage=2  seed=$RANDOM\
    task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.3\
    experiment=SRL_Real_HRI_v1_smp   max_iterations=2000   \
    train.params.config.sym_a_loss_coef=1.0  \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=0.5\
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"  
# check   
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
       checkpoint=runs/SRL_Real_HRI_v1_smp_24-19-05-16/nn/SRL_Real_HRI_v1_smp_24-19-05-22.pth  \
       task.env.train_stage=3  task.env.enableDebugVis=True num_envs=1\
       task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v1_HXYK_175_mesh.xml"
