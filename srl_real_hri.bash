export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/Mrlgpu/lib
# Run wandb login before enabling online logging.
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

# =========== Humanoid MuscleFlat任务(2026.10) ===========
# Humanoid 直线
python SRL_Evo_train.py task=HumanoidAMPMuscleFlat test=False headless=True force_render=False task.env.cameraFollow=False num_envs=4096 task.env.train_stage=2  task.env.episodeLength=500 sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/HumanoidAMPMuscleFlat_P002_Finetune_11-17-46-29/nn/HumanoidAMPMuscleFlat_P002_Finetune_11-17-46-33.pth seed=42 experiment=HumanoidAMPMuscleFlat_P002_Finetune max_iterations=42000
# check
python SRL_Evo_train.py task=HumanoidAMPMuscleFlat test=True force_render=True task.env.cameraFollow=True num_envs=1 task.env.train_stage=2 sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/HumanoidAMPMuscleFlat_P002_Finetune_11-17-46-29/nn/HumanoidAMPMuscleFlat_P002_Finetune_11-17-46-33.pth seed=-1 task.env.episodeLength=4000


# =========== Real HRI ==========
# ================ V2 =================
# 使用V2版本SRL模型：
# 1. 真实电机参数
# 2. 基于电机实际参数的动力学约束
# 3. 基于肌肉骨骼控制器的Humanoid模型
# SRL+Humanoid 连续训练：前三阶段建立协同能力，第四阶段切换 HMI 控制。

# (5.20) obs中去除高度，线速度
# 第一阶段训练:从s4 humanoid预训练结果开始, 训练S2直线
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    train.params.config.humanoid_checkpoint=runs/HumanoidAMPMuscleFlat_P002_Finetune_17-12-34-52/nn/HumanoidAMPMuscleFlat_P002_Finetune_17-12-34-55.pth    \
    task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.0\
    task.env.train_stage=2\
    experiment=SRL_Real_HRI_v1   max_iterations=2000   \
    train.params.config.srl_teacher_checkpoint=runs/SRL_Real_Bot_v2_s4_09-13-35-18/nn/SRL_Real_Bot_v2_s4.pth \
    train.params.config.dagger_loss_coef=1 train.params.config.sym_a_loss_coef=1.0  \
    task.env.pelvis_height_reward_scale=2.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    train.params.config.dagger_anneal_k=1e-5  task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=2.0 task.env.contact_force_cost_scale=0.5\
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"  
# check
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
       checkpoint=runs/SRL_Real_HRI_v1_19-13-52-55/nn/SRL_Real_HRI_v1_19-13-53-02.pth  \
       task.env.train_stage=2\
       task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"

# 第二阶段训练: S3曲线 + 取消高度跟踪，提高水平跟踪权重。增加电机功率惩罚。
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v1_19-13-52-55/nn/SRL_Real_HRI_v1_19-13-53-02.pth \
    task.env.train_stage=3 \
    task.env.srl_motor_cost_scale=0.3\
    experiment=SRL_Real_HRI_v2_stage2   max_iterations=4000   \
    train.params.config.sym_a_loss_coef=1.0  \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=0.5\
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"  
# check  
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
       checkpoint=runs/SRL_Real_HRI_v2_stage2_19-16-50-48/nn/SRL_Real_HRI_v2_stage2_19-16-50-54_2000.pth  \
       task.env.train_stage=3  task.env.enableDebugVis=True num_envs=1\
       task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"
# 第三阶段训练: S4全向行走 + 取消高度跟踪，提高水平跟踪权重。增加电机功率惩罚。
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v2_stage2_19-16-50-48/nn/SRL_Real_HRI_v2_stage2_19-16-50-54_2000.pth \
    task.env.train_stage=4 \
    task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.5\
    experiment=SRL_Real_HRI_v2_stage3   max_iterations=4000   \
    train.params.config.sym_a_loss_coef=1.0  \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=5.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
    task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=1.0\
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"  
# check  
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
       checkpoint=runs/SRL_Real_HRI_v2_stage3_19-20-10-40/nn/SRL_Real_HRI_v2_stage3_19-20-10-46.pth    \
       task.env.train_stage=4  task.env.enableDebugVis=True num_envs=1\
       task.env.episodeLength=4000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"

# 第四阶段训练: 从第三阶段继续，S4全向行走 + HMI控制，保持网络尺寸和第三阶段奖励。
# SRL Actor命令置零，航向跟随人体骨盆；Humanoid保持小幅联合更新，Critic/奖励保留显式任务信息。
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v2_stage3_19-20-10-40/nn/SRL_Real_HRI_v2_stage3_19-20-10-46.pth \
    task.env.train_stage=4 task.env.srl_hmi_control=True task.env.HMI_obs_enable=True \
    task.env.srl_max_effort=150 task.env.srl_motor_cost_scale=0.5 \
    experiment=SRL_Real_HRI_v2_stage4_HMI max_iterations=4000 seed=42 \
    sim_device=cuda:0 rl_device=cuda:0 num_envs=8192 \
    train.params.config.humanoid_checkpoint=False train.params.config.srl_teacher_checkpoint=False \
    train.params.config.dagger_loss_coef=0.0 train.params.config.sym_a_loss_coef=1.0 \
    train.params.config.train_humanoid=True train.params.config.train_srl=True \
    train.params.config.train_humanoid_freq=10 train.params.config.learning_rate=0.0001 \
    train.params.config.central_critic=True \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0 task.env.gait_similarity_penalty_scale=5.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0 \
    task.env.tracking_ang_vel_reward_scale=2.0 \
    task.env.srl_free_actions_num=5 task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=1.0 \
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
        task.env.srl_hmi_control=True task.env.HMI_obs_enable=True \
       checkpoint=runs/SRL_Real_HRI_v2_stage4_HMI_27-12-15-43/nn/SRL_Real_HRI_v2_stage4_HMI_27-12-15-49_best.pth    \
       task.env.train_stage=4  task.env.enableDebugVis=True num_envs=1\
       task.env.episodeLength=4000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"


# 第五阶段训练: 继承stage4，启用独立HMI Encoder；Actor仍为198维。
# 64帧历史压缩为15维，替换零力矩占位；humanoid每10个minibatch更新一次，实际lr=3e-5。
# 以下仅为训练命令，需在 isaacgymenvs 目录手动执行；train_stage=4仍表示S4轨迹。
python SRL_Evo_train.py task=SRL_Real_HRI headless=True wandb_project=SRL_Evo wandb_activate=True \
    train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v2_stage4_HMI_27-12-15-43/nn/SRL_Real_HRI_v2_stage4_HMI_27-12-15-49.pth \
    task.env.train_stage=4 task.env.srl_hmi_control=True task.env.HMI_obs_enable=True \
    train.params.config.hmi_encoder.enabled=True \
    train.params.config.hmi_encoder.humanoid_lr_scale=0.3 train.params.config.hmi_encoder.learning_rate=0.0003 \
    train.params.config.hmi_encoder.microbatch_size=2048 \
    experiment=SRL_Real_HRI_v2_stage5_HMIEncoder max_iterations=4000 seed=42 \
    sim_device=cuda:0 rl_device=cuda:0 num_envs=8192 \
    train.params.config.humanoid_checkpoint=False train.params.config.srl_teacher_checkpoint=False \
    train.params.config.dagger_loss_coef=0.0 train.params.config.sym_a_loss_coef=1.0 \
    train.params.config.train_humanoid=True train.params.config.train_srl=True \
    train.params.config.train_humanoid_freq=10 train.params.config.learning_rate=0.0001 \
    train.params.config.central_critic=True \
    task.env.srl_max_effort=150 task.env.srl_motor_cost_scale=0.5 \
    task.env.pelvis_height_reward_scale=0.0 task.env.orientation_reward_scale=5.0 \
    task.env.no_fly_penalty_scale=2.0 task.env.gait_similarity_penalty_scale=5.0 \
    task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0 \
    task.env.tracking_ang_vel_reward_scale=2.0 \
    task.env.srl_free_actions_num=5 task.env.clearance_penalty_scale=10 \
    task.env.humanoid_share_reward_scale=3.0 task.env.contact_force_cost_scale=1.0 \
    task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"

# 从第五阶段中断点续训时，使用 checkpoint=<第五阶段ckpt> 和 train.params.config.hsrl_checkpoint=False。
# max_iterations 为累计迭代上限。
# check（生成第五阶段ckpt后，替换路径再执行；必须启用encoder及两个HMI开关）：
# python SRL_Evo_train.py test=True task=SRL_Real_HRI checkpoint=<第五阶段ckpt> \
#     train.params.config.hmi_encoder.enabled=True task.env.srl_hmi_control=True task.env.HMI_obs_enable=True \
#     task.env.train_stage=4 num_envs=1 task.env.srl_max_effort=150 task.env.srl_free_actions_num=5 \
#     task.env.episodeLength=4000 force_render=True task.env.cameraFollow=True \
#     task.env.asset.assetFileName="mjcf/srl_real_hri/srl_real_hri_v2_HXYK_175_mesh.xml"
