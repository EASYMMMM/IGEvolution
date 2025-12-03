export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/Mrlgpu/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
# =============================== Nice Work ======================================
# train srl hri Teacher-Student 静止站立
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Default_19-15-17-44/nn/AMP_Pretrain_Default_19-15-17-54.pth   train.params.config.srl_teacher_checkpoint=runs/SRL_teacher_standing_20-18-08-20/nn/SRL_teacher_standing.pth    train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4;   
# 11.6 train SRLHRI 1m/s  使用外肢体距离躯干更远的设计 单个free joint
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_basic_1free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v5_s2_05-16-59-52/nn/SRL_bot_v5_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=1; 
# 11.6 增加critic镜像损失 使用SRL距离躯干更远的模型训练 + 单个free joint，去掉弹簧阻尼
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_basic_1free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v5_s2_05-16-59-52/nn/SRL_bot_v5_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=1; 
# 11.18 朝向为正的SRL模型 + 1 free joint
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_1free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=1; 
# 11.19 朝向为正的SRL模型 + 2 free joint
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_2free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=2; 
# 11.19 独立教师观测 观测中添加Humanoid Euler
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_2free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=2; 
# 11.21 观测中添加Humanoid Euler + Legs Euler 
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_2free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=2; 

# ================================================================================


# humanoid AMP pretrain Random
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1 wandb_project=SRL_Evo experiment=AMP_Pretrain_Hybrid task.env.asset.assetFileName='mjcf/humanoid_srl_v3/hsrl_mode1_v3_s1.xml' headless=True wandb_activate=True max_iterations=1000  task.env.stateInit=Hybrid sim_device=cuda:0 rl_device=cuda:0      

python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1 wandb_project=SRL_Evo experiment=AMP_Pretrain_Hybrid task.env.asset.assetFileName='mjcf/humanoid_srl_v3/hsrl_mode1_v3_s1.xml' headless=True wandb_activate=True max_iterations=2000  task.env.stateInit=Hybrid  checkpoint=saved_runs/AMP_Pretrain_18-16-52-24/nn/AMP_Pretrain_18-16-52-35.pth  sim_device=cuda:0 rl_device=cuda:0      

# humanoid AMP check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1 test=True  num_envs=4   checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth  task.env.stateInit=Default  task.env.asset.assetFileName='mjcf/humanoid_srl_v3/hsrl_mode1_v3_s1.xml'

# train 教师策略：静止站立
python SRL_Evo_train.py task=SRLBot wandb_project=SRL_Evo experiment=SRL_teacher_standing  headless=True wandb_activate=True max_iterations=1000  train.params.config.a_sym_a_loss_coef=1.0    ;  
# check 教师策略：静止站立
python SRL_Evo_train.py task=SRLBot test=True  num_envs=4 checkpoint=runs/SRL_HRI_22-21-00-31/nn/SRL_HRI_22-21-00-41_250.pth  sim_device=cuda:0 rl_device=cuda:0  

# train srl hri 静止站立
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Default_19-15-17-44/nn/AMP_Pretrain_Default_19-15-17-54.pth  task.env.orientation_reward_scale=8.0  train.params.config.disc_reward_w=0.0 train.params.config.sym_a_loss_coef=0.0 ;  
# check srl hri 
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_30-20-20-57/nn/SRL_HRI_30-20-21-06_best.pth

# train srl hri 1m/s 行走
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth  train.params.config.sym_a_loss_coef=0.0 ;  
# check srl hri 
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_07-15-32-20/nn/SRL_HRI_07-15-32-29_best.pth

# train srl hri Teacher-Student 1m/s 行走
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth   train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v5_s2_05-16-59-52/nn/SRL_bot_v5_s2.pth  train.params.config.dagger_loss_coef=0.1;  
# check srl hri Teacher-Student 1m/s 行走
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_07-16-33-27/nn/SRL_HRI_07-16-33-37_best.pth


# train srl hri Teacher-Student 静止站立
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Default_19-15-17-44/nn/AMP_Pretrain_Default_19-15-17-54.pth   train.params.config.srl_teacher_checkpoint=runs/SRL_teacher_standing_20-18-08-20/nn/SRL_teacher_standing.pth    train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4;   
# check srl hri Teacher-Student 静止站立
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_31-16-34-55/nn/SRL_HRI_31-16-35-06.pth


# train srl hri Teacher-Student 变速行走
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=3000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v5_s2_05-16-59-52/nn/SRL_bot_v5_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4; 
# check srl hri Teacher-Student 变速行走
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_07-16-33-27/nn/SRL_HRI_07-16-33-37_best.pth


# srl hri test
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_24-16-57-48/nn/SRL_HRI_24-16-58-00_250.pth
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_17-17-40-01/nn/SRL_HRI_17-17-40-16.pth  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_amp_pretrain.xml
# check
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_01-16-46-39/nn/SRL_HRI_01-16-46-48.pth

# 11.18
# 朝向为正的SRL模型 + 1 free joint
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_1free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=1; 
# check
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_1free.xml  checkpoint=runs/SRL_HRI_18-20-58-33/nn/SRL_HRI_18-20-58-42.pth   task.env.srl_free_actions_num=1

# 朝向为正的SRL模型 + 2 free joint
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_2free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=2; 
# check
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_2free.xml  checkpoint=runs/SRL_HRI_19-11-28-47/nn/SRL_HRI_19-11-28-55.pth   task.env.srl_free_actions_num=2

# 11.20 SRL观测中去除相位信号 结果：暂时失败
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_2free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=3.0 train.params.config.dagger_anneal_k=1e-4  task.env.gait_similarity_penalty_scale=10.0 task.env.srl_free_actions_num=2; 
# check
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_2free.xml  checkpoint=runs/SRL_HRI_20-15-38-28/nn/SRL_HRI_20-15-38-37_best.pth    task.env.srl_free_actions_num=2

# 11.21 观测中添加Humanoid Euler + Legs Euler 
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_2free.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=2; 
# check
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_v2_2free.xml  checkpoint=runs/SRL_HRI_21-17-34-13/nn/SRL_HRI_21-17-34-22.pth   task.env.srl_free_actions_num=2

# 11.28 使用自动生成的hsrl模型 观测中添加Humanoid Euler + Legs Euler 
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000  task.env.asset.assetFileName=mjcf/hsrl_auto_gen/humanoid_with_srl.xml train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.5 train.params.config.sym_a_loss_coef=1.0 train.params.config.sym_c_loss_coef=0.5 task.env.progress_reward_scale=2.0 train.params.config.dagger_anneal_k=1e-4  task.env.srl_free_actions_num=2; 
# check
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4  task.env.asset.assetFileName=mjcf/hsrl_auto_gen/humanoid_with_srl.xml  checkpoint=runs/SRL_HRI_28-14-49-42/nn/SRL_HRI_28-14-49-51.pth     task.env.srl_free_actions_num=2


# 12.02 使用自动生成的hsrl模型，参数微调，在已经用默认模型训练好的基础上修改 
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2500  task.env.asset.assetFileName=mjcf/hsrl_auto_gen/humanoid_with_srl.xml   train.params.config.hsrl_checkpoint=runs/SRL_HRI_21-17-34-13/nn/SRL_HRI_21-17-34-22.pth    train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  train.params.config.dagger_loss_coef=0.0   task.env.progress_reward_scale=2.0   task.env.srl_free_actions_num=2; 
# check
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4  task.env.asset.assetFileName=mjcf/hsrl_auto_gen/humanoid_with_srl.xml  checkpoint=runs/SRL_HRI_02-14-41-49/nn/SRL_HRI_02-14-41-58.pth     task.env.srl_free_actions_num=2

