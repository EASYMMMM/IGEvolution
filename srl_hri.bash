export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/Mrlgpu/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# humanoid AMP pretrain Random
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1 wandb_project=SRL_Evo experiment=AMP_Pretrain_Hybrid task.env.asset.assetFileName='mjcf/humanoid_srl_v3/hsrl_mode1_v3_s1.xml' headless=True wandb_activate=True max_iterations=1000  task.env.stateInit=Hybrid sim_device=cuda:0 rl_device=cuda:0      

python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1 wandb_project=SRL_Evo experiment=AMP_Pretrain_Hybrid task.env.asset.assetFileName='mjcf/humanoid_srl_v3/hsrl_mode1_v3_s1.xml' headless=True wandb_activate=True max_iterations=2000  task.env.stateInit=Hybrid  checkpoint=saved_runs/AMP_Pretrain_18-16-52-24/nn/AMP_Pretrain_18-16-52-35.pth  sim_device=cuda:0 rl_device=cuda:0      

# humanoid AMP check
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1 test=True  num_envs=4   checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth  task.env.stateInit=Default  task.env.asset.assetFileName='mjcf/humanoid_srl_v3/hsrl_mode1_v3_s1.xml'

# train 教师策略：静止站立
python SRL_Evo_train.py task=SRLBot wandb_project=SRL_Evo experiment=SRL_teacher_standing  headless=True wandb_activate=True max_iterations=1000  train.params.config.a_sym_loss_coef=1.0    ;  
# check 教师策略：静止站立
python SRL_Evo_train.py task=SRLBot test=True  num_envs=4 checkpoint=runs/SRL_HRI_22-21-00-31/nn/SRL_HRI_22-21-00-41_250.pth  sim_device=cuda:0 rl_device=cuda:0  

# train srl hri 静止站立
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Default_19-15-17-44/nn/AMP_Pretrain_Default_19-15-17-54.pth  task.env.orientation_reward_scale=8.0  train.params.config.disc_reward_w=0.0 train.params.config.sym_loss_coef=0.0 ;  
# check srl hri 
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_05-17-20-23/nn/SRL_HRI_05-17-20-33_best.pth

# train srl hri 1m/s 行走
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth  train.params.config.sym_loss_coef=0.0 ;  
# check srl hri 
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_05-20-11-45/nn/SRL_HRI_05-20-11-54_best.pth

# train srl hri Teacher-Student 1m/s 行走
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth   train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v5_s2_05-16-59-52/nn/SRL_bot_v5_s2.pth  train.params.config.dagger_loss_coef=0.1;  
# check srl hri Teacher-Student 1m/s 行走
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_06-16-53-17/nn/SRL_HRI_06-16-53-29_best.pth


# train srl hri Teacher-Student 静止站立
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Default_19-15-17-44/nn/AMP_Pretrain_Default_19-15-17-54.pth   train.params.config.srl_teacher_checkpoint=runs/SRL_teacher_standing_20-18-08-20/nn/SRL_teacher_standing.pth    train.params.config.sym_loss_coef=0.0 ;  


# srl hri test
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_24-16-57-48/nn/SRL_HRI_24-16-58-00_250.pth


python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_17-17-40-01/nn/SRL_HRI_17-17-40-16.pth  task.env.asset.assetFileName=mjcf/srl_hri/srl_hri_amp_pretrain.xml


######### 10.6 ###################
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI max_iterations=1000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth  train.params.config.sym_loss_coef=0.0 ; 
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI max_iterations=2000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth  train.params.config.sym_loss_coef=0.1 ;
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=2000   train.params.config.humanoid_checkpoint=runs/AMP_Pretrain_Hybrid_19-17-13-54/nn/AMP_Pretrain_Hybrid_19-17-14-05.pth   train.params.config.srl_teacher_checkpoint=runs/SRL_bot_v5_s2_05-16-59-52/nn/SRL_bot_v5_s2.pth  train.params.config.dagger_loss_coef=0.1;  
