export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/Mrlgpu/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
# 测试SRL-Gym Random Optimization
python SRLGym_train.py task=SRL_HRI experiment=SRLGym_RA_test  headless=True wandb_activate=True max_iterations=500  train.gym.design_opt=RA  train.params.config.hsrl_checkpoint=runs/SRL_HRI_21-17-34-13/nn/SRL_HRI_21-17-34-22.pth  sim_device=cuda:0 rl_device=cuda:0  task.env.design_param_obs=False  train.gym.RA_num_iterations=30
# check RA 
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4  task.env.asset.assetFileName=mjcf/srl_gym/hsrl_best_design.xml checkpoint=runs/SRLGym_RA_test_Evo_05-18-11-19/nn/best_model.pth     task.env.srl_free_actions_num=2

# 测试SRL-Gym Bayesian Optimization
python SRLGym_train.py task=SRL_HRI experiment=SRLGym_BO_test  headless=True wandb_activate=True max_iterations=500  train.gym.design_opt=BO  train.params.config.hsrl_checkpoint=runs/SRL_HRI_21-17-34-13/nn/SRL_HRI_21-17-34-22.pth  sim_device=cuda:0 rl_device=cuda:0  task.env.design_param_obs=False  train.gym.BO_num_iterations=25
# check BO 
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4  task.env.asset.assetFileName=mjcf/srl_gym/hsrl_best_design.xml checkpoint=runs/SRLGym_BO_test_Evo_08-20-32-51/nn/best_model.pth     task.env.srl_free_actions_num=2


# ========= REAL HRI SRL-GYM =============
# BO 测试
python SRLGym_train.py task=SRL_Real_HRI experiment=SRLGym_real_hsrl_BO  headless=True wandb_activate=True \
       max_iterations=400  train.gym.design_opt=BO  train.gym.BO_num_iterations=60 \
       train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v1_02-20-54-41/nn/SRL_Real_HRI_v1_02-20-54-47.pth  \
       sim_device=cuda:0 rl_device=cuda:0  task.env.design_param_obs=False \
        task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.0 \
        train.params.config.sym_a_loss_coef=1.0  \
        task.env.pelvis_height_reward_scale=2.0  \
        task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
        task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
        task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
        task.env.humanoid_share_reward_scale=2.0 task.env.contact_force_cost_scale=0.5\
        train.params.config.learning_rate=3e-5  train.params.config.horizon_length=16\
        train.params.config.minibatch_size=32768 task.env.numEnvs=4096
 


# RA 测试
python SRLGym_train.py task=SRL_Real_HRI experiment=SRLGym_real_hsrl_RA  headless=True wandb_activate=True \
       max_iterations=400  train.gym.design_opt=RA  train.gym.RA_num_iterations=60 \
       train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v1_02-20-54-41/nn/SRL_Real_HRI_v1_02-20-54-47.pth  \
       sim_device=cuda:0 rl_device=cuda:0  task.env.design_param_obs=False \
        task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.0 \
        train.params.config.sym_a_loss_coef=1.0  \
        task.env.pelvis_height_reward_scale=2.0 \
        task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
        task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
        task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
        task.env.humanoid_share_reward_scale=2.0 task.env.contact_force_cost_scale=0.5\
        train.params.config.learning_rate=3e-5  train.params.config.horizon_length=16\
        train.params.config.minibatch_size=32768 task.env.numEnvs=4096 
        
# GA 测试
python SRLGym_train.py task=SRL_Real_HRI experiment=SRLGym_real_hsrl_GA  headless=True wandb_activate=True \
       max_iterations=400  train.gym.design_opt=GA  train.gym.GA_num_iterations=6 train.gym.GA_population_size=10 \
       train.params.config.hsrl_checkpoint=runs/SRL_Real_HRI_v1_02-20-54-41/nn/SRL_Real_HRI_v1_02-20-54-47.pth  \
       sim_device=cuda:0 rl_device=cuda:0  task.env.design_param_obs=False \
        task.env.srl_max_effort=150  task.env.srl_motor_cost_scale=0.0 \
        train.params.config.sym_a_loss_coef=1.0  \
        task.env.pelvis_height_reward_scale=2.0 \
        task.env.no_fly_penalty_scale=2.0  task.env.gait_similarity_penalty_scale=2.0 \
        task.env.progress_reward_scale=0.0 task.env.vel_tracking_reward_scale=3.0\
        task.env.srl_free_actions_num=5   task.env.clearance_penalty_scale=10 \
        task.env.humanoid_share_reward_scale=2.0 task.env.contact_force_cost_scale=0.5\
        train.params.config.learning_rate=3e-5  train.params.config.horizon_length=16\
        train.params.config.minibatch_size=32768 task.env.numEnvs=4096 


# check
python SRL_Evo_train.py test=True task=SRL_Real_HRI  num_envs=4 task.env.srl_max_effort=150 \
       checkpoint=runs/SRLGym_real_hsrl_BO_Evo_25-12-29-27/nn/final_best_model.pth \
       task.env.episodeLength=2000    force_render=True task.env.cameraFollow=True  task.env.srl_free_actions_num=5  \
       task.env.asset.assetFileName="mjcf/hsrl_auto_gen/srl_real_hri_best_design.xml"
