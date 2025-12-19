
# --- stage 0 --- velocity
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_test  headless=True wandb_activate=True max_iterations=1000   train.params.config.a_sym_loss_coef=0.0   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0;  
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_test  headless=True wandb_activate=True max_iterations=1000   train.params.config.a_sym_loss_coef=0.0   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0 task.env.gait_similarity_penalty_scale=0;  

# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True num_envs=4 checkpoint=runs/SRL_Real_Bot_test_19-18-05-34/nn/SRL_Real_Bot_test.pth    sim_device=cuda:1 rl_device=cuda:1  
