
# --- stage 0 --- velocity + standing 
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s0  headless=True wandb_activate=True max_iterations=1000   train.params.config.a_sym_loss_coef=1.0   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0;  

# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 checkpoint=runs/SRL_Real_s0_22-12-57-51/nn/SRL_Real_s0.pth   sim_device=cuda:1 rl_device=cuda:1  
