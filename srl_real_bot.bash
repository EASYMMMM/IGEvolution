# SRL BOT 11.18
# --- inversed v6 ---
# 将root朝向正前方，而不是root坐标系沿z轴旋转180
# --- stage 0 --- velocity
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v6_s0  headless=True wandb_activate=True max_iterations=1000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v6.xml"  train.params.config.a_sym_loss_coef=1.0   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0;  
# --- stage 1 --- velocity+height
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v6_s1  headless=True wandb_activate=True max_iterations=2000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v6.xml"  train.params.config.a_sym_loss_coef=1.0  checkpoint=runs/SRL_bot_v6_s0_18-15-26-47/nn/SRL_bot_v6_s0.pth  task.env.pelvis_height_reward_scale=8.0 task.env.vel_tracking_reward_scale=8.0  task.env.progress_reward_scale=0.0 ;  
# --- stage 2 --- vel+height+ori
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v6_s2  headless=True wandb_activate=True max_iterations=3000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v6.xml"  train.params.config.a_sym_loss_coef=1.0  checkpoint=runs/SRL_bot_v6_s1_18-17-07-52/nn/SRL_bot_v6_s1.pth  task.env.orientation_reward_scale=7 task.env.pelvis_height_reward_scale=5.0 task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0;  
# --- check ---
python SRL_Evo_train.py task=SRLBot test=True  num_envs=4 checkpoint=runs/SRL_bot_v6_s2_18-18-17-47/nn/SRL_bot_v6_s2.pth  sim_device=cuda:0 rl_device=cuda:0  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v6.xml"


# ========= SRL REAL BOT ===========
# --- stage 1 --- vel
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s1  task.env.task_training_stage=1 headless=True wandb_activate=True max_iterations=1000   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0  ;  
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=1 checkpoint=runs/SRL_Real_s1_04-15-15-24/nn/SRL_Real_s1.pth  sim_device=cuda:1 rl_device=cuda:1  
# --- stage 2 --- vel+hei
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s2  task.env.task_training_stage=2 headless=True wandb_activate=True max_iterations=1500    checkpoint=runs/SRL_Real_s1_04-15-15-24/nn/SRL_Real_s1.pth  task.env.pelvis_height_reward_scale=8.0 task.env.vel_tracking_reward_scale=8.0  task.env.progress_reward_scale=0.0 ;  
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=2 checkpoint=runs/SRL_Real_s2_04-17-54-32/nn/SRL_Real_s2.pth  sim_device=cuda:1 rl_device=cuda:1  
# --- stage 3 --- vel+hei+ori
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s3  task.env.task_training_stage=3 headless=True wandb_activate=True max_iterations=2500    checkpoint=runs/SRL_Real_s2_04-17-54-32/nn/SRL_Real_s2.pth  task.env.orientation_reward_scale=7 task.env.pelvis_height_reward_scale=5.0 task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0;  
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=3 checkpoint=runs/SRL_Real_s3_04-19-18-22/nn/SRL_Real_s3.pth sim_device=cuda:1 rl_device=cuda:1  
# --- stage 4 --- Domain Randomization 
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s4  task.env.task_training_stage=3 task.task.randomize=True task.task.vel_pertubation=True headless=True wandb_activate=True max_iterations=3500    checkpoint=runs/SRL_Real_s3_04-19-18-22/nn/SRL_Real_s3.pth  task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0;  
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=3 task.task.randomize=True  task.task.vel_pertubation=True checkpoint=runs/SRL_Real_s4_04-21-00-44/nn/SRL_Real_s4.pth   sim_device=cuda:1 rl_device=cuda:1  




# ========= SRL REAL BOT 1.20 ===========
# 放开髋旋自由度 使用峰值160的电机
# --- stage 1 --- vel
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s1  task.env.task_training_stage=1 headless=True wandb_activate=True max_iterations=1000   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0 task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_full_rotate.xml"  task.env.srl_max_effort=180;  
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=1 checkpoint=runs/SRL_Real_s1_22-15-22-12/nn/SRL_Real_s1.pth  sim_device=cuda:1 rl_device=cuda:1  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_full_rotate.xml" task.env.srl_max_effort=180
# --- stage 1.5 --- motor
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s1.5  task.env.task_training_stage=1 headless=True wandb_activate=True max_iterations=1500   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0 task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_full_rotate.xml"  checkpoint=runs/SRL_Real_s1_21-17-52-23/nn/SRL_Real_s1.pth task.env.srl_motor_cost_scale=1.0  task.env.srl_max_effort=180 ;  
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=1 checkpoint=runs/SRL_Real_s1.5_22-15-39-19/nn/SRL_Real_s1.5.pth  sim_device=cuda:1 rl_device=cuda:1  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_full_rotate.xml" task.env.srl_max_effort=180
# --- stage 2 --- vel+hei
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s2  task.env.task_training_stage=2 headless=True wandb_activate=True max_iterations=1500    checkpoint=runs/SRL_Real_s1_21-17-52-23/nn/SRL_Real_s1.pth  task.env.pelvis_height_reward_scale=8.0 task.env.vel_tracking_reward_scale=8.0  task.env.progress_reward_scale=0.0 task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_full_rotate.xml" task.env.srl_max_effort=160;  
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=2 checkpoint=runs/SRL_Real_s2_22-13-05-11/nn/SRL_Real_s2.pth   sim_device=cuda:1 rl_device=cuda:1  task.env.srl_max_effort=160;
# --- stage 3 --- vel+hei+ori
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s3  task.env.task_training_stage=3 headless=True wandb_activate=True max_iterations=2500    checkpoint=runs/SRL_Real_s2_04-17-54-32/nn/SRL_Real_s2.pth  task.env.orientation_reward_scale=7 task.env.pelvis_height_reward_scale=5.0 task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0;  
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=3 checkpoint=runs/SRL_Real_s3_04-19-18-22/nn/SRL_Real_s3.pth sim_device=cuda:1 rl_device=cuda:1  
# --- stage 4 --- Domain Randomization 
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_s4  task.env.task_training_stage=3 task.task.randomize=True task.task.vel_pertubation=True headless=True wandb_activate=True max_iterations=3500    checkpoint=runs/SRL_Real_s3_04-19-18-22/nn/SRL_Real_s3.pth  task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0;  
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=3 task.task.randomize=True  task.task.vel_pertubation=True checkpoint=runs/SRL_Real_s4_04-21-00-44/nn/SRL_Real_s4.pth   sim_device=cuda:1 rl_device=cuda:1  
