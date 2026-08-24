# SRL Real Bot v2 (8.16) training script
# --- stage 1 --- vel+motor
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s1 task.env.task_training_stage=1 \
       headless=True wandb_activate=True max_iterations=1000 task.env.vel_tracking_reward_scale=8 task.env.progress_reward_scale=1.0 \
       task.env.alive_reward_scale=1.0 task.env.srl_motor_cost_scale=0.05 task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml task.env.forceControl=False task.env.pdControl=True task.env.srl_action_filter_enable=False "task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]" "task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]" \
       train.params.config.bounds_loss_coef=0.0001 train.params.config.horizon_length=64 task.env.numEnvs=8192 seed=51
# --- check ---
####
# --- stage 2 --- vel+hei+motor
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s2  task.env.task_training_stage=2  headless=True wandb_activate=True  max_iterations=1500\
       checkpoint=runs/SRL_Real_Bot_v2_s1_16-12-20-55/nn/SRL_Real_Bot_v2_s1.pth  \
       task.env.pelvis_height_reward_scale=8.0 task.env.vel_tracking_reward_scale=8.0 task.env.srl_motor_cost_scale=0.5 task.env.progress_reward_scale=0.0 \
        task.env.forceControl=False  task.env.pdControl=True \
        task.env.srl_action_filter_enable=True \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]' \
       'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' \
       task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" 
# --- check ---
######
# --- stage 3 --- vel+hei+ori
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s3  task.env.task_training_stage=3 headless=True wandb_activate=True max_iterations=2500\
       checkpoint=runs/SRL_Real_Bot_v2_s2_16-16-32-18/nn/SRL_Real_Bot_v2_s2.pth \
       task.env.orientation_reward_scale=7 task.env.pelvis_height_reward_scale=5.0 task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]' \
       task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True\
       'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' \
       task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0  
# --- check ---
#####
# --- stage 4 --- Domain Randomization 
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s4  task.env.task_training_stage=3 task.task.randomize=True task.task.vel_pertubation=True headless=True wandb_activate=True max_iterations=3500  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
       checkpoint=runs/SRL_Real_Bot_v2_s3_17-20-20-12/nn/SRL_Real_Bot_v2_s3.pth  task.env.progress_reward_scale=0.0  task.env.srl_motor_cost_scale=0.0  task.env.alive_reward_scale=0.0  \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]'  'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]'\
       task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True

