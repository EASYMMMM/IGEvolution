
mujoco_ppo/runs/mujoco_v3_finetune_0720_230951/mujoco_v3_finetune_update_05000.pt 减弱域随机化下，2500回合长度，成功率96%，但是5000步长只有90%
mujoco_ppo/runs/mujoco_v3_finetune_0721_145957/mujoco_v3_finetune_update_01000.pt 在减弱的域随机化范围下， 5000回合长度，成功率能达到99%
# 略微扩大域随机化训练范围(2000轮次，训练结束，成功率只提高了5%)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD python -m mujoco_ppo.train_mujoco_v3_finetune --config mujoco_ppo/cfg/mujoco_v3_dr_finetune_expanded.yaml --checkpoint mujoco_ppo/runs/mujoco_v3_finetune_0721_145957/mujoco_v3_finetune_update_01000.pt --xml ../assets/mjcf/srl_real/srl_real_bot_v2.xml --task-training-stage 3 --domain-randomization --updates 2000 --num-envs 16 --training-episode-steps 5000 --actor-lr 1e-6 --reference-policy-coef 0.005 --actor-sym-loss-coef 0.4 --save-every 500 --device cuda --wandb --wandb-project SRL-Mujoco-V3 --wandb-run-name v3_dr_expanded_2000 --wandb-mode online
# 测试
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD python -m mujoco_ppo.train_mujoco_v3_finetune --config mujoco_ppo/cfg/mujoco_v3_dr_finetune_expanded.yaml --checkpoint mujoco_ppo/runs/mujoco_v3_finetune_0721_192727/mujoco_v3_finetune_best_eval.pt --xml ../assets/mjcf/srl_real/srl_real_bot_v2.xml --task-training-stage 3 --domain-randomization --eval-only --eval-steps 5000 --eval-num-seeds 100 --seed 1 --device cuda
mujoco_ppo/runs/mujoco_v3_finetune_0721_192727/mujoco_v3_finetune_best_eval.pt 增强域随机化下，成功率只有83%
mujoco_ppo/runs/mujoco_v3_finetune_0722_105806/mujoco_v3_finetune_update_02000.pt 继续训练2000轮次，成功率85%，最高90%

IGEvolution/isaacgymenvs/mujoco_ppo/runs/mujoco_v3_finetune_0804_234633/mujoco_v3_finetune_update_03000.pt 修改PD之后，增强域随机化下，成功率最高95%

86 88

# 直接stage3开始，训练扩大域随机化范围的策略，初始使用2500的episode以覆盖尽可能多的参数分布
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD python -m mujoco_ppo.train_mujoco_v3_finetune --config mujoco_ppo/cfg/mujoco_v3_dr_finetune_expanded.yaml --checkpoint mujoco_ppo/runs/mujoco_v3_finetune_0720_145200/mujoco_v3_finetune_best_eval.pt --xml ../assets/mjcf/srl_real/srl_real_bot_v2.xml --task-training-stage 3 --domain-randomization --updates 5000 --num-envs 16 --training-episode-steps 2500 --dr-warmup-updates 50 --dr-ramp-updates 950 --actor-lr 4e-6 --reference-policy-coef 0.001 --actor-sym-loss-coef 0.4 --save-every 500 --eval-every-updates 500 --eval-during-training-seeds 20 --seed 42 --device cuda --wandb --wandb-project SRL-Mujoco-V3 --wandb-run-name v3_stage3_expanded_dr_2500x5000 --wandb-mode online
# 使用5000steps的episode继续训练了2000update
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD python -m mujoco_ppo.train_mujoco_v3_finetune --config mujoco_ppo/cfg/mujoco_v3_dr_finetune_expanded.yaml --checkpoint mujoco_ppo/runs/mujoco_v3_finetune_0722_174431/mujoco_v3_finetune_update_05000.pt --xml ../assets/mjcf/srl_real/srl_real_bot_v2.xml --task-training-stage 3 --domain-randomization --eval-only --eval-steps 5000 --eval-num-seeds 100 --seed 1 --device cuda --no-randomize-initial-phase
90 81（5000） 86（5000）

# 换了关节限位，mujoco里5000轮次域随机化训练： IGEvolution/isaacgymenvs/mujoco_ppo/runs/mujoco_v3_finetune_0730_190743
# 训练pitch过大的问题  --pitch-wobble-penalty-scale 10.0










==========================================
stage1不滤波，从stage2开始滤波
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True headless=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=1 checkpoint=runs/SRL_Real_Bot_v2_s1_23-16-36-10/nn/SRL_Real_Bot_v2_s1.pth  sim_device=cuda:2 rl_device=cuda:2  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml"\
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]'  'task.env.srl_effort_limits=[120, 120, 400, 120, 120, 400]' task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=False
# --- stage 2 --- vel+hei+motor
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s2  task.env.task_training_stage=2  headless=True wandb_activate=True  max_iterations=2000    checkpoint=runs/SRL_Real_Bot_v2_s1_23-16-36-10/nn/SRL_Real_Bot_v2_s1.pth  \
       task.env.pelvis_height_reward_scale=8.0 task.env.vel_tracking_reward_scale=8.0 task.env.srl_motor_cost_scale=0.5 task.env.progress_reward_scale=0.0 \
        task.env.forceControl=False  task.env.pdControl=True sim_device=cuda:3 rl_device=cuda:3 \
        task.env.srl_action_filter_enable=True \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]' \
       'task.env.srl_effort_limits=[120, 120, 400, 120, 120, 400]' \
       task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" 
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=2 checkpoint=runs/SRL_Real_Bot_v2_s2_23-23-08-20/nn/SRL_Real_Bot_v2_s2.pth   sim_device=cuda:3 rl_device=cuda:3 task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]'   'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True
# --- stage 3 --- vel+hei+orisd
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s3  task.env.task_training_stage=3 headless=True wandb_activate=True max_iterations=3000    checkpoint=runs/SRL_Real_Bot_v2_s2_23-23-08-20/nn/SRL_Real_Bot_v2_s2.pth \
       task.env.orientation_reward_scale=7 task.env.pelvis_height_reward_scale=5.0 task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]' sim_device=cuda:3 rl_device=cuda:3 \
       task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True\
       'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' \
       task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0 
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=3 task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml"  checkpoint=runs/SRL_Real_Bot_v2_s3_24-09-11-31/nn/SRL_Real_Bot_v2_s3.pth  \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]'  'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True
# --- stage 4 --- Domain Randomization 
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s4  task.env.task_training_stage=3 task.task.randomize=True task.task.vel_pertubation=True headless=True wandb_activate=True max_iterations=3500  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
       checkpoint=runs/SRL_Real_Bot_v2_s3_24-09-11-31/nn/SRL_Real_Bot_v2_s3.pth  task.env.progress_reward_scale=0.0  task.env.srl_motor_cost_scale=0.0  task.env.alive_reward_scale=0.0  \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]'  'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' sim_device=cuda:3 rl_device=cuda:3 \
       task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=3 task.task.randomize=True  task.task.vel_pertubation=True checkpoint=runs/SRL_Real_Bot_v2_s4_24-15-30-46/nn/SRL_Real_Bot_v2_s4.pth\
       task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml"  'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]'  'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True
# --- stage 5 --- Domain Randomization forced 
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s5  task.env.task_training_stage=3 task.task.randomize=True task.task.vel_pertubation=True headless=True wandb_activate=True max_iterations=4500  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
       checkpoint=runs/SRL_Real_Bot_v2_s4_24-15-30-46/nn/SRL_Real_Bot_v2_s4.pth  task.env.progress_reward_scale=0.0  task.env.srl_motor_cost_scale=0.0  task.env.alive_reward_scale=0.0  \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]'  'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' sim_device=cuda:3 rl_device=cuda:3 \
       task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True


===========================
换了xml的关节限位
# stage1
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s1  task.env.task_training_stage=1 headless=True wandb_activate=True max_iterations=1000   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 \
        task.env.alive_reward_scale=1.0 task.env.srl_motor_cost_scale=0.05   task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2_true.xml"  sim_device=cuda:3 rl_device=cuda:3 \
        task.env.forceControl=False  task.env.pdControl=True \
        task.env.srl_action_filter_enable=False seed=45 \
        'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]' \
        'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]'
# check

# stage2 加上滤波
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s2  task.env.task_training_stage=2  headless=True wandb_activate=True  max_iterations=2000    checkpoint=runs/SRL_Real_Bot_v2_s1_29-16-31-03/nn/SRL_Real_Bot_v2_s1.pth  \
       task.env.pelvis_height_reward_scale=8.0 task.env.vel_tracking_reward_scale=8.0 task.env.srl_motor_cost_scale=0.5 task.env.progress_reward_scale=0.0 \
        task.env.forceControl=False  task.env.pdControl=True \
        task.env.srl_action_filter_enable=True sim_device=cuda:3 rl_device=cuda:3 \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]' \
       'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' \
       task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2_true.xml" 
# check

# stage 3
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s3  task.env.task_training_stage=3 headless=True wandb_activate=True max_iterations=3000    checkpoint=runs/SRL_Real_Bot_v2_s2_29-19-55-34/nn/SRL_Real_Bot_v2_s2.pth \
       task.env.orientation_reward_scale=7 task.env.pelvis_height_reward_scale=5.0 task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2_true.xml" \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]' \
       task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True\
       'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' sim_device=cuda:3 rl_device=cuda:3 \
       task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0  


# stage 4
python SRL_Evo_train.py task=SRL_Real_Bot wandb_project=SRL_Real experiment=SRL_Real_Bot_v2_s4  task.env.task_training_stage=3 task.task.randomize=True task.task.vel_pertubation=True headless=True wandb_activate=True max_iterations=4000  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2_true.xml" \
       checkpoint=runs/SRL_Real_Bot_v2_s3_29-23-31-34/nn/SRL_Real_Bot_v2_s3.pth  task.env.progress_reward_scale=0.0  task.env.srl_motor_cost_scale=0.0  task.env.alive_reward_scale=0.0  \
       'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]'  'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' sim_device=cuda:3 rl_device=cuda:3 \
       task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True
# --- check ---
python SRL_Evo_train.py task=SRL_Real_Bot test=True headless=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=3 task.task.randomize=True  task.task.vel_pertubation=True checkpoint=runs/SRL_Real_Bot_v2_s4_30-08-49-37/nn/SRL_Real_Bot_v2_s4.pth\
       task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2_true.xml"  'task.env.default_joint_angles=[0 , -0.55,  -0.3, 0 , -0.55,  -0.3]'  'task.env.srl_effort_limits=[90, 90, 350, 90, 90, 350]' task.env.forceControl=False  task.env.pdControl=True  task.env.srl_action_filter_enable=True





状态估计器和actor一起训练
# stage1
python SRL_Evo_train.py \
  task=SRL_Real_Bot_Concurrent \
  train=SRL_Real_Bot_ConcurrentPPO \
  wandb_project=SRL_Real \
  experiment=SRL_Real_Bot_v2_concurrent_s1_seed46 \
  wandb_activate=True \
  headless=True \
  seed=45 \
  num_envs=16384 \
  max_iterations=1000 \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=1 \
  task.task.randomize=False \
  task.task.vel_pertubation=False \
  task.env.vel_tracking_reward_scale=8.0 \
  task.env.progress_reward_scale=1.0 \
  task.env.alive_reward_scale=1.0 \
  task.env.srl_motor_cost_scale=0.05 \
  task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=False \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  train.params.config.critic_coef=0 \
  train.params.config.save_frequency=100 \
  train.params.config.concurrent_estimator_history_len=10 \
  'train.params.config.concurrent_bootstrap_levels=[0.0,0.1,0.25,0.5,0.75,1.0]' \
  train.params.config.concurrent_bootstrap_initial_level=0 \
  train.params.config.concurrent_bootstrap_auto_advance=True \
  train.params.config.concurrent_bootstrap_warmup_epochs=300 \
  train.params.config.concurrent_bootstrap_min_epochs_per_level=100
# check

#stage 2 高度随机化+滤波
python SRL_Evo_train.py \
  task=SRL_Real_Bot_Concurrent \
  train=SRL_Real_Bot_ConcurrentPPO \
  wandb_project=SRL_Real \
  experiment=SRL_Real_Bot_v2_concurrent_s2_seed45 \
  wandb_activate=True \
  headless=True \
  seed=45 \
  num_envs=16384 \
  max_iterations=1500 \
  checkpoint=runs/SRL_Real_Bot_v2_concurrent_s1_seed45_07-17-22-56/nn/SRL_Real_Bot_v2_concurrent_s1_seed45.pth \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=2 \
  task.task.randomize=False \
  task.task.vel_pertubation=False \
  task.env.pelvis_height_reward_scale=8.0 \
  task.env.vel_tracking_reward_scale=8.0 \
  task.env.srl_motor_cost_scale=0.5 \
  task.env.progress_reward_scale=0.0 \
  task.env.alive_reward_scale=1.0 \
  task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  train.params.config.critic_coef=0 \
  train.params.config.save_frequency=100 \
  train.params.config.concurrent_estimator_history_len=10 \
  'train.params.config.concurrent_bootstrap_levels=[0.0,0.1,0.25,0.5,0.75,1.0]' \
  train.params.config.concurrent_bootstrap_auto_advance=True \
  train.params.config.concurrent_bootstrap_min_epochs_per_level=100 \
  train.params.config.concurrent_bootstrap_reset_on_restore=True \
  train.params.config.concurrent_bootstrap_reset_probability=0.5 \
  train.params.config.concurrent_bootstrap_clear_metrics_on_reset=True

# check
python SRL_Evo_train.py \
  task=SRL_Real_Bot_Concurrent \
  train=SRL_Real_Bot_ConcurrentPPO \
  test=True \
  headless=True \
  num_envs=4 \
  seed=45 \
  checkpoint=runs/SRL_Real_Bot_nobattery_concurrent_s2_04-00-03-54/nn/SRL_Real_Bot_nobattery_concurrent_s2.pth \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=2 \
  task.task.randomize=False \
  task.task.vel_pertubation=False \
  task.env.pelvis_height_reward_scale=8.0 \
  task.env.vel_tracking_reward_scale=8.0 \
  task.env.srl_motor_cost_scale=0.5 \
  task.env.progress_reward_scale=0.0 \
  task.env.alive_reward_scale=1.0 \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_nobattery_isaac.xml" \
  train.params.config.concurrent_eval_use_estimator=True


# stage 3 加大滤波难度，截止频率降为4Hz
python SRL_Evo_train.py \
  task=SRL_Real_Bot_Concurrent \
  train=SRL_Real_Bot_ConcurrentPPO \
  wandb_project=SRL_Real \
  experiment=SRL_Real_Bot_v2_concurrent_s3_seed45 \
  wandb_activate=True \
  headless=True \
  seed=45 \
  num_envs=16384 \
  max_iterations=3000 \
  checkpoint=runs/SRL_Real_Bot_v2_concurrent_s2_seed45_07-20-16-51/nn/SRL_Real_Bot_v2_concurrent_s2_seed45.pth \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=3 \
  task.task.randomize=False \
  task.task.vel_pertubation=False \
  task.env.orientation_reward_scale=7.0 \
  task.env.pelvis_height_reward_scale=5.0 \
  task.env.progress_reward_scale=0.0 \
  task.env.alive_reward_scale=0.0 \
  task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  train.params.config.critic_coef=0 \
  train.params.config.save_frequency=100 \
  train.params.config.concurrent_estimator_history_len=10 \
  'train.params.config.concurrent_bootstrap_levels=[0.0,0.1,0.25,0.5,0.75,1.0]' \
  train.params.config.concurrent_bootstrap_auto_advance=True \
  train.params.config.concurrent_bootstrap_min_epochs_per_level=100 \
  train.params.config.concurrent_bootstrap_reset_on_restore=True \
  train.params.config.concurrent_bootstrap_reset_probability=0.5 \
  train.params.config.concurrent_bootstrap_clear_metrics_on_reset=True

# check
python SRL_Evo_train.py \
  task=SRL_Real_Bot_Concurrent \
  train=SRL_Real_Bot_ConcurrentPPO \
  test=True \
  headless=True \
  wandb_activate=False \
  seed=45 \
  num_envs=1 \
  checkpoint=runs/SRL_Real_Bot_v2_concurrent_s3_seed45_08-00-08-14/nn/SRL_Real_Bot_v2_concurrent_s3_seed45.pth \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=3 \
  task.env.episodeLength=5000 \
  task.task.randomize=False \
  task.task.vel_pertubation=False \
  task.env.orientation_reward_scale=7.0 \
  task.env.pelvis_height_reward_scale=5.0 \
  task.env.progress_reward_scale=0.0 \
  task.env.alive_reward_scale=0.0 \
  task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  train.params.config.concurrent_estimator_history_len=10 \
  train.params.config.concurrent_eval_use_estimator=True \
  ++train.params.config.player.deterministic=True \
  ++train.params.config.player.games_num=20

# stage 4
python SRL_Evo_train.py \
  task=SRL_Real_Bot_Concurrent \
  train=SRL_Real_Bot_ConcurrentPPO \
  wandb_project=SRL_Real \
  experiment=SRL_Real_Bot_v2_concurrent_s4_seed45 \
  wandb_activate=True \
  headless=True \
  seed=45 \
  num_envs=16384 \
  max_iterations=3500 \
  checkpoint=runs/SRL_Real_Bot_v2_concurrent_s3_seed45_08-00-08-14/nn/SRL_Real_Bot_v2_concurrent_s3_seed45.pth \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=3 \
  task.task.randomize=True \
  task.task.vel_pertubation=True \
  task.env.progress_reward_scale=0.0 \
  task.env.srl_motor_cost_scale=0.0 \
  task.env.alive_reward_scale=0.0 \
  task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  train.params.config.critic_coef=0 \
  train.params.config.save_frequency=100 \
  train.params.config.concurrent_estimator_history_len=10 \
  'train.params.config.concurrent_bootstrap_levels=[0.0,0.1,0.25,0.5,0.75,1.0]' \
  train.params.config.concurrent_bootstrap_auto_advance=True \
  train.params.config.concurrent_bootstrap_min_epochs_per_level=100 \
  train.params.config.concurrent_bootstrap_reset_on_restore=True \
  train.params.config.concurrent_bootstrap_reset_probability=0.5 \
  train.params.config.concurrent_bootstrap_clear_metrics_on_reset=True

# check
python SRL_Evo_train.py \
  task=SRL_Real_Bot_Concurrent \
  train=SRL_Real_Bot_ConcurrentPPO \
  test=True \
  headless=True \
  wandb_activate=False \
  seed=45 \
  num_envs=1 \
  checkpoint=runs/SRL_Real_Bot_v2_concurrent_s4_seed45_08-10-39-22/nn/SRL_Real_Bot_v2_concurrent_s4_seed45.pth \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=3 \
  task.env.episodeLength=5000 \
  task.task.randomize=True \
  task.task.vel_pertubation=True \
  task.env.progress_reward_scale=0.0 \
  task.env.srl_motor_cost_scale=0.0 \
  task.env.alive_reward_scale=0.0 \
  task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  train.params.config.concurrent_estimator_history_len=10 \
  train.params.config.concurrent_eval_use_estimator=True \
  ++train.params.config.player.deterministic=True \
  ++train.params.config.player.games_num=20



python collect_concurrent_estimator_diagnostics.py \
  task=SRL_Real_Bot_Concurrent \
  train=SRL_Real_Bot_ConcurrentPPO \
  checkpoint=runs/SRL_Real_Bot_v2_concurrent_s4_seed45_08-10-39-22/nn/last_SRL_Real_Bot_v2_concurrent_s4_seed45_ep_3400_rew_47289.953.pth \
  headless=True \
  seed=45 \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  task.env.task_training_stage=3 \
  task.task.randomize=True \
  task.task.vel_pertubation=True \
  task.env.asset.assetFileName=mjcf/srl_real/srl_real_bot_v2.xml \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  train.params.config.concurrent_estimator_history_len=10 \
  +diagnostics.num_envs=64 \
  +diagnostics.collect_steps=5000 \
  +diagnostics.startup_steps=300 \
  +diagnostics.fixed_command=True \
  +diagnostics.target_vx=1.0 \
  +diagnostics.target_wz=0.0 \
  +diagnostics.target_height=1.0 \
  +diagnostics.full_strength_dr=True \
  +diagnostics.output=diagnostic_data/isaacgym_full_dr_gt_shadow.npz



















===============================
# 数据采集
python SRL_Evo_train.py \
  task=SRL_Real_Bot \
  test=True \
  headless=True \
  num_envs=512 \
  checkpoint=runs/SRL_Real_Bot_v2_s4_09-08-56-04/nn/SRL_Real_Bot_v2_s4.pth \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=3 \
  task.task.randomize=True \
  task.task.vel_pertubation=True \
  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
  'task.env.srl_policy_obs_remove_ids=[]' \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  train.params.config.collect_privileged_estimator_data=True \
  train.params.config.estimator_data_dir=estimator_data \
  train.params.config.estimator_history_len=64 \
  train.params.config.estimator_collect_steps=5000 \
  train.params.config.estimator_chunk_steps=512 \
  train.params.config.estimator_collect_deterministic=True

# 训练显示估计器
python train_srl_privileged_estimator.py \
  --data-dir estimator_data/srl_privileged_20260806_121936 \
  --output estimator_checkpoints/srl_privileged_estimator_10hz.pt \
  --device cuda:2 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --weight-decay 1e-5 \
  --validation-fraction 0.1 \
  --seed 42

# 评估估计器
python eval_srl_privileged_estimator.py \
  --data-dir estimator_test_data/srl_privileged_20260806_133008 \
  --checkpoint estimator_checkpoints/srl_privileged_estimator_10hz.pt \
  --device cuda:3 \
  --output-dir estimator_eval/independent_seed_20260807 \
  --plot-env 0

# 使用估计器估计的特权信息替代真实信息
A 组
python SRL_Evo_train.py \
  task=SRL_Real_Bot \
  test=True \
  headless=True \
  seed=45 \
  num_envs=128 \
  checkpoint=runs/SRL_Real_Bot_v2_s4_09-08-56-04/nn/SRL_Real_Bot_v2_s4.pth \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=3 \
  task.task.randomize=True \
  task.task.vel_pertubation=True \
  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
  'task.env.srl_policy_obs_remove_ids=[]' \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  train.params.config.collect_privileged_estimator_data=False \
  train.params.config.evaluate_privileged_estimator_closed_loop=True \
  train.params.config.estimator_closed_loop_use_ground_truth=True \
  train.params.config.estimator_closed_loop_steps=6000 \
  train.params.config.estimator_closed_loop_deterministic=True \
  train.params.config.estimator_closed_loop_output_dir=estimator_ab/A_ground_truth

B 组
python SRL_Evo_train.py \
  task=SRL_Real_Bot \
  test=True \
  headless=True \
  seed=45 \
  num_envs=128 \
  checkpoint=runs/SRL_Real_Bot_v2_s4_09-08-56-04/nn/SRL_Real_Bot_v2_s4.pth \
  sim_device=cuda:3 \
  rl_device=cuda:3 \
  task.env.task_training_stage=3 \
  task.task.randomize=True \
  task.task.vel_pertubation=True \
  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
  'task.env.srl_policy_obs_remove_ids=[]' \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  train.params.config.collect_privileged_estimator_data=False \
  train.params.config.evaluate_privileged_estimator_closed_loop=True \
  train.params.config.estimator_closed_loop_use_ground_truth=False \
  train.params.config.privileged_estimator_checkpoint=estimator_checkpoints/srl_privileged_estimator_10hz.pt \
  train.params.config.estimator_closed_loop_steps=6000 \
  train.params.config.estimator_closed_loop_deterministic=True \
  train.params.config.estimator_closed_loop_output_dir=estimator_ab/B_estimated

# DAgger数据采集（第一轮）
python SRL_Evo_train.py \
  task=SRL_Real_Bot \
  test=True \
  headless=True \
  seed=146 \
  num_envs=512 \
  checkpoint=runs/SRL_Real_Bot_v2_s4_09-08-56-04/nn/SRL_Real_Bot_v2_s4.pth \
  sim_device=cuda:2 \
  rl_device=cuda:2 \
  task.env.task_training_stage=3 \
  task.task.randomize=True \
  task.task.vel_pertubation=True \
  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
  'task.env.srl_policy_obs_remove_ids=[]' \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  train.params.config.collect_privileged_estimator_data=True \
  train.params.config.evaluate_privileged_estimator_closed_loop=False \
  train.params.config.estimator_data_dir=estimator_data/dagger_round1 \
  train.params.config.estimator_history_len=64 \
  train.params.config.estimator_collect_steps=5000 \
  train.params.config.estimator_chunk_steps=512 \
  train.params.config.estimator_collect_deterministic=True \
  train.params.config.estimator_collect_alpha=0.75 \
  train.params.config.estimator_collect_checkpoint=estimator_checkpoints/srl_privileged_estimator_10hz.pt

# DAgger第一轮训练
python train_srl_privileged_estimator.py \
  --data-dir \
    estimator_data/srl_privileged_20260806_121936 \
    estimator_data/dagger_round1/srl_privileged_alpha_0p75_20260806_170743 \
  --output estimator_checkpoints/srl_privileged_estimator_dagger_r1.pt \
  --device cuda:2 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --weight-decay 1e-5 \
  --validation-fraction 0.1 \
  --seed 42

# 评估DAgger第一轮训练的结果
python eval_srl_privileged_estimator.py \
  --data-dir estimator_test_data/srl_privileged_20260806_133008 \
  --checkpoint estimator_checkpoints/srl_privileged_estimator_dagger_r1.pt \
  --device cuda:2 \
  --output-dir estimator_eval/dagger_r1

# 第一轮结果并不好，开始第二轮数据采集
python SRL_Evo_train.py \
  task=SRL_Real_Bot \
  test=True \
  headless=True \
  seed=247 \
  num_envs=512 \
  checkpoint=runs/SRL_Real_Bot_v2_s4_09-08-56-04/nn/SRL_Real_Bot_v2_s4.pth \
  sim_device=cuda:2 \
  rl_device=cuda:2 \
  task.env.task_training_stage=3 \
  task.task.randomize=True \
  task.task.vel_pertubation=True \
  task.env.asset.assetFileName="mjcf/srl_real/srl_real_bot_v2.xml" \
  'task.env.srl_policy_obs_remove_ids=[]' \
  'task.env.default_joint_angles=[0,-0.55,-0.3,0,-0.55,-0.3]' \
  'task.env.srl_effort_limits=[90,90,350,90,90,350]' \
  task.env.forceControl=False \
  task.env.pdControl=True \
  task.env.srl_action_filter_enable=True \
  task.env.srl_action_filter_cutoff_hz=10.0 \
  task.env.srl_action_filter_order=2 \
  train.params.config.collect_privileged_estimator_data=True \
  train.params.config.evaluate_privileged_estimator_closed_loop=False \
  train.params.config.estimator_data_dir=estimator_data/dagger_round2 \
  train.params.config.estimator_history_len=64 \
  train.params.config.estimator_collect_steps=5000 \
  train.params.config.estimator_chunk_steps=512 \
  train.params.config.estimator_collect_deterministic=True \
  train.params.config.estimator_collect_alpha=1.0 \
  train.params.config.estimator_collect_checkpoint=estimator_checkpoints/srl_privileged_estimator_dagger_r1.pt

# 训练r2估计器
python train_srl_privileged_estimator.py \
  --data-dir \
    estimator_data/srl_privileged_20260806_121936 \
    estimator_data/dagger_round1/srl_privileged_alpha_0p75_20260806_170743 \
    estimator_data/dagger_round2/srl_privileged_alpha_1p0_20260811_150611 \
  --output estimator_checkpoints/srl_privileged_estimator_dagger_r2.pt \
  --device cuda:2 \
  --epochs 20 \
  --learning-rate 3e-4 \
  --weight-decay 1e-5 \
  --validation-fraction 0.1 \
  --seed 42