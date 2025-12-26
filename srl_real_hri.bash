# SMPL 模型
# 训练：
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl wandb_project=SRL wandb_activate=True max_iterations=2500 sim_device=cuda:1 rl_device=cuda:1 num_envs=4096 headless=True seed=$RANDOM
# 测试：
python SRL_Evo_train.py task=HumanoidAMPSRLGym_s1_Smpl test=True force_render=True task.env.cameraFollow=True num_envs=1 sim_device=cuda:0 rl_device=cuda:0 checkpoint=runs/HumanoidAMP_SMPL_25-17-17-03/nn/HumanoidAMP_SMPL_25-17-17-08.pth seed=$RANDOM
