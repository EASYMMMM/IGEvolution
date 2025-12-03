export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/Mrlgpu/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
# 测试SRL-Gym Random Optimization
python SRLGym_train.py task=SRL_HRI experiment=SRLGym_RA_test  headless=True wandb_activate=True max_iterations=500  train.gym.design_opt=RA  train.params.config.hsrl_checkpoint=runs/SRL_HRI_21-17-34-13/nn/SRL_HRI_21-17-34-22.pth  sim_device=cuda:0 rl_device=cuda:0  task.env.design_param_obs=False  train.gym.RA_num_iterations=20