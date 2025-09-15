export LD_LIBRARY_PATH=/home/zdh232/anaconda3/envs/Mrlgpu/lib
export WANDB_API_KEY=95d44e5266d5325cb6a1b4dda1b8d100de903ace
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# srl hri train
python SRL_Evo_train.py task=SRL_HRI headless=True wandb_project=SRL_Evo wandb_activate=True experiment=SRL_HRI   max_iterations=3000   train.params.config.sym_loss_coef=0.0 ;  
# srl hri test
python SRL_Evo_train.py test=True task=SRL_HRI  num_envs=4   checkpoint=runs/SRL_HRI_15-18-11-56/nn/SRL_HRI_15-18-12-08.pth;  
