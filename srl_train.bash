# train
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=1000  sim_device=cuda:0 rl_device=cuda:0 num_envs=4096    ;  

# A-C sym train
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  checkpoint=runs/TRO_SRL_bot_v3_04-17-00-16/nn/TRO_SRL_bot_v3.pth   headless=True wandb_activate=True max_iterations=2000  sim_device=cuda:0 rl_device=cuda:0   train.params.config.a_sym_loss_coef=1.0  train.params.config.c_sym_loss_coef=0.0

# check
python SRL_Evo_train.py task=SRLBot test=True    num_envs=4 checkpoint=runs/TRO_SRL_bot_v3_08-18-09-17/nn/TRO_SRL_bot_v3.pth  sim_device=cuda:0 rl_device=cuda:0 

# srl model v2
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=1000  sim_device=cuda:0 rl_device=cuda:0 num_envs=4096  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v2.xml" train.params.config.sym_loss_coef=0.1 ;  

# checkpoint training
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  checkpoint=runs/TRO_SRL_bot_v3_04-17-00-16/nn/TRO_SRL_bot_v3.pth   headless=True wandb_activate=True max_iterations=2000  sim_device=cuda:0 rl_device=cuda:0   ; 