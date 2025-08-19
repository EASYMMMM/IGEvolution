# train
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=1000  sim_device=cuda:0 rl_device=cuda:0      ;  

# A-C sym train
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  checkpoint=runs/TRO_SRL_bot_v3_04-17-00-16/nn/TRO_SRL_bot_v3.pth   headless=True wandb_activate=True max_iterations=2000  sim_device=cuda:0 rl_device=cuda:0   train.params.config.a_sym_loss_coef=1.0  train.params.config.c_sym_loss_coef=0.0

# check
python SRL_Evo_train.py task=SRLBot test=True    num_envs=4 checkpoint=runs/SRL_bot_v4_s0_17-17-42-33/nn/SRL_bot_v4_s0.pth   sim_device=cuda:0 rl_device=cuda:0 

# srl model v3
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=3000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v3.xml" train.params.config.a_sym_loss_coef=1.0 ;  
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=5000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v3.xml" train.params.config.a_sym_loss_coef=1.0 ;  
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=1000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v3.xml" train.params.config.a_sym_loss_coef=0.5 ;  
# srl model v3 check
python SRL_Evo_train.py task=SRLBot test=True    num_envs=4 checkpoint=runs/TRO_SRL_bot_v3_18-20-08-44/nn/TRO_SRL_bot_v3.pth  sim_device=cuda:0 rl_device=cuda:0  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v3.xml"

# stage 0 
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v4_s0  headless=True wandb_activate=True max_iterations=1000  sim_device=cuda:0 rl_device=cuda:0  train.params.config.a_sym_loss_coef=1.0    ;  

# stage 1
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v4_s1  headless=True wandb_activate=True max_iterations=2000  sim_device=cuda:0 rl_device=cuda:0  train.params.config.a_sym_loss_coef=0.5  checkpoint=runs/SRL_bot_v4_s0_17-17-42-33/nn/SRL_bot_v4_s0.pth   ;  

# stage 2
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  checkpoint=runs/SRL_bot_v4_s0_14-18-08-35/nn/SRL_bot_v4_s0.pth  headless=True wandb_activate=True max_iterations=3000  sim_device=cuda:0 rl_device=cuda:0  train.params.config.a_sym_loss_coef=1.0 ; 