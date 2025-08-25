# train
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=1000  sim_device=cuda:0 rl_device=cuda:0      ;  

# A-C sym train
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  checkpoint=runs/TRO_SRL_bot_v3_04-17-00-16/nn/TRO_SRL_bot_v3.pth   headless=True wandb_activate=True max_iterations=2000  sim_device=cuda:0 rl_device=cuda:0   train.params.config.a_sym_loss_coef=1.0  train.params.config.c_sym_loss_coef=0.0

# check
python SRL_Evo_train.py task=SRLBot test=True    num_envs=4 checkpoint=runs/SRL_bot_v4_s0_19-16-32-16/nn/SRL_bot_v4_s0.pth   sim_device=cuda:0 rl_device=cuda:0 

# srl model v3
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=3000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v3.xml" train.params.config.a_sym_loss_coef=1.0 ;  
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=5000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v3.xml" train.params.config.a_sym_loss_coef=1.0 ;  
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v3  headless=True wandb_activate=True max_iterations=1000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v3.xml" train.params.config.a_sym_loss_coef=0.5 ;  
# srl model v3 check
python SRL_Evo_train.py task=SRLBot test=True    num_envs=4 checkpoint=runs/TRO_SRL_bot_v3_19-15-34-26/nn/TRO_SRL_bot_v3.pth sim_device=cuda:0 rl_device=cuda:0  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v3.xml"

# ------------------------------------------------------------------------------


# -----------  model v4  -------------------------------------------------------
# srl model v4
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v4  headless=True wandb_activate=True max_iterations=1000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml" train.params.config.a_sym_loss_coef=0.0 ;  
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v4  headless=True wandb_activate=True max_iterations=1000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml" train.params.config.a_sym_loss_coef=1.0 ;  
# srl model v4 check
python SRL_Evo_train.py task=SRLBot test=True    num_envs=4 checkpoint=runs/SRL_bot_v4_s0_24-17-02-50/nn/SRL_bot_v4_s0.pth  sim_device=cuda:0 rl_device=cuda:0  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml"

# stage 0 
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v4_s0  headless=True wandb_activate=True max_iterations=1000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml"  train.params.config.a_sym_loss_coef=1.0    ;  

# stage 0 velocity
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v4_s0  headless=True wandb_activate=True max_iterations=1000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml"  train.params.config.a_sym_loss_coef=1.0   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0;  

# stage 0 orientation
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v4_s0  headless=True wandb_activate=True max_iterations=1000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml"  train.params.config.a_sym_loss_coef=1.0    ;  

# stage 1 velocity
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v4_s1  headless=True wandb_activate=True max_iterations=2000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml"  train.params.config.a_sym_loss_coef=1.0  checkpoint=runs/SRL_bot_v4_s0_22-17-54-52/nn/SRL_bot_v4_s0.pth  task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0;  

# stage 1 vel+ori
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v4_s1  headless=True wandb_activate=True max_iterations=2000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml"  train.params.config.a_sym_loss_coef=1.0  checkpoint=runs/SRL_bot_v4_s0_23-15-30-43/nn/SRL_bot_v4_s0.pth  task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0;  


# stage 2 orientation
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v4_s1  headless=True wandb_activate=True max_iterations=2000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml"  train.params.config.a_sym_loss_coef=1.0  checkpoint=runs/SRL_bot_v4_s0_22-17-54-52/nn/SRL_bot_v4_s0.pth  task.env.orientation_reward_scale=5  task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=1.0;  

# stage 2
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v4_s2  headless=True wandb_activate=True max_iterations=3000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v4.xml"  train.params.config.a_sym_loss_coef=0.0  checkpoint=runs/SRL_bot_v4_s1_22-18-50-44/nn/SRL_bot_v4_s1.pth  task.env.orientation_reward_scale=5  task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0;  
