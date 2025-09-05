# -----------  model v5  -------------------------------------------------------
# srl model v5
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v5  headless=True wandb_activate=True max_iterations=1000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v5.xml" train.params.config.a_sym_loss_coef=0.0 ;  
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=TRO_SRL_bot_v5  headless=True wandb_activate=True max_iterations=1000   task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v5.xml" train.params.config.a_sym_loss_coef=1.0 ;  
# srl model v5 check
python SRL_Evo_train.py task=SRLBot test=True  num_envs=4 checkpoint=runs/SRL_bot_v5_s2_05-14-26-03/nn/SRL_bot_v5_s2.pth sim_device=cuda:0 rl_device=cuda:0  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v5.xml"

# stage 0 
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v5_s0  headless=True wandb_activate=True max_iterations=1000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v5.xml"  train.params.config.a_sym_loss_coef=1.0    ;  

# stage 0 height
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v5_s0  headless=True wandb_activate=True max_iterations=2000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v5.xml"  train.params.config.a_sym_loss_coef=0.0   task.env.pelvis_height_reward_scale=4.0 ;  

# --- stage 0 --- velocity
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v5_s0  headless=True wandb_activate=True max_iterations=1000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v5.xml"  train.params.config.a_sym_loss_coef=1.0   task.env.vel_tracking_reward_scale=8  task.env.progress_reward_scale=1.0 task.env.alive_reward_scale=1.0;  

# --- stage 1 --- velocity+height
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v5_s1  headless=True wandb_activate=True max_iterations=2000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v5.xml"  train.params.config.a_sym_loss_coef=1.0  checkpoint=runs/SRL_bot_v5_s0_05-11-19-38/nn/SRL_bot_v5_s0.pth  task.env.pelvis_height_reward_scale=8.0 task.env.vel_tracking_reward_scale=8.0  task.env.progress_reward_scale=0.0 ;  

# --- stage 2 --- vel+height+ori
python SRL_Evo_train.py task=SRLBot wandb_project=TRO_SRL_Evo experiment=SRL_bot_v5_s2  headless=True wandb_activate=True max_iterations=3000  task.env.asset.assetFileName="mjcf/srl_bot/srl_bot_inversed_v5.xml"  train.params.config.a_sym_loss_coef=1.0  checkpoint=runs/SRL_bot_v5_s1_05-13-44-04/nn/SRL_bot_v5_s1.pth  task.env.orientation_reward_scale=7 task.env.pelvis_height_reward_scale=5.0 task.env.progress_reward_scale=0.0 task.env.alive_reward_scale=0.0;  

