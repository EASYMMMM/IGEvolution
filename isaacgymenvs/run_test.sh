#!/bin/bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
python SRL_Evo_train.py task=SRL_Real_Bot test=True force_render=True task.env.cameraFollow=True num_envs=4 task.env.task_training_stage=3 task.task.randomize=True  task.task.vel_pertubation=True checkpoint=runs/SRL_Real_s4-3_nooffset_25-21-57-44/nn/SRL_Real_s4-3_nooffset.pth   sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0
