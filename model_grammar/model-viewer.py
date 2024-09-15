# ------- 来自于mujoco150在win+py3.9下的矫情的要求 --------
# 手动添加mujoco路径
import os
import time
from getpass import getuser

user_id = getuser()
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco200//bin")
os.add_dll_directory(f"C://Users//{user_id}//.mujoco//mujoco-py-2.0.2.0//mujoco_py")
# -------------------------------------------------------


import sys

from mujoco_py import MjSim, MjViewer, load_model_from_path
import numpy as np

model_path = 'C:\\MLY\\IGEvolution\\assets\\mjcf\\humanoid_srl\\check.xml'

# model_path = 'mjcf_model\\humanoid_srl_mode1.xml'
# model_path = 'mjcf_model\\amp_humanoid_srl_V2_1.xml'
# model_path = 'E:\\CASIA\\RE_SRL\\IGEvolution\\assets\\mjcf\\amp_humanoid_srl_V2_1.xml'
# model_path = 'E:\\CASIA\\RE_SRL\\IGEvolution\\assets\\mjcf\\SRL_seperate.xml'
# model_path = 'mjcf_model\\srl_1.xml'
# model_path = 'E:\\CASIA\\RE_SRL\\grammar_mjcf\\mjcf_model\\amp_humanoid_srl_5.xml'
#model_path = 'E:\\CASIA\\RE_SRL\\IsaacGymEnvs\\assets\\mjcf\\nv_humanoid_srl_test.xml'
print(model_path)
model = load_model_from_path(model_path)
sim = MjSim(model)
viewer = MjViewer(sim)
#ctrl = np.zeros(len(sim.data.ctrl[:]))
#print(len(ctrl))
#ctrl[3] = -0.5
#ctrl[7] = -0.5

for i in range(15000):

    #sim.data.ctrl[:] = ctrl
    #sim.step()
    viewer.render()
    
