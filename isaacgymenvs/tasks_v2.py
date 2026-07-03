"""Isolated task registry for v2 experiments.

This file exists so we can add new experimental tasks without editing the
original ``isaacgymenvs.tasks`` package registry.
"""

from isaacgymenvs.tasks import isaacgym_task_map as _legacy_task_map
from isaacgymenvs.tasks.SRLEvo.humanoid_amp_s1_smpl_v2 import HumanoidAMP_s1_Smpl_v2


isaacgym_task_map_v2 = dict(_legacy_task_map)
isaacgym_task_map_v2["HumanoidAMPSRLGym_s1_Smpl_v2"] = HumanoidAMP_s1_Smpl_v2
