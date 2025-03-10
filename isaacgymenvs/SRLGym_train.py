'''
2025 ICRA SRL-Gym
SRLGym  外层训练框架，进行形态-控制联合优化
参数通过cfg传入
'''
import isaacgym
import hydra
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.learning.SRLEvo.srlgym_mp import SRLGym_process
from isaacgymenvs.learning.SRLEvo.mp_util import subproc_worker 
from datetime import datetime
from isaacgymenvs.utils.reformat import omegaconf_to_dict,print_dict
from isaacgymenvs.learning.SRLEvo.srl_gym import SRLGym

@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):

    srl_gym = SRLGym(cfg)
    # srl_gym.train_test()
    srl_gym.train()
    # srl_gym.train_GA_test()
    # srl_gym.train_wandb_test()

    return



if __name__ == '__main__':
    main()

