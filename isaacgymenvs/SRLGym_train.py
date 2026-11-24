'''
2025 ICRA SRL-Gym
SRLGym  外层训练框架，进行形态-控制联合优化
参数通过cfg传入
'''
import isaacgym
import hydra
from omegaconf import DictConfig
from isaacgymenvs.learning.SRLEvo.srl_gym import SRLGym

@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):

    SRLGym(cfg).train()


if __name__ == '__main__':
    main()
