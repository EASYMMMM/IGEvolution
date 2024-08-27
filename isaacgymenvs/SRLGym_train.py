'''
SRLGym  外层训练框架
'''
import isaacgym
import hydra
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.learning.SRLEvo.srlgym_mp import SRLGym_process
from isaacgymenvs.learning.SRLEvo.mp_util import subproc_worker 
from datetime import datetime



@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):

    print(cfg['experiment'])
    wandb_exp_name = cfg['experiment'] + datetime.now().strftime("_%d-%H-%M-%S")
     
    subproc_cls_runner = subproc_worker(SRLGym_process, ctx="spawn", daemon=False)
    runner = subproc_cls_runner(cfg)
    try:
        _, _, frame = runner.rlgpu(wandb_exp_name).results
        print('frame=',frame)
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        runner.close()
        print('close runner')
    
    cfg['train']['params']['config']['start_frame'] = frame+1
    subproc_cls_runner = subproc_worker(SRLGym_process, ctx="spawn", daemon=False)
    runner = subproc_cls_runner(cfg)
    try:
        _, _, frame = runner.rlgpu(wandb_exp_name).results
        print('frame=',frame)
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        runner.close()
        print('close runner')

    print('----------END----------')
    


if __name__ == '__main__':
    main()

