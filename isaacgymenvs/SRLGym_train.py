'''
SRLGym  外层训练框架
'''
import isaacgym
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf import DictConfig, OmegaConf
from SRLGym_mulpro import SRLGym_process, retry
from isaacgymenvs.learning.SRLEvo.mp_util import subproc_worker 
import wandb
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from datetime import datetime

def my_init_wandb(cfg, experiment_name):
 
    wandb_unique_id = f"uid_{experiment_name}"
    print(f"Wandb using unique id {wandb_unique_id}")


    # this can fail occasionally, so we try a couple more times
    @retry(3, exceptions=(Exception,))
    def init_wandb():
        wandb.init(
            project=cfg.wandb_project,
            #entity=cfg.wandb_entity,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            sync_tensorboard=True,
            id=wandb_unique_id,
            name=experiment_name,
            resume=True,
            settings=wandb.Settings(start_method='spawn'),
        )
    
        if cfg.wandb_logcode_dir:
            wandb.run.log_code(root=cfg.wandb_logcode_dir)
            print('wandb running directory........', wandb.run.dir)

    print('Initializing WandB...')
    try:
        init_wandb()
    except Exception as exc:
        print(f'Could not initialize WandB! {exc}')

    if isinstance(cfg, dict):
        wandb.config.update(cfg, allow_val_change=True)
    else:
        wandb.config.update(omegaconf_to_dict(cfg), allow_val_change=True)


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):

    print(cfg['experiment'])
 
    experiment_name = cfg['experiment'] + datetime.now().strftime("_%d-%H-%M-%S")
    #my_init_wandb(cfg, experiment_name)
    subproc_cls_runner = subproc_worker(SRLGym_process, ctx="spawn", daemon=False)
    runner = subproc_cls_runner(cfg)
    try:
        runner.init_wandb(experiment_name)
        _, _, frame = runner.rlgpu().results
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        runner.finish_wandb().results
        runner.close()
        print('close runner')
    
    cfg['train']['params']['config']['start_frame'] = frame+1
    runner = subproc_cls_runner(cfg)
    try:
        runner.init_wandb(experiment_name)
        runner.rlgpu().results
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        runner.finish_wandb().results
        runner.close()
        print('close runner')
     
    print('----------END----------')
    
    # srlgym_process = SRLGym_process(cfg)
    # srlgym_process.rlgpu()

if __name__ == '__main__':
    main()

