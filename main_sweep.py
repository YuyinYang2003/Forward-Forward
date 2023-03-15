import wandb
import yaml


if __name__ == "__main__":

    with open('/home/intern/scratch/yuyinyang/Forward-Forward-wandb_and_hydra/sweep_mnist.yaml') as file:
        sweep_cfg = yaml.load(file, Loader=yaml.FullLoader)
        print(sweep_cfg)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project='Forward-Forward')
    wandb.agent(sweep_id)