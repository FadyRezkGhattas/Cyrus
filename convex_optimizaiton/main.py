import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from distutils.util import strtobool
import argparse

from TrainerModule import create_data_loaders
from LinearTrainer import LinearTrainer
from linear import LinearModel
from dataloaders.scikit_to_pytorch import RegressionDataset, ScikitDatastToDataLoader
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--batch_size", type=int, default=0,
        help="the number of samples per mini-batches. If set to 0, then whole dataset is used as a batch.")
    parser.add_argument("--wandb-project-name", type=str, default="resnet_cifar10",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='fastautomate',
        help="the entity (team) of wandb's project")
    parser.add_argument("--dataset", type=str, default='synthetic', choices=['synthetic'],
        help="the entity (team) of wandb's project")
    parser.add_argument("--weight-decay", type=float, default=0, help="The total loss will be loss + weight_decay * l2-param-norm")
    parser.add_argument("--add-weight-decay", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="If toggled, the weight decay will be added to the loss producing loss = loss + weight_decay * l2-param-norm.")
    
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_args()
    train_data_dataloader = ScikitDatastToDataLoader(1000, 1000, 10, 42)
    trainer = LinearTrainer(LinearModel,
                            seed = args.seed,
                            model_hparams={
                                'number_out_features': 1
                            },                  
                            optimizer_hparams={
                                'weight_decay': args.weight_decay
                            },
                            logger_params={
                                'track': False,
                                'wandb_project_name': args.wandb_project_name,
                                'wandb_entity': args.wandb_entity
                            },
                            exmp_input=next(iter(train_data_dataloader)),
                            extra_args=args)
    trainer.train_model(
        train_loader=train_data_dataloader,
        num_epochs=500
    )