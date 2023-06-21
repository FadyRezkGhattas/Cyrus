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
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--batch_size", type=int, default=10,
        help="the number of samples per mini-batches (can be equal to n_samples for deterministic gradient descent).")
    parser.add_argument("--n_features", type=int, default=10,
        help="the number of features.")
    parser.add_argument("--n_samples", type=int, default=10,
        help="the number of samples in the dataset.")
    parser.add_argument("--wandb-project-name", type=str, default="velo_convex_optimization",
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
    effective_ranks = list(range(5, args.n_features+1))
    n_informatives = list(range(5, args.n_features+1))
    
    for effective_rank in effective_ranks:
        exp_name = f"effective_rank_{effective_rank}"
        #batch_size=10, n_samples=10, n_features=10, effective_rank=None, n_informative=10, noise=25, random_state=42
        train_data_dataloader = ScikitDatastToDataLoader(batch_size=args.batch_size, n_samples=args.n_samples, n_features=args.n_features, effective_rank=effective_rank, n_informative=args.n_features, random_state=args.seed)
        trainer = LinearTrainer(LinearModel,
                                seed = args.seed,
                                model_hparams={
                                    'number_out_features': 1
                                },                  
                                optimizer_hparams={
                                    'weight_decay': args.weight_decay
                                },
                                logger_params={
                                    'track': True,
                                    'wandb_project_name': args.wandb_project_name,
                                    'wandb_entity': args.wandb_entity
                                },
                                exmp_input=next(iter(train_data_dataloader)),
                                run_name=exp_name,
                                extra_args=args)
        trainer.train_model(
            train_loader=train_data_dataloader,
            num_epochs=500
        )
        time.sleep(2)
    
    for n_informative in n_informatives:
        exp_name = f"n_informative_{n_informative}"
        train_data_dataloader = ScikitDatastToDataLoader(batch_size=args.batch_size, n_samples=args.n_samples, n_features=args.n_features, n_informative=n_informative, effective_rank=None, random_state=args.seed)
        trainer = LinearTrainer(LinearModel,
                                seed = args.seed,
                                model_hparams={
                                    'number_out_features': 1
                                },                  
                                optimizer_hparams={
                                    'weight_decay': args.weight_decay
                                },
                                logger_params={
                                    'track': True,
                                    'wandb_project_name': args.wandb_project_name,
                                    'wandb_entity': args.wandb_entity
                                },
                                exmp_input=next(iter(train_data_dataloader)),
                                run_name=exp_name,
                                extra_args=args)
        trainer.train_model(
            train_loader=train_data_dataloader,
            num_epochs=200
        )