import math
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from distutils.util import strtobool
import argparse

import numpy as np
from TrainerModule import create_data_loaders
from LinearTrainer import LinearTrainer, LinearClassifierTrainer
from linear import LinearModel
from dataloaders.scikit_to_pytorch import ScikitDatastToDataLoader
from sklearn.datasets import *
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--batch_size", type=int, default=100,
        help="the number of samples per mini-batches (can be equal to n_samples for deterministic gradient descent).")
    parser.add_argument("--n_features", type=int, default=100,
        help="the number of features.")
    parser.add_argument("--n_samples", type=int, default=100,
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

    # informative features (ill-conditioned)
    n_informative = math.ceil(args.n_features / 10)
    for batch_size in [int(args.n_samples/10), int(args.n_samples/2), args.n_samples]:
        exp_name = f"n_info_{n_informative}/{args.n_features}_batch_{batch_size}/{args.n_samples}"
        X, Y = make_regression(n_samples=args.n_samples, n_features=args.n_features, n_informative=n_informative, random_state=args.seed, noise=0.1)
        train_data_dataloader = ScikitDatastToDataLoader(X, Y, batch_size)
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

    # classification datasets
    datasets_ = [load_iris, load_digits, load_wine, load_breast_cancer,
                 fetch_olivetti_faces, fetch_lfw_people, fetch_covtype]
    names = ['toy_classification_iris', 'toy_classification_digits', 'toy_classification_wine', 'toy_classification_breast_cancer',
             'real_classification_olivetti_faces', 'real_classification_lfw_people', 'real_classification_covtype']
    for dataset, name in zip(datasets_, names):
        X, Y = dataset(return_X_y=True)
        sample_size = X.shape[0]
        for batch_size in [int(sample_size/2), int(sample_size/10), sample_size]:
            train_data_dataloader = ScikitDatastToDataLoader(X, Y, batch_size, regression=False)
            trainer = LinearClassifierTrainer(LinearModel,
                                    seed = args.seed,
                                    model_hparams={
                                        'number_out_features': np.unique(Y).shape[0]
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
                                    run_name=f"{name}_batch_{batch_size}/{sample_size}",
                                    extra_args=args)
            trainer.train_model(
                train_loader=train_data_dataloader,
                num_epochs=200
            )

    # regression datasets
    datasets_ = [load_diabetes, load_linnerud, fetch_california_housing]
    names = ['toy_regression_diabetes', 'toy_regression_linnerud', 'real_regression_california_housing']
    for dataset, name in zip(datasets_, names):
        X, Y = dataset(return_X_y=True)
        sample_size = X.shape[0]
        for batch_size in [int(sample_size/2), int(sample_size/10), sample_size]:
            train_data_dataloader = ScikitDatastToDataLoader(X, Y, X.shape[0])
            trainer = LinearTrainer(LinearModel,
                                    seed = args.seed,
                                    model_hparams={
                                        'number_out_features': 1 if dataset.__name__ != 'load_linnerud' else 3
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
                                    run_name=f"{name}_batch_{batch_size}/{sample_size}",
                                    extra_args=args)
            trainer.train_model(
                train_loader=train_data_dataloader,
                num_epochs=350
            )