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

    # ill-conditioned problem
    eigenvals = [4,6,8,10]
    for eigenval in eigenvals:
        # Generate an ill-conditioned positive definite matrix A
        D = np.diag([eigenval**(i-1) for i in range(1, args.n_features+1)])
        Q, _ = np.linalg.qr(np.random.randn(args.n_features, args.n_features))
        A = Q.T @ D @ Q
        # Generate input data and corresponding output values
        X, y = make_regression(n_samples=100, n_features=args.n_features, random_state=42, noise=10)
        # Compute the output values using the ill-conditioned matrix A
        y_true = X @ A
        # Add noise to the computed output values
        noise = np.random.normal(loc=0, scale=10, size=y_true.shape)
        y_noisy = y_true + noise
        # normalize in/out features
        X = StandardScaler().fit_transform(X)
        y_noisy = StandardScaler().fit_transform(y_noisy)

        train_data_dataloader = ScikitDatastToDataLoader(X, y_noisy, X.shape[0])
        trainer = LinearTrainer(LinearModel,
                                    seed = args.seed,
                                    model_hparams={
                                        'number_out_features': 10
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
                                    run_name=f"ill_conditioned_normalized_{eigenval}",
                                    extra_args=args)
        trainer.train_model(
            train_loader=train_data_dataloader,
            num_epochs=200
        )
    '''
    # informative features ablation
    n_informatives = list(range(5, args.n_features+1))
    for n_informative in n_informatives:
        exp_name = f"n_informative_{n_informative}"
        X, Y = make_regression(n_samples=args.n_samples, n_features=args.n_features, n_informative=n_informative, effective_rank=None, random_state=args.seed)
        train_data_dataloader = ScikitDatastToDataLoader(X, Y, args.batch_size)
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
        train_data_dataloader = ScikitDatastToDataLoader(X, Y, X.shape[0], regression=False)
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
                                run_name=name,
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
                                run_name=name,
                                extra_args=args)
        trainer.train_model(
            train_loader=train_data_dataloader,
            num_epochs=350
        )'''