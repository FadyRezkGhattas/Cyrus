import sys
import os.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

import argparse

# Jax and Flax
from jax import random
import flax.linen as nn
import jax.numpy as jnp
from ResNet import ResNet50

# Dataset
from dataloaders import cifar10

from ResNetTrainer import ResNetTrainer

from TrainerModule import TrainState

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128,
        help="the number of samples per mini-batches")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--validate-after-n-epochs", type=int, default=5,
        help="the number of epochs after which validation is run")
    # An epoch is 351 steps for CIFAR-10 with 45K training samples and a batch size of 128. Therefore, 150 epochs is 52,650 total gradient updates
    parser.add_argument("--epochs", type=int, default=150,
        help="the number of epochs to train for")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_loader, val_loader, test_loader = cifar10.get_data(args.batch_size)

    trainer = ResNetTrainer(num_classes = 10,                    
                            optimizer_hparams={
                                'optimizer': 'adam',
                                'lr': 1e-3
                            },
                            logger_params={
                                'track': True,
                                'wandb_project_name': args.wandb_project_name,
                                'wandb_entity': args.wandb_entity
                            },
                            exmp_input=next(iter(train_loader)),
                            check_val_every_n_nepoch = args.validate_after_n_epochs,
                            extra_args=args)

    metrics = trainer.train_model(train_loader,
                                val_loader,
                                test_loader=test_loader,
                                num_epochs=50)