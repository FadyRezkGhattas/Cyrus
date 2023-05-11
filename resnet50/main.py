import sys
import os.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

# Jax and Flax
from jax import random
import flax.linen as nn
import jax.numpy as jnp
from ResNet import ResNet50

# Dataset
from dataloaders import cifar10

from ResNetTrainer import ResNetTrainer

from TrainerModule import TrainState

train_loader, val_loader, test_loader = cifar10.get_data(64)

trainer = ResNetTrainer(num_classes = 10,                    
                        optimizer_hparams={
                            'optimizer': 'adam',
                            'lr': 1e-3
                        },
                        logger_params={
                            'track': True,
                            'wandb_project_name': 'test_project',
                            'wandb_entity': 'fastautomate'
                        },
                        exmp_input=next(iter(train_loader)),
                        check_val_every_n_nepoch = 5)

metrics = trainer.train_model(train_loader,
                              val_loader,
                              test_loader=test_loader,
                              num_epochs=50)