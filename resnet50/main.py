import sys
import os.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)
from distutils.util import strtobool
import argparse

# Jax and Flax
from jax import random
import flax.linen as nn
import jax.numpy as jnp
from ResNet import ResNet50, ResNet18, ResNet34, _ResNet1

# Dataset
from dataloaders import cifar10
from ResNetTrainer import ResNetTrainer
from TrainerModule import TrainerModule
from VeloTrainerModule import VeloTrainerModule

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--batch_size", type=int, default=128,
        help="the number of samples per mini-batches")
    parser.add_argument("--wandb-project-name", type=str, default="resnet_cifar10",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default='fastautomate',
        help="the entity (team) of wandb's project")
    parser.add_argument("--validate-after-n-epochs", type=int, default=5,
        help="the number of epochs after which validation is run")
    # An epoch is 351 steps for CIFAR-10 with 45K training samples and a batch size of 128. Therefore, 150 epochs is 52,650 total gradient updates
    parser.add_argument("--epochs", type=int, default=150,
        help="the number of epochs to train for")
    parser.add_argument("--model", type=str, default='resnet1', choices=['resnet1', 'resnet18', 'resnet34', 'resnet50'],
        help="the resnet backbone to train")
    parser.add_argument("--optimizer", type=str, default='velo', choices=['adam', 'sgd', 'adamw', 'velo'])
    parser.add_argument("--weight-decay", type=float, default=0, help="The total loss will be loss + 0.5 * weight_decay * l2-param-norm")
    parser.add_argument("--add-weight-decay", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="If toggled, the weight decay will be added to the loss producing loss = loss + 0.5 * weight_decay * l2-param-norm.")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_loader, val_loader, test_loader = cifar10.get_data(args.batch_size)
    if args.model =='resnet18':
        model = ResNet18
    elif args.model == 'resnet34':
        model = ResNet34
    elif args.model == 'resnet50':
        model = ResNet50
    elif args.model == 'resnet1':
        model = _ResNet1
    else:
        raise Exception(f"Model {args.model} is not supported. Please choose from resnet1, resnet18, resnet34, or resnet50")
    
    if args.optimizer == 'velo':
        ResNetTrainer.__bases__ = (VeloTrainerModule,)
    else:
        ResNetTrainer.__bases__ = (TrainerModule,)
    
    model.__name__ = args.model
    trainer = ResNetTrainer(model,
                            seed = args.seed,
                            num_classes = 10,                    
                            optimizer_hparams={
                                'optimizer': args.optimizer,
                                'lr': 1e-3,
                                'weight_decay': args.weight_decay
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
                                num_epochs=args.epochs)