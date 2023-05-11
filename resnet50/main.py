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

from TrainerModule import TrainState

train_loader, val_loader, test_loader = cifar10.get_data(64)
exmp_input=next(iter(train_loader))

# Create Network
model = ResNet50(num_classes=10)

# Init Model States
model_rng = random.PRNGKey(1)
model_rng, init_rng = random.split(model_rng)
imgs, _ = exmp_input
init_rng, dropout_rng = random.split(init_rng)
variables = model.init({'params': init_rng}, x=imgs, train=False)
state = TrainState(step=0, 
                    apply_fn=model.apply,
                    params=variables['params'],
                    batch_stats=variables.get('batch_stats'),
                    rng=model_rng,
                    tx=None,
                    opt_state=None)

# Forward Pass
batch = next(iter(train_loader))
x, y = batch
pred = model.apply({'params': variables['params'], 'batch_stats': variables['batch_stats']}, x, mutable=['batch_stats'], train=True)
pred