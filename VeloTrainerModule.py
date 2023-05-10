import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from TrainerModule import TrainerModule, TrainState
from typing import Any
from flax.training import train_state
from learned_optimization.research.general_lopt import prefab
import optax
from copy import copy
from jaxopt import tree_util
import jax.numpy as jnp

class VeloState(TrainState):
    def __init__(self,
                 args,
                 **kwargs):
        super().__init__(args, **kwargs)
        self.optimizer_name = 'velo'
        
    def apply_gradients(self, *, grads, **kwargs):
        # Change update signature to pass loss as expected by VeLO
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, extra_args={"loss": self.loss})
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step = self.step + 1,
            params = new_params,
            opt_state = new_opt_state,
            **kwargs
        )

class VeloTrainerModule(TrainerModule):
    def init_optimizer(self, num_epochs : int, num_steps_per_epoch : int):
        self.optimizer_name = 'velo'
        # Initialize frozen VeLO
        NUM_STEPS = num_epochs * num_steps_per_epoch
        opt = prefab.optax_lopt(NUM_STEPS)

        # Initialize training state
        self.state = VeloState.create(
            apply_fn = self.state.apply_fn,
            params = self.state.params,
            batch_stats = self.state.batch_stats,
            tx = opt,
            rng = self.state.rng
        )