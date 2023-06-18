import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from TrainerModule import TrainerModule, TrainState
from typing import Any, Dict
from flax.training import train_state
from learned_optimization.research.general_lopt import prefab
import optax
from copy import copy
from jaxopt import tree_util
import jax.numpy as jnp
from flax import linen as nn

class VeloState(TrainState):
    def apply_gradients(self, *, grads, loss, **kwargs):
        # Change update signature to pass loss as expected by VeLO
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, extra_args={"loss": loss})
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step = self.step + 1,
            params = new_params,
            opt_state = new_opt_state,
            **kwargs
        )

class VeloTrainerModule(TrainerModule):
    def __init__(self,
                 model_class : nn.Module,
                 model_hparams : Dict[str, Any],
                 optimizer_hparams : Dict[str, Any],
                 exmp_input : Any,
                 seed : int = 42,
                 logger_params : Dict[str, Any] = None,
                 enable_progress_bar : bool = True,
                 debug : bool = False,
                 check_val_every_n_epoch : int = 1,
                 **kwargs):
        optimizer_hparams['optimizer'] = 'VeLO'
        super().__init__(
                 model_class,
                 model_hparams,
                 optimizer_hparams,
                 exmp_input,
                 seed,
                 logger_params,
                 enable_progress_bar,
                 debug,
                 check_val_every_n_epoch,
                 **kwargs)
    
    def init_optimizer(self, num_epochs : int, num_steps_per_epoch : int):
        if not self.add_l2reg and self.optimizer_hparams['weight_decay'] > 0:
            raise Exception("""Add L2 regularization flag is off but weight decay is passed.
                      Weight decay is not utilized with VeLO.
                      If this is the intended behaviour, then set weight decay to zero.""")
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