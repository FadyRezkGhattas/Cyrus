import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import jax
import jax.numpy as jnp

from TrainerModule import TrainerModule
from VeloTrainerModule import VeloTrainerModule
import optax
from jaxopt._src import tree_util

class LinearTrainer(VeloTrainerModule):
    def __init__(self, model_class,
                 **kwargs):
        super().__init__(model_class=model_class,
                         **kwargs)
    
    def create_functions(self):
        def mse_loss(params, batch):
            x, y = batch
            pred = self.model.apply({'params': params}, x)
            loss = ((pred - y) ** 2).mean()
            return loss
        
        def train_step(state, batch):
            loss_fn = lambda params: mse_loss(params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads, loss= loss)
            metrics = {'loss': loss}
            return state, metrics
        
        def eval_step(state, batch):
            return {}

        return train_step, eval_step

    def run_model_init(self, exmp_input, init_rng):
        x, y = exmp_input
        return self.model.init({'params': init_rng}, x=x)
    
    def print_tabulate(self, exmp_input):
        x, y = exmp_input
        print(self.model.tabulate(rngs={'params': jax.random.PRNGKey(0)}, x=x))