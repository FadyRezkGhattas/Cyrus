import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from TrainerModule import TrainerModule
from MLP import MLP
from typing import Sequence
import jax
from VeloTrainerModule import VeloTrainerModule

class MLPRegressTrainer(TrainerModule):
    
    def __init__(self,
                 hidden_dims : Sequence[int],
                 output_dim : int,
                 **kwargs):
        super().__init__(model_class=MLP,
                         model_hparams={
                             'hidden_dims': hidden_dims,
                             'output_dim': output_dim
                         },
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
            loss = mse_loss(state.params, batch)
            return {'loss': loss}
        
        return train_step, eval_step