import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import jax
import jax.numpy as jnp

from TrainerModule import TrainerModule
from VeloTrainerModule import VeloTrainerModule
import optax

class ResNetTrainer(TrainerModule):
    def __init__(self, model_class,
                 num_classes : int,
                 dtype : jnp.dtype = jnp.float32,
                 **kwargs):
        super().__init__(model_class=model_class,
                         model_hparams={
                             'num_classes': num_classes,
                             'dtype': dtype
                         },
                         **kwargs)
    
    def create_functions(self):
        def loss_function(params, batch_stats, batch, train):
            img, labels = batch
            output = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                    img,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)
            logits, new_model_state = output if train else (output, None)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (new_model_state, acc)
        
        def train_step(state, batch):
            loss_fn = lambda params: loss_function(params, state.batch_stats, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, new_model_state, acc = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'], loss=loss)
            metrics = {'loss': loss, 'accuracy': acc}
            return state, metrics
        
        def eval_step(state, batch):
            loss, (_, acc) = loss_function(state.params, state.batch_stats, batch, train=False)
            return {'loss': loss, 'accuracy': acc}

        return train_step, eval_step

    def run_model_init(self, exmp_input, init_rng):
        imgs, _ = exmp_input
        return self.model.init({'params': init_rng}, x=imgs, train=True)
    
    def print_tabulate(self, exmp_input):
        imgs, _ = exmp_input
        print(self.model.tabulate(rngs={'params': jax.random.PRNGKey(0)}, x=imgs, train=True))