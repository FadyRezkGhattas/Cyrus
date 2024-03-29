import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import jax
import jax.numpy as jnp

from TrainerModule import TrainerModule
from VeloTrainerModule import VeloTrainerModule
import optax
import jax.tree_util as tree_util

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
            state, updates = state.apply_gradients(grads=grads, loss= loss, return_updates=True)
            grads, updates = grads.unfreeze(), updates.unfreeze()
            grads, updates = jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(updates)
            grads = jnp.concatenate([jnp.ravel(grad) for grad in grads])
            updates = jnp.concatenate([jnp.ravel(update) for update in updates])
            update_direction = jnp.dot(grads, updates)
            cosine_similarity = update_direction / (jnp.linalg.norm(updates)*jnp.linalg.norm(grads))
            metrics = {'loss': loss, 'update_direction': update_direction, 'cosine_similarity': cosine_similarity}
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

class LinearClassifierTrainer(LinearTrainer):
    def __init__(self, model_class,
                 **kwargs):
        super().__init__(model_class=model_class,
                         **kwargs)
        
    def create_functions(self):
        def loss_function(params, batch):
            x, labels = batch
            logits = self.model.apply({'params': params}, x)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, acc
        
        def train_step(state, batch):
            loss_fn = lambda params : loss_function(params, batch)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc = ret[0], ret[1]
            state, updates = state.apply_gradients(grads=grads, loss=loss, return_updates=True)
            grads, updates = grads.unfreeze(), updates.unfreeze()
            grads, updates = jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(updates)
            grads = jnp.concatenate([jnp.ravel(grad) for grad in grads])
            updates = jnp.concatenate([jnp.ravel(update) for update in updates])
            update_direction = jnp.dot(grads, updates)
            cosine_similarity = update_direction / (jnp.linalg.norm(updates)*jnp.linalg.norm(grads))
            metrics = {'loss': loss, 'accuracy': acc, 'update_direction': update_direction, 'cosine_similarity': cosine_similarity}
            return state, metrics
        
        def eval_step(state, batch):
            return {}

        return train_step, eval_step