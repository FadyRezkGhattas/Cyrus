from typing import Any

import jax
import jax.numpy as jnp
import optax
import chex
from flax import struct, core
from flax.training.train_state import TrainState

class VeloState(TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats : Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng : Any = None
    # add args to access some global hyperparameters
    args : Any = None
    # parameters for meta model
    tx_params : core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    # optimizer for meta-model
    meta_tx : optax.GradientTransformation = struct.field(pytree_node=False)
    # optimizer state for meta-model optimizer
    meta_opt_state : optax.OptState = struct.field(pytree_node=True)

    def clip_grads(self, *, grads, max_grad_norm):
        # Clipping gradients as implemented here: https://github.com/deepmind/optax/blob/master/optax/_src/clipping.py#L91
        # replicating logic to avoid editing the chain operation to accept extra_args as velo expects
        g_norm = optax.global_norm(grads)
        trigger = jnp.squeeze(g_norm < max_grad_norm)
        chex.assert_shape(trigger, ())  # A scalar.
        def clip_fn(t):
            return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_grad_norm)
        grads = jax.tree_util.tree_map(clip_fn, grads)

    def apply_gradients(self, *, grads, loss, max_grad_norm, **kwargs):
        # Clipping gradients as implemented here: https://github.com/deepmind/optax/blob/master/optax/_src/clipping.py#L91
        # replicating logic to avoid editing the chain operation to accept extra_args as velo expects
        g_norm = optax.global_norm(grads)
        trigger = jnp.squeeze(g_norm < max_grad_norm)
        chex.assert_shape(trigger, ())  # A scalar.
        def clip_fn(t):
            return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_grad_norm)
        grads = jax.tree_util.tree_map(clip_fn, grads)
    
        # Change update signature to pass loss as expected by VeLO
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, self.tx_params, extra_args={"loss": loss})
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step = self.step + 1,
            params = new_params,
            opt_state = new_opt_state,
            **kwargs
        )
    
    def apply_meta_gradients(self, *, meta_grads, meta_max_grad_norm = None, **kwargs):
        if meta_max_grad_norm != None:
            self.clip_grads(grads=meta_grads, max_grad_norm=meta_max_grad_norm)
        # Change update signature to pass loss as expected by VeLO
        updates, new_meta_opt_state = self.meta_tx.update(meta_grads, self.meta_opt_state, self.tx_params)
        new_meta_params = optax.apply_updates(self.tx_params, updates)
        
        return self.replace(
            tx_params = new_meta_params,
            meta_opt_state = new_meta_opt_state,
            **kwargs
        )