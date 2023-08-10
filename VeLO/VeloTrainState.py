from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
import chex
from flax import struct, core

class VeloState(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, tx_params, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(tx_params, params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def apply_gradients(self, *, grads, tx_params, loss, max_grad_norm, **kwargs):
        if max_grad_norm != None:
            # Clipping gradients as implemented here: https://github.com/deepmind/optax/blob/master/optax/_src/clipping.py#L91
            # replicating logic to avoid editing the chain operation to accept extra_args as velo expects
            g_norm = optax.global_norm(grads)
            trigger = jnp.squeeze(g_norm < max_grad_norm)
            chex.assert_shape(trigger, ())  # A scalar.
            def clip_fn(t):
                return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * max_grad_norm)
            grads = jax.tree_util.tree_map(clip_fn, grads)
    
        # Change update signature to pass loss as expected by VeLO
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, tx_params, extra_args={"loss": loss})
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step = self.step + 1,
            params = new_params,
            opt_state = new_opt_state,
            **kwargs
        )