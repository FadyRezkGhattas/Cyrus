import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Any
import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
from learned_optimization.research.general_lopt import prefab
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from baseline.common import *
from PPOTask import *
from VeLO import get_optax_velo

class VeloState(TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats : Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng : Any = None
    # Save loss as a state because learned optimizers need it as input
    # Strange initialization because mutable jaxlib.xla_extension.ArrayImpl is not allowed (must use default_factory): https://github.com/google/jax/issues/14295
    loss : Any = dc.field(default_factory=lambda: jnp.asarray(0))

    def apply_gradients(self, *, grads, **kwargs):
        # Clipping gradients as implemented here: https://github.com/deepmind/optax/blob/master/optax/_src/clipping.py#L91
        # replicating logic to avoid editing the chain operation to accept extra_args as velo expects
        g_norm = optax.global_norm(grads)
        trigger = jnp.squeeze(g_norm < args.max_grad_norm)
        chex.assert_shape(trigger, ())  # A scalar.
        def clip_fn(t):
            return jax.lax.select(trigger, t, (t / g_norm.astype(t.dtype)) * args.max_grad_norm)
        grads = jax.tree_util.tree_map(clip_fn, grads)
    
        # Change update signature to pass loss as expected by VeLO
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params, extra_args={"loss": self.loss})
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step = self.step + 1,
            params = new_params,
            opt_state = new_opt_state,
            **kwargs
        )

if __name__ == '__main__':
    args = parse_args()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    ppo_task = PPOTask(args)
    params, key = ppo_task.init(key)

    total_steps = args.num_updates * args.update_epochs * args.num_minibatches
    agent_state = VeloState.create(
        apply_fn=None,
        params=params,
        tx=get_optax_velo(total_steps)
    )

    start_time = time.time()
    for update in range(1, args.num_updates + 1):
        agent_state, key, loss = ppo_task.update(agent_state, key, start_time)