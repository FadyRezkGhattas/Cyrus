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

def short_segment_unroll(agent_state,
                         ppo_task,
                         args,
                         key,
                         inner_problem_length,
                         on_iteration,
                         truncation_length, # truncation_length is the length of steps to take in in the inner problem before doing meta-updates
                         ):
    """Run the ppo_task for a total of truncation_length. Each
     step increments on_iteration. if the on_iteration reaches inner_problem_length,
     the ppo_task and on_iteration are resetted. The unrolling continues with the new
     task until the truncation_length is reached. All losses are returned which can
     be over a single or multiple ppo_tasks. 

    Args:
        agent_state (VeloState): contains ppo task parameters, and optimizer
        ppo_task (PPOTask): Encapsulates the PPO algorithm, RL agent and environments to train on 
        args : arguments to be used to reset ppo_task if necessary
        key : random key to be used to reset ppo_task if necessary
        inner_problem_length (int): the total length the inner-problem is allowed to run for before resetting (equivalent to args.num_updates)
        on_iteration (int): the iteration at which the ppo_task is currently on (how far are we along args.num_updates)
        truncation_length (int): the total number of steps to run PPO agent for, collecting losses, irrespective of on_iteration

    Returns:
        _type_: agent_state, ppo_task, key, on_iteration, losses
    """
    #losses = np.array([])
    start_time = time.time()
    for i in range(truncation_length):
        # If we have trained for longer than total inner problem length, reset the inner problem.
        if on_iteration >= inner_problem_length:
            key1, key = jax.random.split(key)
            ppo_task = PPOTask(args)
            params, key = ppo_task.init(key)
            agent_state = VeloState.create(apply_fn=None, params=params, tx=agent_state.tx)
            on_iteration = 0
        
        agent_state, key, step_losses = ppo_task.update(agent_state, key, start_time)

        # clip the loss to prevent diverging inner models
        step_losses = np.array(step_losses.flatten())
        cutoff = np.full_like(step_losses, 3.0, np.float64)
        step_losses = jnp.where(jnp.isnan(step_losses), cutoff, step_losses)
        losses = np.append(losses, step_losses)

        on_iteration += 1
    return agent_state, ppo_task, key, on_iteration, losses


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

    on_iteration = 0
    truncation_length = int(args.num_updates/args.num_meta_updates)
    for update in range(10):
        agent_state, ppo_task, key, on_iteration, losses = short_segment_unroll(agent_state, ppo_task, args, key, args.num_updates, on_iteration, truncation_length)