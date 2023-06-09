import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Any, Generic, TypeVar
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
from VeloTrainState import VeloState

_T = TypeVar('_T')

class RefHolder(Generic[_T]):
    def __init__(self, value: _T):
        self.value = value

def _refholder_flatten(self):
    return (), self.value

def _refholder_unflatten(value, _):
    return RefHolder(value)

jax.tree_util.register_pytree_node(RefHolder, _refholder_flatten, _refholder_unflatten)

def short_segment_unroll(agent_state,
                         ppo_task,
                         args,
                         key,
                         inner_problem_length,
                         on_iteration,
                         truncation_length, # truncation_length is the length of steps to take in in the inner problem before doing meta-updates
                         start_time):
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
    losses = np.array([])
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
    return agent_state, ppo_task, key, on_iteration, losses, start_time

def short_segment_unroll_(agent_state,
                            ppo_task,
                            key,
                            inner_problem_length,
                            on_iteration,
                            truncation_length, # truncation_length is the length of steps to take in in the inner problem before doing meta-updates
                            start_time): #TODO: reset start_time in reset_problem
        def step(carry, step):
            agent_state, key, on_iteration, start_time = carry
            def reset_problem(k, agent_state):
                key1, key = jax.random.split(k)
                params, key = ppo_task.reinitialize_agent_params(key)
                agent_state = VeloState.create(apply_fn=None, params=params, tx=agent_state.tx)
                on_iteration = 0
                return agent_state, on_iteration, key
            
            # If we have trained for longer than total inner problem length, reset the inner problem.
            jax.lax.cond(on_iteration >= inner_problem_length,
                        lambda k, agent_state: (reset_problem(k, agent_state)),
                        lambda k, agent_state: (agent_state, on_iteration, k),
                        key, agent_state)
            
            # Optimizer Application to RL Agent
            agent_state, key, step_losses = ppo_task.update(agent_state, key, start_time)

            # clip the loss to prevent diverging inner models
            step_losses = np.array(step_losses.flatten())
            cutoff = np.full_like(step_losses, 3.0, np.float64)
            step_losses = jnp.where(jnp.isnan(step_losses), cutoff, step_losses)
            losses = np.append(losses, step_losses)

            on_iteration += 1
        
            return (agent_state, key, on_iteration, start_time), losses
        
        (agent_state, key, on_iteration, start_time), losses = jax.lax.scan(step, (agent_state, key, on_iteration, start_time), (), length=truncation_length)
        return agent_state, key, on_iteration, start_time, losses

#@partial(jax.jit, static_argnames=('ppo_tasks'))
def vec_short_segment_unroll(agent_states,
                             ppo_tasks,
                             args,
                             keys,
                             inner_problem_length,
                             on_iterations,
                             truncation_length,
                             start_times):
    agent_states, ppo_tasks, keys, on_iterations, losses, start_times = jax.vmap(short_segment_unroll,
             in_axes=(0, 0, None, 0, None, 0, None, 0))(agent_states,
                                                           ppo_tasks,
                                                           args,
                                                           keys,
                                                           inner_problem_length,
                                                           on_iterations,
                                                           truncation_length,
                                                           start_times)
    return agent_states, ppo_tasks, keys, on_iterations, jnp.mean(losses), start_times


if __name__ == '__main__':
    args = parse_args()
    num_tasks = 2
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    

    def init_single_problem(key):
        ppo_task = PPOTask()
        ppo_task.args = args
        params, key = ppo_task.init(key)

        total_steps = args.num_updates * args.update_epochs * args.num_minibatches
        agent_state = VeloState.create(
            apply_fn=None,
            params=params,
            tx=get_optax_velo(total_steps)
        )

        return RefHolder(ppo_task), agent_state, key

    keys = jax.random.split(key, num_tasks)
    ppo_tasks, agent_states, keys = jax.vmap(init_single_problem)(keys)

    on_iterations = jax.random.randint(key, [num_tasks], 0, args.num_updates)

    truncation_length = int(args.num_updates/args.num_meta_updates)
    start_time = time.time()
    for update in range(10):
        agent_state, key, on_iteration, start_time, losses = vec_short_segment_unroll(agent_states, ppo_tasks, args, keys, args.num_updates, on_iterations=on_iterations, truncation_length=truncation_length, start_times=start_time)