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
from VeloTrainState import VeloState

def make_env_vmap(env_id, seed, num_envs, key=None):
    envs = envpool.make(
        env_id,
        env_type="gym",
        num_envs=num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=seed
    )
    envs.num_envs = num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    handle, recv, send, step_env = envs.xla()
    next_obs, info = envs.reset()
    terminated = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    truncated = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)

    episode_stats = EpisodeStatistics(
                episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
                episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
                returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
                returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
            )

    return next_obs, terminated, truncated, handle, episode_stats

if __name__ == '__main__':
    args = parse_args()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Init env
    # note: all environments are initialized to the same state. what will differ is the agent parameters working with the different environments
    keys = jax.random.split(key, 4) #TODO: change to population size
    next_obs, terminated, truncated, handle, episode_stats = jax.vmap(make_env_vmap, in_axes=(None, None, None, 0))(args.env_id, args.seed, args.num_envs, keys)

    # Init PPO
    ppo_task = PPOTask()
    ppo_task.args = args
    params, keys = jax.vmap(ppo_task.init)(keys)

    def init_velo_state(agent_params):
        total_steps = args.num_updates * args.update_epochs * args.num_minibatches
        return VeloState.create(
            apply_fn=None,
            params=agent_params,
            tx=get_optax_velo(total_steps)
        )
    velo_states = jax.vmap(init_velo_state)(params)

    for update in range(1, args.num_updates + 1):
        _ = jax.vmap(ppo_task.update)(velo_states, keys, episode_stats, next_obs, terminated, truncated, handle)