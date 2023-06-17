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

    def debug(velo_state):
        jax.debug.print("this is loss {l}", l=velo_state.params.actor_params)
    
    jax.vmap(debug)(velo_states)