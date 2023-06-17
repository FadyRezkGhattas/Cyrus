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

    # Tracking and Logging
    use_velo = 'velo' if args.use_velo else 'adam'
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Init env
    envs = make_env(args.env_id, args.seed, args.num_envs)()
    handle, recv, send, step_env = envs.xla()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # TRY NOT TO MODIFY: start the game
    next_obs, info = envs.reset()
    terminated = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    truncated = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)

    # Data Containers
    episode_stats = EpisodeStatistics(
                episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
                episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
                returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
                returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
            )
    
    # Init PPO
    ppo_task = PPOTask()
    ppo_task.args = args
    params, key = ppo_task.init(key)

    total_steps = args.num_updates * args.update_epochs * args.num_minibatches
    agent_state = VeloState.create(
        apply_fn=None,
        params=params,
        tx=get_optax_velo(total_steps)
    )

    start_time = time.time()
    global_step = 0
    for update in range(1, args.num_updates + 1):
        update_time_start = time.time()
        agent_state, key, episode_stats, next_obs, terminated, truncated, handle, loss, pg_loss, v_loss, entropy_loss, approx_kl = ppo_task.update(agent_state, key, episode_stats, next_obs, terminated, truncated, handle)

        global_step += args.num_steps * args.num_envs
        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)), global_step
        )
        if not args.use_velo:
            writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), global_step)
        writer.add_scalar("losses/loss", loss[-1, -1].item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - update_time_start)), global_step
        )