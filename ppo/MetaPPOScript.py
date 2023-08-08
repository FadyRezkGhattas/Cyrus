import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Any
import dataclasses as dc

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter
from baseline.common import *
from VeLO.LoadVeLO import get_optax_velo
from VeLO.VeloTrainState import VeloState

def flatten_params(params):
    ff_mod_stack = jax.tree_leaves(params["ff_mod_stack"])
    lstm_init_state = jax.tree_leaves(params["lstm_init_state"])
    rnn_params = jax.tree_leaves(params["rnn_params"])
    return {"ff_mod_stack": ff_mod_stack, "lstm_init_state": lstm_init_state, "rnn_params": rnn_params}

if __name__ == '__main__':
    args = parse_args()
    use_velo = 'velo' if args.use_velo else 'adam'
    run_name = f"{use_velo}__{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # env setup
    envs = make_env(args.env_id, args.seed, args.num_envs)()
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )
    handle, recv, send, step_env = envs.xla()
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # TRY NOT TO MODIFY: start the game
    next_obs, info = envs.reset()
    terminated = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    truncated = jnp.zeros(args.num_envs, dtype=jax.numpy.bool_)
    episode_stats = EpisodeStatistics(
                episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
                episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
                returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
                returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
            )

    # Agent Setup
    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
    actor_params = actor.init(actor_key, network.apply(network_params, np.array([envs.single_observation_space.sample()])))
    critic_params = critic.init(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()])))
    params = AgentParams(network_params, actor_params, critic_params)

    # Agent Optimizer Setup
    total_steps = args.num_updates * args.update_epochs * args.num_minibatches
    lopt, meta_params = get_optax_velo(total_steps)
    flattened_meta_params = flatten_params(meta_params)
    agent_state = VeloState.create(
        apply_fn=None,
        params=params,
        tx=lopt,
        tx_params=meta_params,
    )

    meta_optimizer = optax.chain(
      optax.clip(1),
      optax.adam(1e-3)
    )
    meta_optimizer_state = meta_optimizer.init(meta_params)

    # Meta-Optimizer Setup
    meta_optimizer = optax.chain(
      optax.clip(1),
      optax.adam(1e-3)
    )
    meta_optimizer_state = meta_optimizer.init(meta_params)

    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        hidden = network.apply(params.network_params, x)
        logits = actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value
    
    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages
    
    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    def compute_gae(
        agent_state: VeloState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(
            agent_state.params.critic_params, network.apply(agent_state.params.network_params, next_obs)
        ).squeeze()

        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage
    
    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, norm_adv):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))
    
    def meta_loss(params, x, a, logp, mb_advantages, mb_returns, meta_params, norm_adv, initial_meta_params):
        loss, (pg_loss, v_loss, entropy_loss, approx_kl) = ppo_loss(params, x, a, logp, mb_advantages, mb_returns, norm_adv)
        flattened_updated_meta_params = flatten_params(meta_params)
        regularization_value = jnp.sum(flattened_meta_params - initial_meta_params)
        loss = loss + args.loss_distance_penalty
    
    agent_loss_fn = partial(ppo_loss, norm_adv = args.norm_adv)
    agent_grad_update_fn = jax.value_and_grad(agent_loss_fn, has_aux=True)

    meta_loss_fn = partial(ppo_loss, norm_adv = args.norm_adv)
    
    def get_action_and_value(agent_state: VeloState,
            next_obs: np.ndarray,
            key: jax.random.PRNGKey
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        hidden = network.apply(agent_state.params.network_params, next_obs)
        logits = actor.apply(agent_state.params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = critic.apply(agent_state.params.critic_params, hidden)
        return action, logprob, value.squeeze(1), key

    def step_once( carry, step, env_step_fn):
        agent_state, episode_stats, obs, terminated, truncated, key, handle = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        episode_stats, handle, (next_obs, reward, terminated, truncated, info) = env_step_fn(episode_stats, handle, action, step_env)
        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=jnp.logical_or(terminated, truncated),
            values=value,
            rewards=reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return ((agent_state, episode_stats, next_obs, terminated, truncated, key, handle), storage)
    step_once_fn=partial(step_once, env_step_fn=step_env_wrapped)

    def minibatch_step(
            carry,
            minibatch
    ):
        meta_params, agent_state = carry
        (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = agent_grad_update_fn(
            agent_state.params,
            minibatch.obs,
            minibatch.actions,
            minibatch.logprobs,
            minibatch.advantages,
            minibatch.returns,
        )
        agent_state = agent_state.apply_gradients(grads=grads, tx_params=meta_params, loss=loss, max_grad_norm=args.max_grad_norm)
        return (meta_params, agent_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)
    
    # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
    def convert_data(x, key):
        x = x.reshape((-1,) + x.shape[2:])
        x = jax.random.permutation(key, x)
        x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
        return x
    
    def sgd_step(carry, unused_t, storage):
        meta_params, agent_state, key = carry
        key, subkey = jax.random.split(key)
        shuffled_storage = jax.tree_map(partial(convert_data, key=subkey), storage)
        
        (meta_params, agent_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            minibatch_step, (meta_params, agent_state), xs=shuffled_storage
        )
        return (meta_params, agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)
    
    def training_step(carry, unused_t):
        meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle = carry
        
        # Rollout
        (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), storage = jax.lax.scan(
            step_once_fn, (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), (), args.num_steps
        )

        storage = compute_gae(agent_state, next_obs, jnp.logical_or(terminated, truncated), storage)

        (meta_params, agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            partial(sgd_step, storage=storage),
            init=(meta_params, agent_state, key),
            xs=None,
            length=args.update_epochs
        )

        return (meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle), (loss, pg_loss, v_loss, entropy_loss, approx_kl)
    
    def agent_update_and_meta_loss(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle):
        """Agent learning: update agent params"""
        (meta_params, agent_state, key, inner_episode_stats, next_obs, terminated, truncated, handle), (inner_loss, inner_pg_loss, inner_v_loss, inner_entropy_loss, inner_approx_kl) = jax.lax.scan(
            f=training_step,
            init=(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle),
            length=1,
            xs=None,
        )
        
        """Meta learning: update meta params"""
        # Rollout
        (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), storage = jax.lax.scan(
            step_once_fn, (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), (), args.num_meta_steps
        )

        storage = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), storage)

        meta_loss, (pg_loss, v_loss, entropy_loss, approx_kl) = meta_loss_fn(agent_state.params,
            storage.obs,
            storage.actions,
            storage.logprobs,
            storage.advantages,
            storage.returns)
        
        return meta_loss, (meta_params, agent_state, key, inner_episode_stats, next_obs, terminated, truncated, handle, inner_loss, inner_pg_loss, inner_v_loss, inner_entropy_loss, inner_approx_kl)

    def meta_training_step(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle, meta_optim_state):
        ret, meta_grads = jax.value_and_grad(agent_update_and_meta_loss, has_aux=True)(
            meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle
        )
        meta_loss_, meta_params, agent_state, key, inner_episode_stats, next_obs, terminated, truncated, handle, inner_loss, inner_pg_loss, inner_v_loss, inner_entropy_loss, inner_approx_kl = ret[0], *ret[1]

        meta_param_update, meta_optim_state = meta_optimizer.update(meta_grads, meta_optim_state)
        meta_params = optax.apply_updates(meta_params, meta_param_update)
        
        return meta_params, agent_state, key, inner_episode_stats, next_obs, terminated, truncated, handle, meta_optim_state, meta_loss_, inner_loss, inner_pg_loss, inner_v_loss, inner_entropy_loss, inner_approx_kl

    #agent_update_and_meta_loss_jitted = jax.jit(agent_update_and_meta_loss)
    jitted_meta_training_step = jax.jit(meta_training_step)

    start_time = time.time()
    global_step = 0
    for _ in range(1000000):
        update_time_start = time.time()
        # Without meta-learning (this function scans gracefully without recompilation issues)
        # (meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
        #    f=training_step,
        #    init=(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle),
        #    length=1,
        #    xs=None,
        #)

        # Without meta-gradients (agent_update_and_meta_loss works gracefully without recompilation issues even when decorated with @jax.jit)
        #meta_loss, (meta_params, agent_state, key, inner_episode_stats, next_obs, terminated, truncated, handle, inner_loss, inner_pg_loss, inner_v_loss, inner_entropy_loss, inner_approx_kl) = agent_update_and_meta_loss_jitted(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle)

        # Complete meta-learning (no recompilation issues but script throws OOM with standard batch size of 1024)
        meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle, meta_optim_state, meta_loss_, inner_loss, inner_pg_loss, inner_v_loss, inner_entropy_loss, inner_approx_kl = jitted_meta_training_step(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle, meta_optimizer_state)

        global_step += args.num_steps * args.num_envs
        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)), global_step
        )
        writer.add_scalar("losses/value_loss", inner_v_loss[-1, -1, -1].item(), global_step)
        writer.add_scalar("losses/policy_loss", inner_pg_loss[-1, -1, -1].item(), global_step)
        writer.add_scalar("losses/entropy", inner_entropy_loss[-1, -1, -1].item(), global_step)
        writer.add_scalar("losses/approx_kl", inner_approx_kl[-1, -1, -1].item(), global_step)
        writer.add_scalar("losses/inner_loss", inner_loss[-1, -1, -1].item(), global_step)
        #writer.add_scalar("losses/meta_loss", meta_loss_.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - update_time_start)), global_step
        )