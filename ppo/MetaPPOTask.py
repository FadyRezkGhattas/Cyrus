import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Any
import dataclasses as dc

import envpool
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex
from learned_optimization.research.general_lopt import prefab
from VeLO.VeloTrainState import VeloState
from torch.utils.tensorboard import SummaryWriter
from baseline.common import *

class MetaPPOTask():
    args : Any = None
    network : Network = None
    actor : Actor = None
    critic : Critic = None
    ppo_loss_grad_fn : Any = None
    step_once_fn : Any = None
    start_time = time.time()
    
    def init(self, key):
        # early environment initialization for getting observation/action spaces
        envs = make_env(self.args.env_id, self.args.seed, self.args.num_envs)()
        handle, recv, send, self.step_env = envs.xla()

        # agent setup
        self.network = Network()
        self.actor = Actor(action_dim=envs.single_action_space.n)
        self.critic = Critic()
        self.params : AgentParams = None

        # Set Jitted Functions and Partial Definitions
        self.ppo_loss_grad_fn = jax.value_and_grad(MetaPPOTask.ppo_loss, has_aux=True)
        self.step_once_fn=partial(self.step_once, env_step_fn=step_env_wrapped)

        # Initialize Agent
        return self.reinitialize_agent_params(key)

    def reinitialize_agent_params(self, key):
        # early environment initialization for getting observation/action spaces
        envs = make_env(self.args.env_id, self.args.seed, self.args.num_envs)()

        key, network_key, actor_key, critic_key = jax.random.split(key, 4)
        network_params = self.network.init(network_key, np.array([envs.single_observation_space.sample()]))
        actor_params = self.actor.init(actor_key, self.network.apply(network_params, np.array([envs.single_observation_space.sample()])))
        critic_params = self.critic.init(critic_key, self.network.apply(network_params, np.array([envs.single_observation_space.sample()])))

        return AgentParams(
            network_params,
            actor_params,
            critic_params
        ), key
    
    @staticmethod
    def step_once(carry, step, env_step_fn, step_env):
        agent_state, episode_stats, obs, terminated, truncated, key, handle = carry
        action, logprob, value, key = MetaPPOTask.get_action_and_value(agent_state, obs, key)

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
    
    @staticmethod
    def get_action_and_value(
            agent_state: VeloState,
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

    @staticmethod 
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
    
    @staticmethod
    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages
    
    @staticmethod
    def compute_gae(agent_state: VeloState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        args
    ):
        next_value = critic.apply(
            agent_state.params.critic_params, network.apply(agent_state.params.network_params, next_obs)
        ).squeeze()

        advantages = jnp.zeros((args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        
        compute_gae_once = partial(MetaPPOTask.compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    @staticmethod
    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, norm_adv, args):
        newlogprob, entropy, newvalue = MetaPPOTask.get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - args.mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))
    
    @staticmethod
    def update_minibatch(carry, minibatch, ppo_loss_grad_fn, args):
        meta_params, agent_state = carry
        (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
            agent_state.params,
            minibatch.obs,
            minibatch.actions,
            minibatch.logprobs,
            minibatch.advantages,
            minibatch.returns,
            args
        )
        agent_state = agent_state.apply_gradients(grads=grads, tx_params=meta_params, loss=loss, max_grad_norm=args.max_grad_norm)
        return (meta_params, agent_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

    @staticmethod
    def update_epoch(carry, unused_inp, storage, ppo_loss_grad_fn, args):
        meta_params, agent_state, key = carry
        key, subkey = jax.random.split(key)

        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])
        
        # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(subkey, x)
            x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
            return x
        
        flatten_storage = jax.tree_map(flatten, storage)
        shuffled_storage = jax.tree_map(convert_data, flatten_storage)
        
        (meta_params, agent_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            partial(MetaPPOTask.update_minibatch, ppo_loss_grad_fn=ppo_loss_grad_fn, args=args),
            init=(meta_params, agent_state),
            xs=shuffled_storage
        )
        return (meta_params, agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)
    
    def meta_rollout(self, agent_state, episode_stats, next_obs, terminated, truncated, key, handle):
        (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), storage = jax.lax.scan(
            self.step_once_fn, (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), (), self.args.num_steps // self.args.num_envs
        )
        return agent_state, episode_stats, next_obs, terminated, truncated, storage, key, handle
    
    @staticmethod
    def inner_epoch(carry, unused_t, ppo_loss_grad_fn, step_env, args):
        (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), storage = jax.lax.scan(
            partial(MetaPPOTask.step_once, env_step_fn=step_env_wrapped, step_env=step_env),
            (agent_state, episode_stats, next_obs, terminated, truncated, key, handle),
            (),
            length=args.num_steps
        )
        
        storage = MetaPPOTask.compute_gae(agent_state, next_obs, jnp.logical_or(terminated, truncated), storage, args)

        (meta_params, agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            partial(MetaPPOTask.update_epoch, storage=storage, ppo_loss_grad_fn=ppo_loss_grad_fn, args=args),
            init=(meta_params, agent_state, key),
            xs=None,
            length=args.update_epoch
        )

        return (meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle), (loss, pg_loss, v_loss, entropy_loss, approx_kl)
    
    def meta_loss(self, meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle):
        # Inner-Loop: Agent Learning
        #(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = self.inner_epoch(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle)

        (meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            partial(MetaPPOTask.inner_epoch, ppo_loss_grad_fn=self.ppo_loss_grad_fn, step_env=self.step_env, args=self.args),
            init=(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle),
            xs=None,
            length=1
        )
        
        # Outer-Loop (meta-loss)
        # rollout for batch_size (1024) which comes from num_steps * num_envs
        agent_state, episode_stats, next_obs, terminated, truncated, storage, key, handle = self.meta_rollout(agent_state, episode_stats, next_obs, terminated, truncated, key, handle)
        storage = self.compute_gae(agent_state, next_obs, jnp.logical_or(terminated, truncated), storage, self.args)

        key, subkey = jax.random.split(key)
        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])
        
        flatten_storage = jax.tree_map(flatten, storage)
        
        meta_loss, ret = self.ppo_loss(
            agent_state.params,
            flatten_storage.obs,
            flatten_storage.actions,
            flatten_storage.logprobs,
            flatten_storage.advantages,
            flatten_storage.returns)

        return meta_loss, (agent_state, key,
                           episode_stats, next_obs, terminated, truncated, handle,
                           loss, pg_loss, v_loss, entropy_loss, approx_kl)