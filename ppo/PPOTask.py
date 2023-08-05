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
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter
from baseline.common import *

class PPOTask():
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
        self.network.apply = jax.jit(self.network.apply)
        self.actor.apply = jax.jit(self.actor.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.ppo_loss_grad_fn = jax.jit(jax.value_and_grad(self.ppo_loss, has_aux=True))
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

    def step_once(self, carry, step, env_step_fn):
        agent_state, episode_stats, obs, terminated, truncated, key, handle = carry
        action, logprob, value, key = self.get_action_and_value(agent_state, obs, key)

        episode_stats, handle, (next_obs, reward, terminated, truncated, info) = env_step_fn(episode_stats, handle, action, self.step_env)
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
    
    def get_action_and_value(self,
            agent_state: TrainState,
            next_obs: np.ndarray,
            key: jax.random.PRNGKey
    ):
        """sample action, calculate value, logprob, entropy, and update storage"""
        hidden = self.network.apply(agent_state.params.network_params, next_obs)
        logits = self.actor.apply(agent_state.params.actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        value = self.critic.apply(agent_state.params.critic_params, hidden)
        return action, logprob, value.squeeze(1), key
        
    def get_action_and_value2(self,
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray
    ):
        """calculate value, logprob of supplied `action`, and entropy"""
        hidden = self.network.apply(params.network_params, x)
        logits = self.actor.apply(params.actor_params, hidden)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        # normalize the logits https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        p_log_p = logits * jax.nn.softmax(logits)
        entropy = -p_log_p.sum(-1)
        value = self.critic.apply(params.critic_params, hidden).squeeze()
        return logprob, entropy, value
    
    def compute_gae_once(self, carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages
    
    def compute_gae(self,
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = self.critic.apply(
            agent_state.params.critic_params, self.network.apply(agent_state.params.network_params, next_obs)
        ).squeeze()

        advantages = jnp.zeros((self.args.num_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        
        compute_gae_once = partial(self.compute_gae_once, gamma=self.args.gamma, gae_lambda=self.args.gae_lambda)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def ppo_loss(self, params, x, a, logp, mb_advantages, mb_returns):
        newlogprob, entropy, newvalue = self.get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if self.args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    def update_minibatch(self, carry, minibatch):
        meta_params, agent_state = carry
        (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = self.ppo_loss_grad_fn(
            agent_state.params,
            minibatch.obs,
            minibatch.actions,
            minibatch.logprobs,
            minibatch.advantages,
            minibatch.returns,
        )
        agent_state = agent_state.apply_gradients(grads=grads, tx_params=meta_params, loss=loss, max_grad_norm=self.args.max_grad_norm)
        return (meta_params, agent_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

    def update_epoch(self, carry, unused_inp):
        meta_params, agent_state, storage, key = carry
        key, subkey = jax.random.split(key)

        def flatten(x):
            return x.reshape((-1,) + x.shape[2:])
        
        # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(subkey, x)
            x = jnp.reshape(x, (self.args.num_minibatches, -1) + x.shape[1:])
            return x
        
        flatten_storage = jax.tree_map(flatten, storage)
        shuffled_storage = jax.tree_map(convert_data, flatten_storage)
        
        (meta_params, agent_state), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            self.update_minibatch, (meta_params, agent_state), shuffled_storage
        )
        return (meta_params, agent_state, storage, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)
    
    def rollout(self, agent_state, episode_stats, next_obs, terminated, truncated, key, handle):
        (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), storage = jax.lax.scan(
            self.step_once_fn, (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), (), self.args.num_steps
        )
        return agent_state, episode_stats, next_obs, terminated, truncated, storage, key, handle

    @partial(jax.jit, static_argnums=(0,))
    def inner_epoch(self, carry, unused_t):
        meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle = carry
        agent_state, episode_stats, next_obs, terminated, truncated, storage, key, handle = self.rollout(
            agent_state, episode_stats, next_obs, terminated, truncated, key, handle
        )
        
        storage = self.compute_gae(agent_state, next_obs, jnp.logical_or(terminated, truncated), storage)

        (meta_params, agent_state, storage, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            self.update_epoch, (meta_params, agent_state, storage, key), (), length=self.args.update_epochs
        )

        return (meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle), (loss, pg_loss, v_loss, entropy_loss, approx_kl)
    
    @partial(jax.jit, static_argnums=(0,))
    def meta_loss(self, meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle):
        # Inner-Loop: Agent Learning
        (meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle), (loss, pg_loss, v_loss, entropy_loss, approx_kl) = jax.lax.scan(
            f=self.inner_epoch,
            init=(meta_params, agent_state, key, episode_stats, next_obs, terminated, truncated, handle),
            length=self.args.num_inner_epochs,
            xs=None
        )
        # Outer-Loop (meta-loss)
        # rollout for batch_size (1024) which comes from num_steps * num_envs
        agent_state, episode_stats, next_obs, terminated, truncated, storage, key, handle = self.rollout(
            agent_state, episode_stats, next_obs, terminated, truncated, key, handle
        )
        storage = self.compute_gae(agent_state, next_obs, jnp.logical_or(terminated, truncated), storage)

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