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
    def __init__(self, args):
        self.args = args

        # Tracking and Logging
        use_velo = 'velo' if args.use_velo else 'adam'
        self.run_name = f"{use_velo}__{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=self.run_name,
                monitor_gym=True,
                save_code=True,
            )
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        # temp environment for getting observation/action spaces
        envs = make_env(self.args.env_id, self.args.seed, self.args.num_envs)()

        # agent setup
        self.network = Network()
        self.actor = Actor(action_dim=envs.single_action_space.n)
        self.critic = Critic()
        self.params : AgentParams = None

        # Set Jitted Functions
        self.network.apply = jax.jit(self.network.apply)
        self.actor.apply = jax.jit(self.actor.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.ppo_loss_grad_fn = jax.value_and_grad(self.ppo_loss, has_aux=True)

        # Tracked metrics
        self.global_step = 0

    def init(self, key):
        # Initialize Envs
        self.envs = make_env(self.args.env_id, self.args.seed, self.args.num_envs)()
        self.episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros(self.args.num_envs, dtype=jnp.float32),
            episode_lengths=jnp.zeros(self.args.num_envs, dtype=jnp.int32),
            returned_episode_returns=jnp.zeros(self.args.num_envs, dtype=jnp.float32),
            returned_episode_lengths=jnp.zeros(self.args.num_envs, dtype=jnp.int32),
        )
        self.handle, self.recv, self.send, self.step_env = self.envs.xla()
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        self.step_once_fn=partial(self.step_once, env_step_fn=step_env_wrapped)

        # TRY NOT TO MODIFY: start the game
        self.global_step = 0
        self.start_time = time.time()
        self.next_obs, info = self.envs.reset()
        self.terminated = jnp.zeros(self.args.num_envs, dtype=jax.numpy.bool_)
        self.truncated = jnp.zeros(self.args.num_envs, dtype=jax.numpy.bool_)
        
        # Initialize Agent
        key, network_key, actor_key, critic_key = jax.random.split(key, 4)
        network_params = self.network.init(network_key, np.array([self.envs.single_observation_space.sample()]))
        actor_params = self.actor.init(actor_key, self.network.apply(network_params, np.array([self.envs.single_observation_space.sample()])))
        critic_params = self.critic.init(critic_key, self.network.apply(network_params, np.array([self.envs.single_observation_space.sample()])))

        self.params = AgentParams(
            network_params,
            actor_params,
            critic_params
        )

        return key

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
    
    @partial(jax.jit, static_argnums=(0,))
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
        
    @partial(jax.jit, static_argnums=(0,))
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
    
    @partial(jax.jit, static_argnums=(0,))
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

    @partial(jax.jit, static_argnums=(0,))
    def update_ppo(self,
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
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

            def update_minibatch(agent_state, minibatch):
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = self.ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                )
                # TODO: do whatever with gradients here
                agent_state = agent_state.apply_gradients(grads=grads, loss=loss)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=self.args.update_epochs
        ) #return the grads here if needed? If args.update_epochs is 4 and the minibatches are 4, then we have (4, 4, grads_shape)
        # critic dense_0 bias is for example has shape of (4,4,1)
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    def rollout(self, agent_state, episode_stats, next_obs, terminated, truncated, key, handle):
        (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), storage = jax.lax.scan(
            self.step_once_fn, (agent_state, episode_stats, next_obs, terminated, truncated, key, handle), (), self.args.num_steps
        )
        return agent_state, episode_stats, next_obs, terminated, truncated, storage, key, handle

    def update(self, agent_state, key, start_time):
        update_time_start = time.time()
        agent_state, self.episode_stats, self.next_obs, self.terminated, self.truncated, storage, key, self.handle = self.rollout(
            agent_state, self.episode_stats, self.next_obs, self.terminated, self.truncated, key, self.handle
        )
        storage = self.compute_gae(agent_state, self.next_obs, np.logical_or(self.terminated, self.truncated), storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = self.update_ppo(
            agent_state,
            storage,
            key,
        )

        self.global_step += self.args.num_steps * self.args.num_envs
        avg_episodic_return = np.mean(jax.device_get(self.episode_stats.returned_episode_returns))
        print(f"global_step={self.global_step}, avg_episodic_return={avg_episodic_return}")
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, self.global_step)
        self.writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(self.episode_stats.returned_episode_lengths)), self.global_step
        )
        if not self.args.use_velo:
            self.writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), self.global_step)
        self.writer.add_scalar("losses/value_loss", v_loss[-1, -1].item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss[-1, -1].item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss[-1, -1].item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl[-1, -1].item(), self.global_step)
        self.writer.add_scalar("losses/loss", loss[-1, -1].item(), self.global_step)
        print("SPS:", int(self.global_step / (time.time() - start_time)))
        self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - start_time)), self.global_step)
        self.writer.add_scalar(
            "charts/SPS_update", int(self.args.num_envs * self.args.num_steps / (time.time() - update_time_start)), self.global_step
        )

        return agent_state, key