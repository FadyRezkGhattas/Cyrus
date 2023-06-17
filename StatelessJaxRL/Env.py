import jax
import jax.numpy as jnp
from jax import random
import math
from gym import spaces
from functools import partial

class SkeletonEnv:

    def __init__(self) -> None:
        self.random_limit = 0.05

    def _get_obs(self, state):
        return state
    
    def _maybe_reset(self, env_state, done):
        key = env_state[1]
        return jax.lax.cond(done, self._reset, lambda key: env_state, key)
    
    def _reset(self, key):
        new_state = random.uniform(key, minval=-self.random_limit,
                                   maxval=self.random_limit, shape=(4,))
        new_key = random.split(key)[0]
        return new_state, new_key
    
    def step(self, env_state, action):
        state, key = env_state
        new_state = state + action

        reward, done, info = 1., False, None

        env_state = new_state, key
        env_state = self._maybe_reset(env_state, done)
        new_state = env_state[0]

        return env_state, self._get_obs(new_state), reward, done, info
    
    def reset(self, key):
        env_state = self._reset(key)
        new_state = env_state[0]
        return env_state, self._get_obs(new_state)

class JaxCartPole:
    def __init__(self) -> None:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.random_limit = 0.05

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = jnp.array(
            [
                self.x_threshold * 2,
                jnp.finfo(jnp.float32).max,
                self.theta_threshold_radians * 2,
                jnp.finfo(jnp.float32).max,
            ],
            dtype=jnp.float32,
        )
        self.action_space = spaces.Discrete(2)
        #self.observation_space = spaces.Box(-high, high, dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, env_state, action):
        state, key = env_state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag * (2 * action - 1)
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        done = ((x < -self.x_threshold) | (x > self.x_threshold)
                | (theta > self.theta_threshold_radians) 
                | (theta < -self.theta_threshold_radians))

        reward = 1.

        env_state = jnp.array([x, x_dot, theta, theta_dot]), key
        env_state = self._maybe_reset(env_state, done)
        new_state = env_state[0]
        return env_state, self._get_obsv(new_state), reward, done, {}
    
    def _get_obsv(self, state):
      return state

    def _maybe_reset(self, env_state, done):
      key = env_state[1]
      return jax.lax.cond(done, self._reset, lambda key: env_state, key)

    def _reset(self, key):
      new_state = random.uniform(key, minval=-self.random_limit, 
                                      maxval=self.random_limit, shape=(4,))
      new_key = random.split(key)[0]
      return new_state, new_key

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
      env_state = self._reset(key)
      initial_state = env_state[0]
      return env_state, self._get_obsv(initial_state)


# This is to show how stateless CartPole can be
seed = 0
key = random.PRNGKey(seed)
env = JaxCartPole()
env_state, inital_obsv = env.reset(key)
action = 1
env_state, obsv, reward, done, info = env.step(env_state, 1)

# This is how to vmap step and reset
vstep = jax.jit(jax.vmap(env.step, in_axes=((0, 0), 0), out_axes=((0, 0), 0, 0, 0, 0), axis_name="batch_axis"))
vreset = jax.jit(jax.vmap(env.reset, out_axes=((0, 0), 0), axis_name="batch_axis"))