# adapted from: https://github.com/google/learned_optimization/blob/main/learned_optimization/optimizers/opt_to_optax.py
# with the differene that opt is accessed from outside the wrapper class
# this allows changing parameters in-place

import os
import uuid
import dataclasses
from typing import Any, Mapping, Optional, Tuple, NamedTuple

import gin
import jax
import optax
import chex
from VeLO.hyper_v2 import HyperV2
from learned_optimization import checkpoints
from learned_optimization import filesystem
from learned_optimization import summary
from learned_optimization.optimizers import base as opt_base
from learned_optimization.outer_trainers import gradient_learner
from learned_optimization import tree_utils
from learned_optimization.research.general_lopt.prefab import LearnedOptimizer

class OptaxWrapper(optax.GradientTransformation):
    def __init__(self, opt: LearnedOptimizer, num_steps: int):
        self.opt = opt
        self.num_steps = num_steps

    def init(self, tx_params: chex.ArrayTree, params: chex.ArrayTree,
              *,
              extra_args: Optional[Mapping[str, Any]] = None) -> chex.ArrayTree:
        del extra_args
        key = jax.random.PRNGKey(0)
        opt_state = self.opt.init(tx_params, params, num_steps=self.num_steps, key=key)
        if dataclasses.is_dataclass(opt_state):
            return self.opt.set_params(opt_state, ())
        else:
            raise NotImplementedError("Only flax dataclasses are supported!")

    def update(self,
            updates: chex.ArrayTree,
            state: chex.ArrayTree,
            params: Optional[chex.ArrayTree] = None,
            meta_params: Optional[chex.ArrayTree] = None,
            *,
            extra_args: Optional[Mapping[str, Any]] = None
        ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        state = self.opt.set_params(state, params)
        key = jax.random.PRNGKey(0)
        next_state = self.opt.update(meta_params, state, updates, key=key, **extra_args)
        step = tree_utils.tree_sub(self.opt.get_params(next_state), params)
        next_state = self.opt.set_params(next_state, ())
        return step, next_state

def parameters_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    extra_bindings=tuple([])
) -> opt_base.Optimizer:
  """Load an optimizer from a checkpoint path, and gin config.

  Args:
    checkpoint_path: Path to `ParameterCheckpoint` saved to disk.
    config_path: Optional path to operative gin config for this checkpoint. If
      not provided, we look in the same folder for a config.gin
    extra_bindings: Optional extra gin bindings to load with this optimizer.

  Returns:
    Optimizer instance created from the learned optimizer + weights.
  """

  if config_path is None:
    config_path = "/".join(checkpoint_path.split("/")[:-1]) + "/config.gin"

  with gin.unlock_config():
    scope = f"opt_from_checkpoint__{str(uuid.uuid4()).replace('-', '_')}"
    with gin.config_scope(None):
      with gin.config_scope(scope):
        if config_path:
          with filesystem.file_open(config_path, "rb") as f:
            content = bytes(f.read()).decode("utf-8")

          # gin writes out multi line sometimes, undo this.
          content = content.replace("\\\n", "")

          def maybe_add_scope(c):
            # filter out train as this overlaps with outer_training.
            if c.startswith("#"):
              return None
            if "=" in c:
              return scope + "/" + c
            return c

          bindings = [maybe_add_scope(c) for c in content.split("\n")]
          bindings = [b for b in bindings if b]
          bindings = bindings + [maybe_add_scope(c) for c in extra_bindings]

          for b in bindings:
            print(b)
          gin.parse_config(bindings, skip_unknown=True)

        configurable = gin.query_parameter(f"{scope}/run_train.lopt")
        if isinstance(configurable, gin.config._UnknownConfigurableReference):  # pylint: disable=protected-access
          raise ValueError("Gin couldn't find the learned optimizer in current"
                           " imports. Did you forget to import the module?")

        with summary.summary_scope("opt_from_checkpoint"):
          lopt = configurable.configurable.wrapped()
          theta = lopt.init(jax.random.PRNGKey(0))
          ckpt = gradient_learner.ParameterCheckpoint(theta, "", 0)
          ckpt = checkpoints.load_state(checkpoint_path, ckpt)
          return ckpt.params  # type: ignore

def get_optax_velo(num_steps):
    # load pretrained optimizer parameters
    _pretrain_root = 'gs://gresearch/learned_optimization/pretrained_lopts/'
    lopt_config_path = 'aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_5620'
    ckpt = parameters_from_checkpoint(os.path.join(_pretrain_root, lopt_config_path, 'params'))
    # load optimizer model
    lopt = HyperV2(lstm_hidden_size=512, param_inits=256, use_bugged_loss_features=False)
    opt = lopt.opt_fn()
    
    return OptaxWrapper(opt, num_steps), ckpt

if __name__ == '__main__':
   lopt, meta_params = get_optax_velo(1)