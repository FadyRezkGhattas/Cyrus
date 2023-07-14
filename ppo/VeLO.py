# adapted from: https://github.com/google/learned_optimization/blob/main/learned_optimization/optimizers/opt_to_optax.py
# with the differene that opt is accessed from outside the wrapper class
# this allows changing parameters in-place

import dataclasses
from typing import Any, Mapping, Optional, Tuple, NamedTuple

import optax
import chex
from learned_optimization import tree_utils
from learned_optimization.research.general_lopt.prefab import LearnedOptimizer
from learned_optimization.research.general_lopt import pretrained_optimizers

_default_lopt_fn = pretrained_optimizers.aug12_continue_on_bigger_2xbs_200kstep_bigproblem_v2_5620

class OptaxWrapper(optax.GradientTransformation):
    def __init__(self, opt: LearnedOptimizer, num_steps: int):
        self.opt = opt
        self.num_steps = num_steps

    def init(self, params: chex.ArrayTree,
              *,
              extra_args: Optional[Mapping[str, Any]] = None) -> chex.ArrayTree:
        del extra_args
        opt_state = self.opt.init(params, num_steps=self.num_steps)
        if dataclasses.is_dataclass(opt_state):
            return self.opt.set_params(opt_state, ())
        else:
            raise NotImplementedError("Only flax dataclasses are supported!")

    def update(self,
            updates: chex.ArrayTree,
            state: chex.ArrayTree,
            params: Optional[chex.ArrayTree] = None,
            *,
            extra_args: Optional[Mapping[str, Any]] = None
        ) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        if extra_args is None:
            extra_args = {}

        if params is None:
            raise ValueError("Params must not be None!")

        if dataclasses.is_dataclass(state):
            state = self.opt.set_params(state, params)
        else:
            raise NotImplementedError("Only flax dataclasses are supported!")

        next_state = self.opt.update(state, updates, **extra_args)

        step = tree_utils.tree_sub(self.opt.get_params(next_state), params)

        next_state = self.opt.set_params(next_state, ())

        return step, next_state

def get_optax_velo(num_steps,
                    weight_decay=0.0,
                    max_training_steps=200_000,
                    base_lopt_fn=_default_lopt_fn):
    opt = LearnedOptimizer(
      num_steps,
      weight_decay=weight_decay,
      max_training_steps=max_training_steps,
      base_lopt_fn=base_lopt_fn)
    
    return OptaxWrapper(opt, num_steps)