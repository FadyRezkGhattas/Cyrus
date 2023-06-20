from typing import Any
from flax import linen as nn

class LinearModel(nn.Module):
    number_out_features : int

    @nn.compact
    def __call__(self, x) -> Any:
        return nn.Dense(self.number_out_features)(x)