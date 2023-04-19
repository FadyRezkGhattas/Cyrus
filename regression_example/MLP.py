from typing import Sequence
from flax import linen as nn

class MLP(nn.Module):
    hidden_dims : Sequence[int]
    output_dim : int
        
    @nn.compact
    def __call__(self, x, **kwargs):
        for dims in self.hidden_dims:
            x = nn.Dense(dims)(x)
            x = nn.silu(x)
        x = nn.Dense(self.output_dim)(x)
        return x