import sys
import os.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

import torch
from torch.utils.data import TensorDataset
from TrainerModule import create_data_loaders
from torch.utils.data import Dataset
from sklearn.datasets import make_regression
import jax.numpy as jnp

def numpy_scikit_collate(batch):
  """Collate function that converts a TensorDataset to a NumPy array."""
  x, y = zip(*batch)
  x = torch.stack(x)
  y = torch.stack(y)
  return x.numpy(), y.numpy()

def ScikitDatastToDataLoader(X, Y, batch_size, regression=True):
    if regression:
        X, Y = torch.from_numpy(X).to(torch.float32), torch.from_numpy(Y).unsqueeze(1).to(torch.float32)
    else:
        X, Y = torch.from_numpy(X).to(torch.float32), torch.from_numpy(Y).to(torch.int16)
    dataset = TensorDataset(X, Y)
    return create_data_loaders(dataset, train=[True], batch_size=batch_size, collate_fn=numpy_scikit_collate)[0]

class RegressionDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=10, random_state=42):
        self.n_samples = 1000
        self.n_features=10
        self.random_state=42
        # Generate regression data
        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=random_state)
        # Convert data to PyTorch tensors
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(y).unsqueeze(1).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]