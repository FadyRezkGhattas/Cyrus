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
  x = torch.stack(x, dim=1)
  y = torch.stack(y, dim=1)
  return x.numpy(), y.numpy()

def _ScikitDatasetToPytorch(n_samples=1000, n_features=10, random_state=42):
    X, Y = make_regression(n_samples=n_samples, n_features=n_features, random_state=random_state)
    X = torch.from_numpy(X).to(torch.float32)
    Y = torch.from_numpy(Y).unsqueeze(1).to(torch.float32)
    return TensorDataset(X, Y)

def ScikitDatastToDataLoader(batch_size, n_samples=1000, n_features=10, random_state=42):
    dataset = _ScikitDatasetToPytorch(n_samples, n_features, random_state)
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