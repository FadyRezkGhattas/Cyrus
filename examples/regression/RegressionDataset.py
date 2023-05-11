import numpy as np
import torch.utils.data as data

def target_function(x):
    return np.sin(x * 3.0)

class RegressionDataset(data.Dataset):
    
    def __init__(self, num_points, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(low=-2.0, high=2.0, size=num_points)
        self.y = target_function(self.x)
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx:idx+1], self.y[idx:idx+1]