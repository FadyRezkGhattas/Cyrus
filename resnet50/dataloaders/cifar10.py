import sys
import os.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)
import numpy as np

# Torch
import torch
from torchvision.datasets import CIFAR10
import torch.utils.data as data

# Cyrus
from TrainerModule import create_data_loaders

DATASET_PATH = './data/'
def get_data(batch_size):
    # Transformations applied on each image => bring them into a numpy array
    DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
    DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])
    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=image_to_numpy, download=True)
    train_set, val_set = data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=image_to_numpy, download=True)

    return create_data_loaders(train_set, val_set, test_set,
                                train=[True, False, False],
                                batch_size=batch_size)
    