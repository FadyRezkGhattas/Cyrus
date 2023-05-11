import sys
import os.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.utils.data as data
import torch
from TrainerModule import create_data_loaders  
from MLPClassTrainer import MLPClassTrainer

DATASET_PATH = './data/'
CHECKPOINT_PATH = './saved_models/'

# Transformations applied on each image => bring them into a numpy array
DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    return img


test_transform = image_to_numpy
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      image_to_numpy])
# Loading the training dataset. We need to split it into a training and validation part
# We need to do a little trick because the validation set should not use the augmentation.
train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
train_set, _ = data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
_, val_set = data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

# Loading the test set
test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

train_loader, val_loader, test_loader = create_data_loaders(train_set, val_set, test_set,
                                                            train=[True, False, False],
                                                            batch_size=256)

trainer = MLPClassTrainer(hidden_dims=[512, 512],
                          num_classes=10,
                          dropout_prob=0.4,
                          optimizer_hparams={
                              'weight_decay': 2e-4,
                              'lr': 1e-3
                          },
                          logger_params={
                                'track': True,
                                'wandb_project_name': 'test_project',
                                'wandb_entity': 'fastautomate'
                            },
                          exmp_input=next(iter(train_loader)),
                          check_val_every_n_epoch=5)

metrics = trainer.train_model(train_loader,
                              val_loader,
                              test_loader=test_loader,
                              num_epochs=50)

print(f'Validation accuracy: {metrics["val/acc"]:4.2%}')
print(f'Test accuracy: {metrics["test/acc"]:4.2%}')

