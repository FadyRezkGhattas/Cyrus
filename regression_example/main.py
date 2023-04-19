import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from RegressionDataset import RegressionDataset
from TrainerModule import create_data_loaders  
from Trainer import MLPRegressTrainer

train_set = RegressionDataset(num_points=1000, seed=42)
val_set = RegressionDataset(num_points=200, seed=43)
test_set = RegressionDataset(num_points=500, seed=44)
train_loader, val_loader, test_loader = create_data_loaders(train_set, val_set, test_set,
                                                            train=[True, False, False],
                                                            batch_size=64)

trainer = MLPRegressTrainer(hidden_dims=[128, 128],
                            output_dim=1,
                            optimizer_hparams={'lr': 4e-3},
                            logger_params={'base_log_dir': './saved_models/'},
                            exmp_input=next(iter(train_loader))[0:1],
                            check_val_every_n_epoch=5)

metrics = trainer.train_model(train_loader, 
                              val_loader, 
                              test_loader=test_loader, 
                              num_epochs=50)