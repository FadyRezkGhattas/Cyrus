# Standard Libraries
import os
import sys
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import json
import time
from tqdm.auto import tqdm
import numpy as np
from copy import copy
from glob import glob
from collections import defaultdict
import dataclasses as dc

# JAX/Flax
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state
import optax

# PyTorch for data loading
import torch
import torch.utils.data as data

# Logging with Tensorboard and/or Weights and Biases
from torch.utils.tensorboard import SummaryWriter
import wandb

from collections.abc import MutableMapping

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

####
## Entities
####
class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats : Any = None
    # You can further extend the TrainState by any additional part here
    # For example, rng to keep for init, dropout, etc.
    rng : Any = None
    # Save loss as a state because learned optimizers need it as input
    # Strange initialization because mutable jaxlib.xla_extension.ArrayImpl is not allowed (must use default_factory): https://github.com/google/jax/issues/14295
    loss : Any = dc.field(default_factory=lambda: jnp.asarray(0))

####
## Module
####
class TrainerModule:

    def __init__(self,
                 model_class : nn.Module,
                 model_hparams : Dict[str, Any],
                 optimizer_hparams : Dict[str, Any],
                 exmp_input : Any,
                 seed : int = 42,
                 logger_params : Dict[str, Any] = None,
                 enable_progress_bar : bool = True,
                 debug : bool = False,
                 check_val_every_n_epoch : int = 1,
                 **kwargs) -> None:
        """A basic Trainer module summarizing most common training functionalities like logging, model initialization, training loop, etc.

        Args:
            model_class (nn.Module): The class of the model that should be trained
            model_hparams (Dict[str, Any]): A dictionary of all hyperparameters of the model. Is used as input to the model when created.
            optimizer_hparams (Dict[str, Any]): A dictionary of all hyperparameters of the optimizer. Used during initialization of the optimizer.
            exmp_input (Any): Input to the model for initialization and tabulate.
            seed (int, optional): Seed to initialize PRNG. Defaults to 42.
            logger_params (Dict[str, Any], optional): A dictionary containing the specification of the logger. Defaults to None.
            enable_progress_bar (bool, optional): If False, no progress is shown. Defaults to True.
            debug (bool, optional): If true, no jitting is applied. Can be helpful for debugging. Defaults to False.
            check_val_every_n_epoch (int, optional): The frequency with which the model is evaluated on the validation set. Defaults to 1.
        """
        super().__init__()
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.enable_progress_bar = enable_progress_bar
        self.debug = debug
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.exmp_input = exmp_input
        self.optimizer_name = self.optimizer_hparams.pop('optimizer', 'adamw')
        # Set of hyperparameters to save
        self.config = dict({
            'model_class': model_class.__name__,
            'model_hparams': model_hparams,
            'optimizer_hparams': optimizer_hparams,
            'logger_params': logger_params,
            'enable_progress_bar': self.enable_progress_bar,
            'debug': self.debug,
            'check_val_every_n_epoch': check_val_every_n_epoch,
            'seed': self.seed
        })
        self.config.update(kwargs)
        # Set experiment name
        model = self.config["model_class"]
        regularization = 'l2reg' if self.optimizer_hparams['l2reg-weight'] > 0 else 'basic-loss'
        self.run_name = f"{model}__{self.optimizer_name}__{regularization}__{self.seed}"

        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        self.print_tabulate(exmp_input)

        # Init trainer parts
        self.init_logger(logger_params)
        self.create_jitted_functions()
        self.init_model(exmp_input)

    def init_logger(self, logger_params : Optional[Dict] = None):
        """Initializes a logger and creates a logging directory.

        Args:
            logger_params (Optional[Dict], optional): A dictionary containing the specification of the logger. Defaults to None.
        """  
        # Create logger object
        track = logger_params.get('track', False)
        if track:
            wandb.init(
                project=logger_params['wandb_project_name'],
                entity=logger_params['wandb_entity'],
                config=self.config, # Save all hyperparameters
                name=self.run_name
            )
        self.writer = SummaryWriter(f"runs/{self.run_name}")

        # Create metrics folder for logging
        os.makedirs(os.path.join(f"runs/{self.run_name}", 'metrics/'), exist_ok=True)

    def init_model(self, exmp_input : Any):
        """Creates an initial training state with newly generated network paramters.

        Args:
            exmp_input (Any): An input to the model with which the shapes are inferred.
        """
        # Prepare PRNG and input
        model_rng = random.PRNGKey(self.seed)
        model_rng, init_rng = random.split(model_rng)
        exmp_input = [exmp_input] if not isinstance(exmp_input, (list, tuple)) else exmp_input
        # Run model initialization
        variables = self.run_model_init(exmp_input, init_rng)
        # Create default state. Optimizer is initialized later
        self.state = TrainState(step=0, 
                                apply_fn=self.model.apply,
                                params=variables['params'],
                                batch_stats=variables.get('batch_stats'),
                                rng=model_rng,
                                tx=None,
                                opt_state=None)
        
    def run_model_init(self, exmp_input : Any, init_rng : Any) -> Dict:
        """The model initialization call

        Args:
            exmp_input (Any): An input to the model with which the shapes are inferred
            init_rng (Any): A jax.random.PRNGKey.

        Returns:
            Dict: The initialized variable dictionary.
        """
        return self.model.init(init_rng, *exmp_input, train=True)

    def print_tabulate(self, exmp_input : Any):
        """Prints a summary of the Module represented as table.

        Args:
            exmp_input (Any): An input to the model with which the shapes are inferred
        """
        print(self.model.tabulate(random.PRNGKey(0), *exmp_input, train=True))
    
    def init_optimizer(self, num_epochs : int, num_steps_per_epoch : int):
        """Initializes the optimizer and learning rate scheduler.

        Args:
            num_epochs (int): Number of epochs the model will be trained for.
            num_steps_per_epoch (int): Number of training steps per epoch.
        """
        hparams = copy(self.optimizer_hparams)

        # Initialize optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'

        # Initialize learning rate scheduler
        # A cosine decay scheduler is used, but others are also possible
        lr = hparams.pop('lr', 1e-3)
        warmup = hparams.pop('warmup', 0)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup,
            decay_steps=int(num_epochs * num_steps_per_epoch),
            end_value=0.01 * lr
        )

        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip_by_global_norm(hparams.pop('gradient_clip', 1.0))]
        if opt_class == optax.sgd and 'weight_decay' in hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(hparams.pop('weight_decay', 0.0)))
        
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **hparams)
        )

        # Initialize training state
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=self.state.params,
                                       batch_stats=self.state.batch_stats,
                                       tx=optimizer,
                                       rng=self.state.rng)
        
    def create_jitted_functions(self):
        """
        Creates jitted versions of the training and evaluation functions.
        If self.debug is True, not jitting is applied.
        """
        train_step, eval_step = self.create_functions()
        if self.debug: # Skip jitting
            print('Skipping jitting due to debug=True')
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            self.train_step = jax.jit(train_step)
            self.eval_step = jax.jit(eval_step)

    def create_functions(self) -> Tuple[Callable[[TrainState, Any], Tuple[TrainState,Dict]],
                                        Callable[[TrainState,Any], Tuple [TrainState,Dict]]]:
        """
        Creates and returns functions for the training and evaluation step. The
        functions take as input the training state and a batch from the train/
        val/test loader. Both functions are expected to return a dictionary of
        logging metrics, and the training function a new train state. This
        function needs to be overwritten by a subclass. The train_step and
        eval_step functions here are examples for the signature of the functions.
        """
        def train_step(state : TrainState, 
                       batch : Any):
            metrics = {}
            return state, metrics
        def eval_step(state : TrainState, 
                      batch : Any):
            metrics = {}
            return metrics
        raise NotImplementedError
    
    def train_model(self, train_loader : Iterator, val_loader : Iterator, test_loader : Optional[Iterator] = None, num_epochs : int = 500) -> Dict[str, Any]:
        """Starts a training loop for the given number of epochs/

        Args:
            train_loader (Iterator): Data loader for the training set.
            val_loader (Iterator): Data loader for the validation set.
            test_loader (Optional[Iterator], optional): If given, best model will be evaluated on the test set.. Defaults to None.
            num_epochs (int, optional): Number of epochs for which to train the model. Defaults to 500.

        Returns:
            Dict[str, Any]: A dictionary of the train, validation and evt. test metrics for the best model on the validation set
        """
        # Create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))

        # Prepare training loop
        self.on_training_start()
        best_eval_metrics = None

        # Training loop
        for epoch_idx in self.tracker(range(1, num_epochs+1), desc='Epochs'):
            train_metrics = self.train_epoch(train_loader)
            self.log(train_metrics, step=epoch_idx)
            self.on_training_epoch_end(epoch_idx)
            # Validation every N epochs
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader, log_prefix='val/')
                self.on_validation_epoch_end(epoch_idx, eval_metrics, val_loader)
                self.log(eval_metrics, step=epoch_idx)
                self.save_metrics(f'eval_epoch_{str(epoch_idx).zfill(3)}', eval_metrics)
                # Save best model
                if self.is_new_model_better(eval_metrics, best_eval_metrics):
                    best_eval_metrics = eval_metrics
                    best_eval_metrics.update(train_metrics)
                    self.save_model(step=epoch_idx)
                    self.save_metrics('best_eval', eval_metrics)
        
        # Test best model if possible
        if test_loader is not None:
            self.load_model()
            test_metrics = self.eval_model(test_loader, log_prefix='test/')
            self.log(test_metrics, step=epoch_idx)
            self.save_metrics('test', test_metrics)
            best_eval_metrics.update(test_metrics)
        
        return best_eval_metrics
    
    def train_epoch(self, train_loader : Iterator) -> Dict[str, Any]:
        """Trains a model for one epoch.

        Args:
            train_loader (Iterator): Data loader of the training set.

        Returns:
            Dict[str, Any]: A dictionary of the average training metrics over all batches for logging.
        """
        # Train model for one epoch, and log avg loss and accuracy
        metrics = defaultdict(float)
        num_train_steps = len(train_loader)
        start_time = time.time()
        for batch in self.tracker(train_loader, desc='Training', leave=False):
            self.state, step_metrics = self.train_step(self.state, batch)
            for key in step_metrics:
                metrics['train/' + key] += step_metrics[key] / num_train_steps
        metrics = {key: metrics[key].item() for key in metrics}
        metrics['epoch_time'] = time.time() - start_time
        return metrics
    
    def eval_model(self, data_loader : Iterator, log_prefix : Optional[str] = '') -> Dict[str, Any]:
        """Evaluates the model on a dataset.

        Args:
            data_loader (Iterator): Data loader of the dataset to evaluate on.
            log_prefix (Optional[str], optional): Prefix to add to all metrics (e.g. 'val/' or 'test/'). Defaults to ''.

        Returns:
            Dict[str, Any]: A dictionary of the evaluation metrics, averaged over data points in the dataset.
        """
        metrics = defaultdict(float)
        num_elements = 0
        for batch in data_loader:
            step_metrics = self.eval_step(self.state, batch)
            batch_size = batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
            for key in step_metrics:
                metrics[key] += step_metrics[key] * batch_size
            num_elements += batch_size
        metrics = {(log_prefix + key): (metrics[key] / num_elements).item() for key in metrics}
        return metrics
    
    def is_new_model_better(self, new_metrics : Dict[str, Any], old_metrics : Dict[str, Any]) -> bool:
        """
        Compares two sets of evaluation metrics to decide whether the new model is better than
        the previous one or not.

        Args:
            new_metrics (Dict[str, Any]): A dictionary of the evaluation metrics of the new model.
            old_metrics (Dict[str, Any]): A dictionary of the evaluation metrics of the previously
            best model, i.e. the one to compare to.

        Returns:
            bool: True if the new model is better than the old one, and False otherwise.
        """
        #TODO: this comparison is sensitive to how the metrics are named
        # for now, maybe override this in inheriting classes?
        if old_metrics is None:
            return True
        for key, is_larger in [('val/val_metric', False), ('val/loss', False)]:
            if key in new_metrics:
                if is_larger:
                    return new_metrics[key] > old_metrics[key]
                else:
                    return new_metrics[key] < old_metrics[key]
        assert False, f'No known metrics to log on: {new_metrics}'

    def tracker(self, iterator : Iterator, **kwargs) -> Iterator:
        """Wraps an iterator in a progress bar tracker (tqdm) if the progress bar is enabled.

        Args:
            iterator (Iterator): Iterator to wrap in tqdm.
            **kwargs: Additional arguments to tqdm.

        Yields:
            Iterator: Wrapped iterator if progress bar is enabled, otherwise some iterator as input.
        """
        if self.enable_progress_bar:
            return tqdm(iterator, **kwargs)
        else:
            return iterator
        
    def save_metrics(self, filename : str, metrics : Dict[str, Any]):
        """
        Saves a dictionary of metrics to file. Can be used as textual
        representation of the validation performance for checking in the terminal.

        Args:
            filename (str): Name of the metrics file without folders and postfix.
            metrics (Dict[str, Any]): A dictionary of metrics to save in the file.
        """
        with open(os.path.join(f'runs/{self.run_name}/metrics/{filename}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        wandb.save(f'runs/{self.run_name}/metrics/{filename}.json')

    def log(self, metrics : Dict[str, Any], step : int):
        wandb.log(metrics, step)

    def on_training_start(self):
        """
        Method called before training is started. Can be used for additional
        initialization operations etc.
        """
        pass

    def on_training_epoch_end(self, epoch_idx : int):        
        """
        Method called at the end of each training epoch. Can be used for additional
        logging or similar.
        
        Args:
          epoch_idx (int): Index of the training epoch that has finished.
        """
        pass

    def on_validation_epoch_end(self, 
                                epoch_idx : int, 
                                eval_metrics : Dict[str, Any], 
                                val_loader : Iterator):
        """
        Method called at the end of each validation epoch. Can be used for additional
        logging and evaluation.
        
        Args:
          epoch_idx (int): Index of the training epoch at which validation was performed.
          eval_metrics (Dict[str, Any]): A dictionary of the validation metrics. New metrics added to
            this dictionary will be logged as well.
          val_loader (Iterator): Data loader of the validation set, to support additional 
            evaluation.
        """
        pass

    def save_model(self, 
                   step : int = 0):
        """
        Saves current training state at certain training iteration. Only the model
        parameters and batch statistics are saved to reduce memory footprint. To
        support the training to be continued from a checkpoint, this method can be
        extended to include the optimizer state as well.
        
        Args:
          step (int): Index of the step to save the model at, e.g. epoch.
        """
        checkpoints.save_checkpoint(ckpt_dir=f"runs/{self.run_name}",
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                    overwrite=True)
        
    def load_model(self):
        """
        Loads model parameters and batch statistics from the logging directory.
        """
        state_dict = checkpoints.restore_checkpoint(ckpt_dir=f"runs/{self.run_name}", target=None)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       # Optimizer will be overwritten when training starts
                                       tx=self.state.tx if self.state.tx else optax.sgd(0.1),
                                       rng=self.state.rng
                                      )
    
    def bind_model(self):
        """
        Returns a model with parameters bound to it. Enables an easier inference
        access.
        
        Returns:
          The model with parameters and evt. batch statistics bound to it.
        """
        params = {'params': self.state.params}
        if self.state.batch_stats:
            params['batch_stats'] = self.state.batch_stats
        return self.model.bind(params)
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint : str, exmp_input : Any) -> Any:
        """
        Creates a Trainer object with same hyperparameters and loaded model from
        a checkpoint directory.

        Args:
            checkpoint (str): Folder in which the checkpoint and hyperparameter file is stored.
            exmp_input (Any): An input to the model for shape inference.

        Returns:
            A trainer object with model loaded from the checkpoint folder
        """
        hparams_file = os.path.join(checkpoint, 'hparams.json')
        assert os.path.isfile(hparams_file), 'Could not find hparams file'
        with open(hparams_file, 'r') as f:
            hparams = json.load(f)
        hparams.pop('model_class')
        hparams.update(hparams.pop('model_hparams'))
        if not hparams['logger_params']:
            hparams['logger_params'] = dict()
        hparams['logger_params']['log_dir'] = checkpoint
        trainer = cls(exmp_input=exmp_input,
                      **hparams)
        trainer.load_model()
        return trainer

####
## Utilities
####
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def create_data_loaders(*datasets : Sequence[data.Dataset], 
                        train : Union[bool, Sequence[bool]] = True, 
                        batch_size : int = 128, 
                        num_workers : int = 4, 
                        seed : int = 42):
    """
    Creates data loaders used in JAX for a set of datasets.
    
    Args:
      datasets: Datasets for which data loaders are created.
      train: Sequence indicating which datasets are used for 
        training and which not. If single bool, the same value
        is used for all datasets.
      batch_size: Batch size to use in the data loaders.
      num_workers: Number of workers for each dataset.
      seed: Seed to initialize the workers and shuffling with.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=is_train,
                                 drop_last=is_train,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers,
                                 persistent_workers=is_train,
                                 generator=torch.Generator().manual_seed(seed))
        loaders.append(loader)
    return loaders