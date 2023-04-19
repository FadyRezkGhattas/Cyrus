import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from TrainerModule import TrainerModule, TrainState
from VeloTrainerModule import VeloTrainerModule
from typing import Any, Sequence
from MLPClassifier import MLPClassifier
from jax import random
import optax
import jax
import optuna

class MLPClassTrainer(VeloTrainerModule):

    def __init__(self,
                 hidden_dims : Sequence[int],
                 num_classes : int,
                 dropout_prob : float,
                 trial : Any = None,
                 **kwargs):
        super().__init__(model_class=MLPClassifier,
                         model_hparams={
                             'hidden_dims': hidden_dims,
                             'num_classes': num_classes,
                             'dropout_prob': dropout_prob
                         },
                         **kwargs)
        self.trial = trial

    def create_functions(self):
        def loss_function(params, batch_stats, rng, batch, train):
            imgs, labels = batch
            rng, dropout_rng = random.split(rng)
            output = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                      imgs,
                                      train=train,
                                      rngs={'dropout': dropout_rng},
                                      mutable=['batch_stats'] if train else False)
            logits, new_model_state = output if train else (output, None)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (rng, new_model_state, acc)

        def train_step(state : TrainState, batch):
            loss_fn = lambda params: loss_function(params, state.batch_stats, state.rng, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, rng, new_model_state, acc = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'], rng=rng, loss=loss)
            metrics = {'loss': loss, 'acc': acc}
            return state, metrics

        def eval_step(state, batch):
            _, (_, _, acc) = loss_function(state.params, state.batch_stats, state.rng, batch, train=False)
            return {'acc': acc}

        return train_step, eval_step

    def run_model_init(self, exmp_input, init_rng):
        imgs, _ = exmp_input
        init_rng, dropout_rng = random.split(init_rng)
        return self.model.init({'params': init_rng, 'dropout': dropout_rng}, x=imgs, train=True)

    def print_tabulate(self, exmp_input):
        imgs, _ = exmp_input
        print(self.model.tabulate(rngs={'params': random.PRNGKey(0), 'dropout': random.PRNGKey(0)}, x=imgs, train=True))

    def on_validation_epoch_end(self, epoch_idx, eval_metrics, val_loader):
        if self.trial:
            self.trial.report(eval_metrics['val/acc'], step=epoch_idx)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()