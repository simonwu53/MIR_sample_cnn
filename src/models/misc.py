from src.utils import LOG, CONSOLE
import numpy as np
from torch import nn, optim
import torch
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Iterable, Optional, Union, Dict, Tuple
from logging import RootLogger
from pathlib import Path
from abc import ABC, abstractmethod
from multiprocessing import Pool


def _get_activation(name: str) -> Optional[nn.Module]:
    if name.lower() == 'relu':
        return nn.ReLU()
    # TODO: test other activations
    else:
        LOG.error(f'Unrecognized activation function: [bold red]{name}[/].')
        raise ValueError(f'Unrecognized activation function: {name}.')


def get_optimizer(params: Iterable,
                  args: argparse.Namespace) -> Optional[optim.Optimizer]:
    name = args.optim_type.lower()
    if name == 'sgd':
        LOG.info(f"SGD Optimizer <lr={args.lr}, momentum={args.momentum}, nesterov=True>")
        return optim.SGD(params=params, lr=args.lr, momentum=args.momentum, nesterov=True)
    elif name == 'adam':
        LOG.info(f"Adam Optimizer <lr={args.lr}, betas={args.betas}, eps={args.eps}>")
        return optim.Adam(params=params, lr=args.lr, betas=args.betas, eps=args.eps)
    elif name == 'adamw':
        LOG.info(f"Adam Optimizer <lr={args.lr}, betas={args.betas}, eps={args.eps}, weight_decay={args.weight_decay}>")
        return optim.AdamW(params=params, lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    else:
        LOG.error(f"Unsupported optimizer: [bold red]{name}[/].")
        raise ValueError(f"Unsupported optimizer: {name}.")


def get_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == 'bce':
        return nn.BCELoss()
    elif name == 'bcelogits':
        return nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError


def count_params(model: nn.Module, only_trainable: bool = True) -> int:
    if not only_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_print_hock(module, inputs, outputs):
    if not getattr(module, 'name', 0):
        CONSOLE.print(f'{module.__class__.__name__}')
    else:
        CONSOLE.print(f'{module.name}({module.__class__.__name__})')
    CONSOLE.print(f'input shape: [bold cyan]{inputs[0].shape}[/]')
    CONSOLE.print(f'output shape: [bold cyan]{outputs.shape}[/]')
    CONSOLE.print(f'------------------------------------------')


def find_optimal_model(path: Union[str, Path],
                       reset_epoch: Optional[int] = None,
                       ) -> Dict:
    if isinstance(path, str):
        path = Path(path)

    # f'epoch-{i:03d}-loss-{val_loss:.6f}.tar'
    best_val = 9999
    selected = None
    for p in path.absolute().glob('*epoch-*-loss-*.tar'):
        _, epoch, _, val_loss = p.as_posix().split('/')[-1][:-4].split('-')
        epoch, val_loss = int(epoch), float(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            selected = p

    state_dict = load_ckpt(selected.as_posix(),
                           reset_epoch=reset_epoch)

    return state_dict


def load_ckpt(path: str,
              reset_epoch: Optional[int] = None,
              no_scheduler: bool = False,
              no_optimizer: bool = False,
              no_loss_fn: bool = False,
              map_values: Optional[Dict] = None) -> Dict:
    # load state dict
    state_dict = torch.load(path)
    excluded_keys = []

    # modify keys
    if reset_epoch is not None:
        state_dict['epoch'] = reset_epoch

    if no_scheduler:
        excluded_keys.extend([k for k in state_dict if 'scheduler' in k])
    if no_optimizer:
        excluded_keys.append('optim')
    if no_loss_fn:
        excluded_keys.append('loss_fn')

    for k in excluded_keys:
        state_dict.pop(k, 0)

    # update other keys
    if map_values is not None:
        state_dict.update(map_values)

    return state_dict


def show_ckpt(path: str, out: Optional[Dict] = None, printer: Optional = None) -> None:
    # load state dict
    state_dict = torch.load(path)

    schedulers = [k for k in state_dict if 'scheduler' in k]

    printer = print if printer is None else printer
    printer(f"Model info: \n"
            f"{[k for k in state_dict['model']]}\n")
    printer(f"Optimizer info: \n"
            f"{[k for k in state_dict['optim']['param_groups']]}\n")
    printer(f"Loss Func info: \n"
            f"{[k for k in state_dict['loss_fn']]}\n")
    printer(f"Scheduler info: \n"
            f"{[state_dict[k] for k in schedulers]}\n")

    printer(f"Other status: \n"
            f"epoch: {state_dict['epoch']} \n"
            f"global i: {state_dict['global_i']} \n"
            f"train loss: {state_dict['loss']} \n"
            f"valid loss: {state_dict['val_loss']} \n"
            f"path: {state_dict['p_out'].as_posix()}")

    if isinstance(out, dict):
        out.update(state_dict)
    return


def apply_lr(optimizer, lr: Union[float, Iterable[float]]):
    if isinstance(lr, (list, tuple)):
        assert len(lr) == len(optimizer.param_groups), \
            "lr quantity should be the same with optimizer groups, " \
            "otherwise specify one float value instead"
    for i, param_group in enumerate(optimizer.param_groups):
        if isinstance(lr, float):
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr[i]
    return


def apply_state_dict(state_dict, model=None, optimizer=None, loss_fn=None, scheduler=None):
    if model is not None:
        k, v = list(model.items())[0]
        v.load_state_dict(state_dict[k])
    if optimizer is not None:
        k, v = list(optimizer.items())[0]
        v.load_state_dict(state_dict[k])
    if loss_fn is not None:
        k, v = list(loss_fn.items())[0]
        v.load_state_dict(state_dict[k])
    if scheduler is not None:
        k, v = list(scheduler.items())[0]
        v.load_state_dict(state_dict[k])
    return


class Scheduler(ABC):
    """
    Modify learning rate when the loss does not improve after certain epochs. (base class)
    """
    def __init__(self, patience: int = 3,
                 min_delta: int = 0,
                 verbose: bool = False,
                 prefix: str = '',
                 logger: RootLogger = None):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        :param verbose: if True, print in console when metric is not improving
        :param prefix: prefix text for console printing
        :param logger: logging object to print information instead of bulletin print
        """
        if patience == 0:
            self.enabled = False
        else:
            self.enabled = True
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.verbose = verbose
        self.prefix = prefix
        self.print = logger.info if logger is not None else print
        return

    def step(self, val_loss: float):
        if self.enabled:
            if self.best_loss is None:
                self.best_loss = val_loss
                self.counter = 0
            elif self.best_loss - val_loss >= self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            elif self.best_loss - val_loss < self.min_delta:
                self.counter += 1
                # do policy
                self._policy()
        return

    @abstractmethod
    def _policy(self):
        pass

    def _display(self, msg):
        if self.verbose:
            self.print(msg)
        return

    def state_dict(self) -> Dict:
        return {
            'enabled': self.enabled,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'counter': self.counter,
            'best_loss': self.best_loss,
            'verbose': self.verbose,
            'prefix': self.prefix
        }

    def load_state_dict(self, state_dict: Dict):
        self.enabled = state_dict.get('enabled', False)
        self.patience = state_dict.get('patience', 5)
        self.min_delta = state_dict.get('min_delta', 0)
        self.counter = state_dict.get('counter', 0)
        self.best_loss = state_dict.get('best_loss', None)
        self.verbose = state_dict.get('verbose', False)
        self.prefix = state_dict.get('prefix', '')
        return


class EarlyStopping(Scheduler):
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience: int = 5,
                 min_delta: int = 0,
                 verbose: bool = False,
                 prefix: str = '',
                 logger: RootLogger = None):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        :param verbose: if True, print in console when metric is not improving
        :param prefix: prefix text for console printing
        :param logger: logging object to print information instead of bulletin print
        """
        super(EarlyStopping, self).__init__(patience=patience,
                                            min_delta=min_delta,
                                            verbose=verbose,
                                            prefix=prefix,
                                            logger=logger)
        self.early_stop = False
        return

    def _policy(self):
        self._display(self.prefix + f"Early stopping counter {self.counter} of {self.patience}")

        if self.counter >= self.patience:
            self._display(self.prefix + 'Early stopping triggered')
            self.early_stop = True
        return

    def state_dict(self) -> Dict:
        state_dict = {'early_stop': self.early_stop}
        state_dict.update(super(EarlyStopping, self).state_dict())
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        super(EarlyStopping, self).load_state_dict(state_dict)
        self.early_stop = state_dict.get('early_stop')
        return


class ReduceLROnPlateau(Scheduler):
    """
    Reduce learning rate when the loss does not improve after certain epochs.
    """
    def __init__(self, optimizer,
                 factor: float,
                 patience: int = 3,
                 min_lr: float = 0,
                 min_delta: int = 0,
                 verbose: bool = False,
                 prefix: str = '',
                 logger: RootLogger = None):
        """
        :param optimizer: optimizer for adjusting lr
        :param factor: reduce factor, new_lr = lr * factor
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_lr: minimum lr that scheduler will not reduce lr below this value
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        :param verbose: if True, print in console when metric is not improving
        :param prefix: prefix text for console printing
        :param logger: logging object to print information instead of bulletin print
        """
        super(ReduceLROnPlateau, self).__init__(patience=patience,
                                                min_delta=min_delta,
                                                verbose=verbose,
                                                prefix=prefix,
                                                logger=logger)
        self.optim = optimizer
        self.factor = factor
        self.min_lr = min_lr
        return

    def _policy(self):
        self._display(self.prefix + f"Plateau counter {self.counter} of {self.patience}")

        if self.counter >= self.patience:
            # reduce lr
            for i, g in enumerate(self.optim.param_groups):
                new_lr = g['lr'] * self.factor

                if new_lr >= self.min_lr:
                    g['lr'] = new_lr
                    self._display(self.prefix + f'Optimizer group {i}, lr reduced to {new_lr}')
                else:
                    self._display(self.prefix + f'Optimizer group {i}, lr reached minimum threshold')

            # reset counter
            self.counter = 0
        return

    def state_dict(self) -> Dict:
        state_dict = {
            'factor': self.factor,
            'min_lr': self.min_lr
        }
        state_dict.update(super(ReduceLROnPlateau, self).state_dict())
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        super(ReduceLROnPlateau, self).load_state_dict(state_dict)
        self.factor = state_dict.get('factor')
        self.min_lr = state_dict.get('min_lr')
        return


class AUCMetric:
    def __init__(self, n_tags: int = 50, cache_size: int = 50000, n_workers: int = 12):
        self.cache_size = cache_size
        self.n_tags = n_tags
        self.cache_y_true = np.zeros((cache_size, n_tags), np.float32)
        self.cache_y_pred = np.zeros((cache_size, n_tags), np.float32)
        self.idx_y_true = 0                            # index pointer for y_true cache
        self.idx_y_pred = 0                            # index pointer for y_pred cache
        self.running_auc_col = 0                       # running roc_auc_score for tags (columns)
        self.running_auc_row = 0                       # running roc_auc_score for songs (rows)
        self.running_ap_col = 0                        # running average precision for tags (columns)
        self.running_ap_row = 0                        # running average precision for songs (rows)
        self.counter = 0
        self.n_workers = n_workers                     # for faster roc_auc ap calculations
        return

    def step(self, y_true: np.ndarray, y_pred: np.ndarray):
        bs = y_true.shape[0]
        # clear cache data
        if bs + self.idx_y_true > self.cache_size:
            self._do_calculation()

        # record data only
        self.cache_y_true[self.idx_y_true:self.idx_y_true+bs] = y_true
        self.cache_y_pred[self.idx_y_pred:self.idx_y_pred+bs] = y_pred
        self.idx_y_true += bs
        self.idx_y_pred += bs
        return

    def _do_calculation(self):
        # prepare data for calculation
        y_true = self.cache_y_true[:self.idx_y_true]
        y_pred = self.cache_y_pred[:self.idx_y_pred]
        avg = self.idx_y_true // self.n_workers

        # multiprocessing for auc/ap scores
        LOG.info("waiting for calculating auc/ap scores...")
        with Pool(processes=self.n_workers) as pool:
            scores = pool.starmap(self._calculation_job, [(y_true[avg*i:avg*(i+1)], y_pred[avg*i:avg*(i+1)])
                                                          if i != self.n_workers-1 else (y_true[avg*i:], y_pred[avg*i:])
                                                          for i in range(self.n_workers)])
            scores = np.array(scores, dtype=np.float32)

        # apply results
        auc_col, auc_row, ap_col, ap_row = scores.mean(axis=0)
        self.running_auc_col += auc_col
        self.running_auc_row += auc_row
        self.running_ap_col += ap_col
        self.running_ap_row += ap_row

        # count step
        self.counter += 1

        # reset cache and idx pointer
        self.reset_cache()
        return

    @staticmethod
    def _calculation_job(y_true: np.ndarray, y_pred: np.ndarray):
        # column-wise metrics
        m_col = ~(y_true.sum(axis=0) == 0)             # mask to exclude all zero columns in y_pred
        y_true_col = y_true[:, m_col]
        y_pred_col = y_pred[:, m_col]
        auc_col = roc_auc_score(y_true_col, y_pred_col, average='macro')
        ap_col = average_precision_score(y_true_col, y_pred_col, average='macro')

        # row-wise metrics
        m_row = ~(y_true.sum(axis=1) == 0)             # mask to exclude all zero rows in y_pred
        y_true_row = y_true[m_row, :].transpose()      # sklearn calculate metrics for columns, need to transpose first
        y_pred_row = y_pred[m_row, :].transpose()
        auc_row = roc_auc_score(y_true_row, y_pred_row, average='macro')
        ap_row = average_precision_score(y_true_row, y_pred_row, average='macro')
        return auc_col, auc_row, ap_col, ap_row

    @property
    def auc_ap_score(self) -> Union[Tuple[float, float, float, float]]:
        # if no records yet
        if self.counter == 0 and self.idx_y_true == 0:
            return 0, 0, 0, 0

        # calculate cache data
        if self.idx_y_true != 0:
            self._do_calculation()

        # (AUC tag, AUC sample, AP tag, AP sample)
        return self.running_auc_col/self.counter, \
            self.running_auc_row/self.counter, \
            self.running_ap_col / self.counter, \
            self.running_ap_row / self.counter

    def reset_cache(self):
        self.cache_y_true = np.zeros((self.cache_size, self.n_tags), np.float32)
        self.cache_y_pred = np.zeros((self.cache_size, self.n_tags), np.float32)
        self.idx_y_true = 0
        self.idx_y_pred = 0
        return

    def state_dict(self):
        return {
            'auc_col': self.running_auc_col,
            'auc_row': self.running_auc_row,
            'ap_col': self.running_ap_col,
            'ap_row': self.running_ap_row,
            'counter': self.counter,
            'n_tags': self.n_tags,
            'cache_size': self.cache_size,
            'cache_y_true': self.cache_y_true,
            'cache_y_pred': self.cache_y_pred,
            'idx_y_true': self.idx_y_true,
            'idx_y_pred': self.idx_y_pred,
            'n_workers': self.n_workers
        }

    def load_state_dict(self, state_dict: Dict):
        self.running_auc_col = state_dict.get('auc_col', 0)
        self.running_auc_row = state_dict.get('auc_row', 0)
        self.running_ap_col = state_dict.get('ap_col', 0)
        self.running_ap_row = state_dict.get('ap_row', 0)
        self.counter = state_dict.get('counter', 0)
        self.n_tags = state_dict.get('n_tags', 50)
        self.cache_size = state_dict.get('cache_size', 5000)
        self.cache_y_true = state_dict.get('cache_y_true')
        self.cache_y_pred = state_dict.get('cache_y_pred')
        self.idx_y_true = state_dict.get('idx_y_true', 0)
        self.idx_y_pred = state_dict.get('idx_y_pred', 0)
        self.n_workers = state_dict.get('n_workers', 0)
        return
