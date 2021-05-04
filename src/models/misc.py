from src.utils import LOG, CONSOLE
from torch import nn, optim
import torch
import argparse
from typing import Iterable, Optional, Union, Dict
from logging import RootLogger
from pathlib import Path
from abc import ABC, abstractmethod


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
    for p in path.absolute().glob('epoch-*-loss-*.tar'):
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
            elif self.best_loss - val_loss > self.min_delta:
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
        if self.verbose:
            self.print(self.prefix + f"Early stopping counter {self.counter} of {self.patience}")

        if self.counter >= self.patience:
            if self.verbose:
                self.print(self.prefix + 'Early stopping triggered.')
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
        super(ReduceLROnPlateau, self).__init__(patience=patience,
                                                min_delta=min_delta,
                                                verbose=verbose,
                                                prefix=prefix,
                                                logger=logger)
        self.optim = optimizer
        self.factor = factor
        return

    def _policy(self):
        if self.verbose:
            self.print(self.prefix + f"Plateau counter {self.counter} of {self.patience}")

        if self.counter >= self.patience:
            lr_group = []
            # reduce lr
            for g in self.optim.param_groups:
                new_lr = g['lr'] * self.factor
                g['lr'] = new_lr
                lr_group.append(new_lr)

            if self.verbose:
                self.print(self.prefix + f'Optimizer LR reduced to {lr_group}!')
        return

    def state_dict(self) -> Dict:
        state_dict = {'factor': self.factor}
        state_dict.update(super(ReduceLROnPlateau, self).state_dict())
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        super(ReduceLROnPlateau, self).load_state_dict(state_dict)
        self.factor = state_dict.get('factor')
        return
