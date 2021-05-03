from src.utils import LOG, CONSOLE
from torch import nn, optim
import torch
import argparse
from typing import Iterable, Optional, Union, Dict
from logging import RootLogger
from pathlib import Path


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


def find_optimal_model(path: Union[str, Path]) -> Dict:
    if isinstance(path, str):
        path = Path(path)

    # f'stage-{stage:02d}-epoch-{i:03d}-loss-{val_loss:.6f}.tar'
    best_val = 9999
    selected = None
    for p in path.absolute().glob('stage-*-epoch-*-loss-*.tar'):
        _, stage, _, epoch, _, val_loss = p.as_posix().split('/')[-1][:-4].split('-')
        stage, epoch, val_loss = int(stage), int(epoch), float(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            selected = p

    return torch.load(selected.as_posix())


class EarlyStopping:
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
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        self.prefix = prefix
        self.print = logger.info if logger is not None else print
        return

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            self.print(self.prefix + f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.print(self.prefix + 'Early stopping!')
                self.early_stop = True
        return

    # TODO add state dict
