from src.utils import LOG, CONSOLE
from torch import nn, optim
import argparse
from typing import Iterable, Optional


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
