from src.utils import LOG, CONSOLE
from torch import nn


def get_activation(name: str) -> nn.Module:
    if name.lower() == 'relu':
        return nn.ReLU()
    # TODO: test other activations
    else:
        LOG.warning(f'Unrecognized activation function: [bold yellow]{name}[/], fallback to default [bold yellow]ReLU[/].')
        return nn.ReLU()


def count_params(model: nn.Module, only_trainable: bool = True) -> int:
    if not only_trainable:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
