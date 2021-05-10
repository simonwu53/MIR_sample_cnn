# configs
try:
    from src.utils import LOG, CONSOLE
    from src.models.misc import count_params, layer_print_hock
except ModuleNotFoundError as e:
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent.absolute().as_posix())
    from src.utils import LOG, CONSOLE, traceback_install
    from src.models.misc import count_params, layer_print_hock
    traceback_install(console=CONSOLE, show_locals=True)
# libs
import torch
from torch import nn
from torch import Tensor
import argparse
# typing
from typing import Optional, List, Type


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    """1x1 Convolution"""
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


def conv1x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv1d:
    """1x3 Convolution"""
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class Bottleneck(nn.Module):
    """             k   conv    in_C      out_C       stride
    Consist of (1): 1x1 conv1d, channels, channels,   stride 1
               (2): 1x3 conv1d, channels, channels,   stride 3 (downscale)
               (3): 1x1 conv1d, channels, channels*4, stride 1
    """
    def __init__(self,
                 in_channels: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 expansion: int = 2):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channels, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv1x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes*expansion)
        self.bn3 = nn.BatchNorm1d(planes*expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        return

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ExpSampleCNN(nn.Module):
    """
    channels: in_channels, 64, 128, 256, 512, ...
    """
    def __init__(self,
                 block: Type[Bottleneck],
                 layers: List[int],
                 channels: List[int] = (64, 128, 256, 512),
                 in_channels: int = 1,
                 expansion: int = 2,
                 n_class: int = 50,
                 zero_init_residual: bool = False,
                 name: Optional[str] = None,
                 robust: bool = False):
        super(ExpSampleCNN, self).__init__()
        assert len(layers) == len(channels), "layers and channels length mismatch!"
        self.n_modules = len(layers)
        self.n_layers = sum(layers) * 3 + 2
        self._norm_layer = nn.BatchNorm1d
        self.in_planes = 64

        # input head
        self.conv1 = nn.Conv1d(in_channels, self.in_planes, kernel_size=3, stride=3, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.in_planes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)

        # layers
        for i in range(len(layers)):
            if i == 0:
                setattr(self, f'layer{i+1}', self._make_layer(block, channels[i], layers[i], stride=1, expansion=expansion))
            else:
                setattr(self, f'layer{i+1}', self._make_layer(block, channels[i], layers[i], stride=3, expansion=expansion))

        # fc
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1] * expansion, n_class)

        # init params
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch
            # so that the residual branch starts with zeros, and each residual block behaves like an identity
            if zero_init_residual and isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

        # name
        self.name = 'ExpSampleCNN' if name is None else name
        if robust:
            LOG.info(f"{self.name}\nmodel modules: {self.n_modules}\nmodel layers: {self.n_layers}\nmodel params: {self.n_params}")
        return

    def _make_layer(self, block: Type[Bottleneck], planes: int, n_blocks: int, stride: int = 1, expansion: int = 2):
        if stride != 1 or self.in_planes != planes * expansion:
            downsample = nn.Sequential(
                conv1x3(self.in_planes, planes * expansion, stride=stride),
                self._norm_layer(planes * expansion)
            )
        else:
            downsample = None

        layers = [block(self.in_planes, planes, stride, downsample, expansion)]
        self.in_planes = planes * expansion
        for _ in range(1, n_blocks):
            layers.append(
                block(self.in_planes, planes, 1, expansion=expansion)
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # head
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layers
        for i in range(self.n_modules):
            x = getattr(self, f'layer{i+1}')(x)

        # output
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @property
    def n_params(self):
        return count_params(self)


def expcnn_arg_parser(p: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if not p:
        p = argparse.ArgumentParser('Experimental Sample CNN Sanity Check', add_help=False)
        p.add_argument('--n_class', default=50, type=int)
        p.add_argument('--name', type=str)
    p.add_argument('--layers', default=[1, 2, 2, 6, 3, 2], type=int, nargs='+')
    p.add_argument('--channels', default=[64, 128, 128, 256, 256, 512], type=int, nargs='+')
    p.add_argument('--in_channels', default=1, type=int)
    p.add_argument('--expansion', default=2, type=int)
    p.add_argument('--zero_init_residual', action='store_true', help='Zero-initialize the last BN in each residual branch')
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample CNN Sanity Check Script', parents=[expcnn_arg_parser()])
    args = parser.parse_args()

    exp = ExpSampleCNN(block=Bottleneck,
                       layers=args.layers,
                       channels=args.channels,
                       in_channels=args.in_channels,
                       expansion=args.expansion,
                       n_class=args.n_class,
                       zero_init_residual=args.zero_init_residual,
                       name=args.name,
                       robust=True)
    for mod_name, mod in exp.named_modules():
        if mod_name == 'avgpool' and mod_name != '':
            mod.register_forward_hook(layer_print_hock)
    CONSOLE.print(f'model:\n{exp}')

    CONSOLE.print(f'In shape: {(3, 1, 59049)}')
    CONSOLE.print(f'Out shape: {exp(torch.randn(3, 1, 59049)).shape}')
