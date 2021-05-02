# configs
try:
    from src.utils import LOG, CONSOLE
except ModuleNotFoundError as e:
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent.absolute().as_posix())
    from src.utils import LOG, CONSOLE, traceback_install
    traceback_install(console=CONSOLE, show_locals=True)
from .misc import _get_activation, count_params, layer_print_hock
# libs
import torch
import torch.nn.functional as F
from torch import nn
import argparse
# typing
from typing import Optional, Union, Tuple


class Padding(nn.Module):
    def __init__(self, padding: Union[int, Tuple[int, ...]], mode: str = 'constant', value: int = 0):
        """
        A wrapper module for padding function

        :param padding: m-elements tuple, where m/2 ≤ input dimensions and m is even
        :param mode:    'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        :param value:   fill value for 'constant' padding. Default: 0
        """
        super(Padding, self).__init__()
        self.activated = False if padding == 0 else True
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.value = value
        if mode.lower() not in ['constant', 'reflect', 'replicate' or 'circular']:
            LOG.warning(f"Unexpected padding mode: {mode}, fallback to default mode: 'constant'.")
            self.mode = 'constant'
        else:
            self.mode = mode
        return

    def forward(self, x):
        if self.activated:
            return F.pad(x, self.padding, mode=self.mode, value=self.value)
        else:
            return x


class CNNModule(nn.Module):
    def __init__(self,
                 len_in: int,
                 len_out: int,
                 kernel_size: Union[int, Tuple[int,...]] = 3,
                 stride: Union[int, Tuple[int,...]] = 1,
                 padding: Union[int, Tuple[int,...]] = 1,
                 activation: str = 'relu',
                 pool_stride: Optional[int] = None,
                 name: str = 'CNNModule') -> None:
        """
        Sample CNN layer module.
        :param len_in:      input num of filters for Conv1d
        :param len_out:     output num of filters for Conv1d
        :param kernel_size: kernel size for Conv1d
        :param stride:      stride for Conv1d
        :param padding:     padding for Conv1d
        :param activation:  activation function for the layer module
        :param pool_stride: Optional, pooling kernel size and stride if enabled
        :param name:        Optional, name the module
        """
        super(CNNModule, self).__init__()
        # padding
        self.padding = Padding(padding=padding)
        # conv1d
        self.conv1d = nn.Conv1d(len_in, len_out, kernel_size=kernel_size, stride=stride)
        # batch normalization
        self.norm = nn.BatchNorm1d(len_out)
        # activation
        self.activate = _get_activation(activation)
        # optional max pooling after layer
        if pool_stride is not None:
            self.maxpool = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        else:
            self.maxpool = None
        # name
        self.name = name
        return

    def forward(self, x):
        if self.maxpool is None:
            return self.activate(self.norm(self.conv1d(self.padding(x))))
        else:
            return self.maxpool(self.activate(self.norm(self.conv1d(self.padding(x)))))


class SampleCNN(nn.Module):
    def __init__(self,
                 n_class: int = 50,
                 m: int = 3,
                 n: int = 9,
                 samples_in=59049,
                 module_filters: Tuple[int, ...] = (128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512),
                 dropout: float = 0.5,
                 name: Optional[str] = None):
        """
        Original Sample CNN in PyTorch implementation

        m^n architecture:
        Constraints     : 1) kernel_size * num_frames = num_samples (e.g. 3 * 19683 = 59049; 9 * 6561 = 59049)
                          2) m**n = num_frames                      (e.g. 3**9 = 19683; 3**8 = 6561)
                          3) kernel_size = num_samples / (m**n)     (derived from formula above)
                          4) layer downscaling ratio = m
        Explanation: Given an input with 59049 samples and a 3^9-model,
                     the input will be divided into 19683 frames in the first layer (input head).
                     In the following module layers, tensor length will be downscaled by m for every layer.
                     In the last out layer, the output tensor length will be fixed at 1.
                     Finally, the output tensor will be flattened and a fully connected layer will be used to output classifications.

        :param n_class:         number of output tags
        :param m:               filter/kernel size and pooling filter/kernel size of module layers
        :param n:               number of module layers (depth)
        :param samples_in:      1d input dimension for raw waveform
        :param module_filters:  list of filter quantity for each module layer
        :param dropout:         dropout rate for the last output before the classifier
        :param name:            Optional, name the module
        """
        super(SampleCNN, self).__init__()
        # infer the kernel size and model structure based on m, n, and len_in
        self.frames = m**n                              # number of frames in the first conv1d to segment the raw waveform
        assert samples_in % self.frames == 0, \
            LOG.error(f"Model size mismatch! "
                      f"Given m^n={m}^{n}={self.frames} frames, can not be divided by number of samples {samples_in}. "
                      f"Constraint formula: [red]<Samples_in%(m**n)!=0>[/]")
        kernel_size = samples_in // self.frames
        stride = kernel_size
        # padding for both odd and even kernel size
        padding = (kernel_size//2-1, kernel_size//2) if kernel_size % 2 == 0 else kernel_size // 2
        assert len(module_filters) == n+2, \
            LOG.error(f"The size of module filters<len:{len(module_filters)}> should be 'n+2'=<{n+2}>, which includes input, output layers!")
        self.n = n

        # build layers
        for i in range(n+2):
            if i == 0:
                # input layer
                setattr(self, 'head', CNNModule(1, module_filters[i], kernel_size, stride, 0, name='head'))

            elif i == n+1:
                # output layer
                setattr(self, 'out', CNNModule(out_filters, module_filters[i], 1, 1, 0, name='out'))

            else:
                # module layers
                setattr(self, f'module_layer_{i}', CNNModule(out_filters, module_filters[i], kernel_size, 1, padding, pool_stride=m, name=f'module_layer_{i}'))

            # out filters for next layer
            out_filters = module_filters[i]

        # dropout
        self.drop = nn.Dropout(dropout)

        # classifier for tags
        self.classifier = nn.Linear(module_filters[-1], n_class)
        self.sigmoid = nn.Sigmoid()

        # name
        self.name = f'{m}^{n} Model with {samples_in} samples' if not name else name
        LOG.info(f'SampleCNN args <n_class={n_class}, m={m}, n={n}, samples_in={samples_in}, module_filters={module_filters}, dropout={dropout}, '
                 f'kernel_size={kernel_size}, stride={stride}, padding={padding}>')
        return

    def forward(self, x):
        x = self.head(x)
        for i in range(self.n):
            x = getattr(self, f'module_layer_{i+1}')(x)
        out = self.out(x)
        drop = self.drop(out)
        flat = drop.flatten(1)
        logits = self.classifier(flat)
        return self.sigmoid(logits)


def cnn_arg_parser(p: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if not p:
        p = argparse.ArgumentParser('Sample CNN Sanity Check', add_help=False)
    p.add_argument('--n_class', default=50, type=int)
    p.add_argument('--m', default=3, type=int)
    p.add_argument('--n', default=9, type=int)
    p.add_argument('--samples_in', default=59049, type=int)
    p.add_argument('--module_filters', default=[128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512], type=int, nargs='+')
    p.add_argument('--dropout', default=0.5, type=float)
    p.add_argument('--name', type=str)
    return p


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser('Sample CNN Sanity Check Script', parents=[cnn_arg_parser()])
    args = parser.parse_args()

    # build model
    CONSOLE.rule('[bold cyan]Sample CNN Sanity Check[/]')
    CONSOLE.print('Creating model...')
    model = SampleCNN(
        n_class=args.n_class,
        m=args.m,
        n=args.n,
        samples_in=args.samples_in,
        module_filters=args.module_filters,
        dropout=args.dropout,
        name=args.name
    )
    CONSOLE.print(f'{model.name} summary:\n{model}')
    CONSOLE.print(f'Total trainable parameters: [bold cyan]{count_params(model)}[/]')

    # register hock to print layer shapes
    for mod_name, mod in model.named_modules():
        if len(mod_name.split('.')) == 1 and mod_name != '':
            mod.register_forward_hook(layer_print_hock)

    CONSOLE.print('Pushing model to GPU...')
    model.cuda()

    # sample inputs
    sample = torch.randn(5, 1, args.samples_in, device='cuda')

    # model forward pass
    CONSOLE.rule('[bold cyan]Forwarding test[/]')
    output = model(sample)

    CONSOLE.print('Test done.')
