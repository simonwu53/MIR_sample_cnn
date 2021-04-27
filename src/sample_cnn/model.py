# configs
try:
    from src.utils import LOG, CONSOLE
except ModuleNotFoundError as e:
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent.absolute().as_posix())
    from src.utils import LOG, CONSOLE, traceback_install
    traceback_install(console=CONSOLE, show_locals=True)
# libs
import torch
from torch import nn
# typing
from typing import Optional, Union, Tuple


def get_activation(name: str) -> nn.Module:
    if name.lower() == 'relu':
        return nn.ReLU()
    else:
        LOG.warning(f'Unrecognized activation function: [bold yellow]{name}[/], fallback to default [bold yellow]ReLU[/].')
        return nn.ReLU()


class CNNModule(nn.Module):
    def __init__(self,
                 len_in: int,
                 len_out: int,
                 kernel_size: Union[int, Tuple[int,...]] = 3,
                 stride: Union[int, Tuple[int,...]] = 1,
                 padding: Union[int, Tuple[int,...]] = 1,
                 activation: str = 'relu',
                 pool_stride: Optional[int] = None) -> None:
        """
        Sample CNN layer module.
        :param len_in:      input num of filters for Conv1d
        :param len_out:     output num of filters for Conv1d
        :param kernel_size: kernel size for Conv1d
        :param stride:      stride for Conv1d
        :param padding:     padding for Conv1d
        :param activation:  activation function for the layer module
        :param pool_stride: Optional, pooling kernel size and stride if enabled
        """
        super(CNNModule, self).__init__()
        # conv1d
        self.conv1d = nn.Conv1d(len_in, len_out, kernel_size=kernel_size, stride=stride, padding=padding)
        # batch normalization
        self.norm = nn.BatchNorm1d(len_out)
        # activation
        self.activate = get_activation(activation)
        # optional max pooling after layer
        if pool_stride is not None:
            self.maxpool = nn.MaxPool1d(kernel_size=pool_stride, stride=pool_stride)
        else:
            self.maxpool = None
        return

    def forward(self, x):
        if self.maxpool is None:
            return self.activate(self.norm(self.conv1d(x)))
        else:
            return self.maxpool(self.activate(self.norm(self.conv1d(x))))


class SampleCNN(nn.Module):
    def __init__(self,
                 n_class: int = 50,
                 m: int = 3,
                 n: int = 9,
                 samples_in=59049,
                 module_filters: Tuple[int, ...] = (128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512),
                 dropout: float = 0.5):
        """
        Original Sample CNN in PyTorch implementation
        :param n_class:         number of output tags
        :param m:               filter/kernel size and pooling filter/kernel size of module layers
        :param n:               number of module layers (depth)
        :param samples_in:      1d input dimension for raw waveform
        :param module_filters:  list of filter quantity for each module layer
        :param dropout:         dropout rate for the last output before the classifier
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
        LOG.info(f'SampleCNN args <n_class={n_class}, m={m}, n={n}, samples_in={samples_in}, module_filters={module_filters}, dropout={dropout}>')
        # TODO: add support for even kernel size!
        padding = kernel_size // 2
        assert len(module_filters) == n+2, \
            LOG.error(f"The size of module filters<len:{len(module_filters)}> should be 'n+2'=<{n+2}>, which includes input, output layers!")
        self.n = n

        # build layers
        for i in range(n+2):
            if i == 0:
                # input layer
                setattr(self, 'head', CNNModule(1, module_filters[i], kernel_size, stride, 0))

            elif i == n+1:
                # output layer
                setattr(self, 'out', CNNModule(out_filters, module_filters[i], 1, 1, 0))

            else:
                # module layers
                setattr(self, f'module_layer_{i}', CNNModule(out_filters, module_filters[i], kernel_size, 1, padding, pool_stride=m))

            # out filters for next layer
            out_filters = module_filters[i]

        # dropout
        self.drop = nn.Dropout(dropout)

        # classifier for tags
        self.classifier = nn.Linear(module_filters[-1], n_class)
        self.activation = nn.Sigmoid()
        return

    def forward(self, x):
        x = self.head(x)
        for i in range(self.n):
            x = getattr(self, f'module_layer_{i+1}')(x)
        out = self.out(x)
        drop = self.drop(out)
        flat = drop.flatten(1)
        logits = self.classifier(flat)
        return self.activation(logits)


if __name__ == '__main__':
    def layer_print_hock(module, inputs, outputs):
        CONSOLE.print(f'{module.__class__}')
        CONSOLE.print(f'input shape: [bold red]{inputs[0].shape}[/]')
        CONSOLE.print(f'output shape: [bold red]{outputs.shape}[/]')

    CONSOLE.rule('[bold cyan]Sample CNN moodel test[/]')
    CONSOLE.print('Creating model...')
    model = SampleCNN(
        n_class=50,
        m=3,
        n=9,
        samples_in=59049,
        module_filters=(128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512),
        dropout=0.5
    )

    # register hock to print layer shapes
    for name, mod in model.named_modules():
        if len(name.split('.')) == 1 and name != '':
            mod.register_forward_hook(layer_print_hock)

    CONSOLE.print(f'Model summary:\n{model}')
    CONSOLE.print('Pushing model to GPU...')
    model.cuda()

    # sample inputs
    sample = torch.randn(5, 1, 59049, device='cuda')

    # model forward pass
    output = model(sample)

    CONSOLE.print('Test done.')
