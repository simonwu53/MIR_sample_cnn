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
                 module_filters: Tuple[int, ...] = (128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512),
                 dropout: float = 0.5):
        """
        Original Sample CNN in PyTorch implementation
        :param n_class:         number of output tags
        :param m:               filter/kernel size and pooling filter/kernel size of module layers
        :param n:               number of module layers (depth)
        :param module_filters:  list of filter quantity for each module layer
        :param dropout:         dropout rate for the last output before the classifier
        """
        super(SampleCNN, self).__init__()
        assert len(module_filters) == n+2, \
            LOG.error(f"The size of module filters[{len(module_filters)}] should be 'n+2'=[{n+2}], which includes input, output layers!")
        # configs
        self.n = 9

        # build layers
        for i in range(n+2):
            if i == 0:
                # input layer
                setattr(self, 'head', CNNModule(1, module_filters[i], m, m, 0))

            elif i == n+1:
                # output layer
                setattr(self, 'out', CNNModule(out_filters, module_filters[i], m, 1, 1))

            else:
                # module layers
                setattr(self, f'module_layer_{i}', CNNModule(out_filters, module_filters[i], m, 1, 1, pool_stride=m))

            # out filters for next layer
            out_filters = module_filters[i]

        # dropout
        self.drop = nn.Dropout(dropout)

        # classifier for tags
        self.classifier = nn.Linear(module_filters[-1], n_class)
        self.activation = nn.Sigmoid()
        return

    def forward(self, x):
        # x shape (B, 1, L)
        x = self.head(x)
        # x shape (B, 128, L//m)
        for i in range(self.n):
            x = getattr(self, f'module_layer_{i+1}')(x)
        # x shape (B, 512, L//(m^(n+1))), L//(m^(n+1)) == 1, 3^10=59049
        drop = self.drop(x)
        flat = drop.flatten(1)
        logits = self.classifier(flat)
        return self.activation(logits)


def build_model():
    # TODO Log model spec here

    # TODO return model, criterion, *others
    return


if __name__ == '__main__':
    CONSOLE.rule('[bold cyan]Sample CNN moodel test[/]')
    CONSOLE.print('Creating model...')
    model = SampleCNN(
        n_class=50,
        m=3,
        n=9,
        module_filters=(128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512),
        dropout=0.5
    )
    CONSOLE.print(f'Model summary:\n{model}')
    CONSOLE.print('Pushing model to GPU...')
    model.cuda()

    # sample inputs
    sample = torch.randn(5, 1, 59049, device='cuda')
    CONSOLE.print(f'Test input shape: [bold red]{sample.shape}[/]')

    # model forward pass
    out = model(sample)
    CONSOLE.print(f'Model output shape: [bold red]{out.shape}[/]')
    CONSOLE.print('Test done.')
