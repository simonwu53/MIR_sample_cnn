from .cnn import cnn_arg_parser, SampleCNN
from .misc import get_optimizer, get_loss, find_optimal_model, EarlyStopping


def build_model(args):
    # build model
    model = SampleCNN(
        n_class=args.n_class,
        m=args.m,
        n=args.n,
        samples_in=args.samples_in,
        module_filters=args.module_filters,
        dropout=args.dropout,
        name=args.name
    )

    return model
