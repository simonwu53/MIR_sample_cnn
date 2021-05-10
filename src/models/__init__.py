from .sample_cnn import cnn_arg_parser, SampleCNN
from .misc import get_optimizer, get_loss, find_optimal_model, load_ckpt, apply_lr, show_ckpt, apply_state_dict, EarlyStopping, ReduceLROnPlateau, AUCMetric


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
