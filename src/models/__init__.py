from .sample_cnn import cnn_arg_parser, SampleCNN
from .exp_cnn import expcnn_arg_parser, ExpSampleCNN, Bottleneck
from .misc import get_optimizer, get_loss, find_optimal_model, load_ckpt, apply_lr, show_ckpt, apply_state_dict, EarlyStopping, ReduceLROnPlateau, AUCMetric


def build_model(args):
    # build model
    if args.model == 'samplecnn':
        model = SampleCNN(
            n_class=args.n_class,
            m=args.m,
            n=args.n,
            samples_in=args.samples_in,
            module_filters=args.module_filters,
            dropout=args.dropout,
            name=args.name
        )
    elif args.model == 'expcnn':
        model = ExpSampleCNN(block=Bottleneck,
                             layers=args.layers,
                             channels=args.channels,
                             in_channels=args.in_channels,
                             expansion=args.expansion,
                             n_class=args.n_class,
                             zero_init_residual=args.zero_init_residual,
                             name=args.name,
                             robust=True)
    else:
        raise NotImplementedError(f"Unknown model name: {args.model}")

    return model
