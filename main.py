from src.models import cnn_arg_parser
from src.utils import PATH, LOG, CONSOLE, traceback_install
import argparse
from train import train_on_model


# substitute default traceback
traceback_install(console=CONSOLE, show_locals=True)


def main_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('SAMPLE-LEVEL DEEP CONVOLUTIONAL NEURAL NETWORKS FOR MUSIC AUTO-TAGGING USING RAW WAVEFORMS', add_help=False)
    # ---SampleCNN---
    # p.add_argument('--n_class', default=50, type=int)
    # p.add_argument('--m', default=3, type=int)
    # p.add_argument('--n', default=9, type=int)
    # p.add_argument('--samples_in', default=59049, type=int)
    # p.add_argument('--module_filters', default=[128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512], type=int, nargs='+')
    # p.add_argument('--dropout', default=0.5, type=float)
    # p.add_argument('--name', type=str)
    p = cnn_arg_parser(p)  # check args listed above
    p.add_argument('--mode',
                   default='train',
                   choices=['train', 'val', 'test'],
                   type=str)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                   type=str, help='Training device')
    # ---Optimizer---
    p.add_argument('--optim_type', default='sgd', type=str, help='Optimizer type')
    p.add_argument('--lr', default=1e-2, type=float, help='learning rate during training')
    p.add_argument('--init_lr', default=1e-2, type=float, help='Initial learning rate')
    p.add_argument('--min_lr', default=0.000016, type=float, help='Minimum learning rate')
    p.add_argument('--lr_decay', default=0.2, type=float,
                   help='Learning rate decay for each retraining')
    p.add_argument('--lr_decay_plateau', default=0.2, type=float, help='Plateau decay')
    p.add_argument('--plateau_patience', default=3, type=int, help='Plateau patience')
    p.add_argument('--loss', default='bce', choices=['bce'], type=str,
                   help='Loss function selection')
    # ---SGD---
    p.add_argument('--momentum', default=0.9, type=float, help='SGD configuration')
    # ---Adam/AdamW---
    p.add_argument('--betas', default=[0.9, 0.999], type=float, nargs='+',
                   help='Adam/AdamW configuration')
    p.add_argument('--eps', default=1e-8, type=float, help='Adam/AdamW configuration')
    # ---AdamW---
    p.add_argument('--weight_decay', default=1e-2, type=float, help='AdamW configuration')
    # ---Dataset---
    p.add_argument('--max_train', default=5, type=int, help='Number of retraining')
    p.add_argument('--max_epoch', default=100, type=int, help='Epoch for each training')
    p.add_argument('--p_data', default='./dataset/processed', type=str,
                   help='Path to the pre-processed dataset.')
    p.add_argument('--batch_size', default=23, type=int, help='Batch size for training')
    p.add_argument('--n_workers', default=4, type=int, help='Number of workers for data loading')
    # ---resume---
    p.add_argument('--checkpoint', type=str, help='Resume training from checkpoint, '
                                                  'other params will be ignored. '
                                                  'Params from last session will be restored.')
    p.add_argument('--p_out', type=str, help='Output directory for saving')

    return p


def main(args):
    CONSOLE.rule("Sample CNN Main Script")
    CONSOLE.print(args)

    # determine mode
    if args.mode == 'train':
        LOG.info(f"Mode selected: {args.mode}")
        train_on_model(args)

    elif args.mode == 'val':
        raise NotImplementedError

    elif args.mode == 'test':
        raise NotImplementedError
    else:
        raise NotImplementedError

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample CNN Main Utility Script', parents=[main_arg_parser()])
    config = parser.parse_args()
    main(config)
