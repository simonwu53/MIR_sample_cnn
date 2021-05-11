from src.models import cnn_arg_parser, expcnn_arg_parser
from src.utils import LOG, CONSOLE, traceback_install
from train import train_on_model
from eval import test_on_model
import argparse
import json


# substitute default traceback
traceback_install(console=CONSOLE, show_locals=True)


def main_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('SAMPLE-LEVEL DEEP CONVOLUTIONAL NEURAL NETWORKS FOR MUSIC AUTO-TAGGING USING RAW WAVEFORMS', add_help=False)
    # ---SampleCNN args---
    # p.add_argument('--m', default=3, type=int)
    # p.add_argument('--n', default=9, type=int)
    # p.add_argument('--samples_in', default=59049, type=int)
    # p.add_argument('--module_filters', default=[128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512], type=int, nargs='+')
    # p.add_argument('--dropout', default=0.5, type=float)
    p = cnn_arg_parser(p)  # check args listed above
    # p.add_argument('--layers', default=[1, 2, 4, 3, 2], type=int, nargs='+')
    # p.add_argument('--channels', default=[64, 128, 256, 256, 512], type=int, nargs='+')
    # p.add_argument('--in_channels', default=1, type=int)
    # p.add_argument('--expansion', default=2, type=int)
    # p.add_argument('--zero_init_residual', action='store_true', help='Zero-initialize the last BN in each residual branch')
    p = expcnn_arg_parser(p)
    p.add_argument('--mode',
                   default='train',
                   choices=['train', 'test'],
                   type=str)
    p.add_argument('--device', default='cuda:0', type=str, help='Training device')
    p.add_argument('--p_out', default='./out', type=str, help='Output directory for saving, '
                                                              'will be ignored if use checkpoint')

    # ---Model---
    p.add_argument('--model',
                   default='samplecnn',
                   choices=['samplecnn', 'expcnn'],
                   type=str)
    p.add_argument('--n_class', default=50, type=int)
    p.add_argument('--name', type=str)

    # ---Optimizer---
    p.add_argument('--optim_type', default='sgd', type=str, help='Optimizer type')
    p.add_argument('--lr', default=1e-2, type=float, help='learning rate during training')
    p.add_argument('--min_lr', default=0.000016, type=float, help='Minimum learning rate')
    p.add_argument('--lr_decay_plateau', default=0.2, type=float, help='Plateau decay')
    p.add_argument('--plateau_patience', default=3, type=int, help='Plateau patience')
    p.add_argument('--load_optimal_on_plateau', action='store_true', help='Load optimal model when plateau')
    p.add_argument('--early_stop_patience', default=0, type=int, help='Early stop settings for training. default 0, disabled.')
    p.add_argument('--early_stop_delta', default=0, type=float, help='Early stop settings for training')
    # ---loss---
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
    p.add_argument('--max_epoch', default=100, type=int, help='Epoch for each training')
    p.add_argument('--p_data', default='./dataset/processed', type=str,
                   help='Path to the pre-processed dataset.')
    p.add_argument('--data_normalization', action='store_true', help='Normalize batch data while training')
    p.add_argument('--batch_size', default=23, type=int, help='Batch size for training')
    p.add_argument('--n_workers', default=4, type=int, help='Number of workers for data loading')
    # ---resume training---
    p.add_argument('--checkpoint', type=str, help='Resume training from checkpoint, '
                                                  'other params will be ignored. '
                                                  'Params from last session will be restored')
    p.add_argument('--ckpt_epoch', type=int, help='Override checkpoint epoch value')
    p.add_argument('--ckpt_no_scheduler', action='store_true', help='Remove checkpoint schedulers')
    p.add_argument('--ckpt_no_optimizer', action='store_true', help='Remove checkpoint optimizer')
    p.add_argument('--ckpt_no_loss_fn', action='store_true', help='Remove checkpoint loss function')
    p.add_argument('--ckpt_map_values', type=json.loads, help="""Other values to modify in checkpoint, inputs are string dict, e.g. '{"value1":"key1"}'""")
    # ---misc---
    p.add_argument('--tensorboard_interval', default=200, type=int, help='Tensorboard writer update interval')
    p.add_argument('--tensorboard_exp_name', default='exp1', type=str, help='Tensorboard experiment name for log folder')
    return p


def main(args):
    CONSOLE.rule("Sample CNN Main Script")
    CONSOLE.print(args)

    # determine mode
    LOG.info(f"Mode selected: {args.mode}")
    if args.mode == 'train':
        train_on_model(args)
    elif args.mode == 'test':
        test_on_model(args)
    else:
        raise ValueError

    CONSOLE.rule("Task finished")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample CNN Main Utility Script', parents=[main_arg_parser()])
    config = parser.parse_args()
    main(config)
