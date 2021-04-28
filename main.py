from src.models import cnn_arg_parser, build_model
from src.utils import PATH, LOG, CONSOLE, traceback_install
import argparse
# from typing import Optional


# substitute default traceback
traceback_install(console=CONSOLE, show_locals=True)


def main_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser('SAMPLE-LEVEL DEEP CONVOLUTIONAL NEURAL NETWORKS FOR MUSIC AUTO-TAGGING USING RAW WAVEFORMS', add_help=False)
    # ---SampleCNN---
    p = cnn_arg_parser(p)
    # ---Optimizer---
    p.add_argument('--lr_type', default='sgd', type=str)
    p.add_argument('--lr', default=1e-2, type=float)
    # ---SGD---
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--lr_decay', default=1e-6, type=float)
    # ---Adam/AdamW---
    p.add_argument('--betas', default=[0.9, 0.999], type=float, nargs='+')
    p.add_argument('--eps', default=1e-8, type=float)
    # ---AdamW---
    p.add_argument('--weight_decay', default=1e-2, type=float)

    return p


def main(args):
    # create model
    model, optim = build_model(args)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sample CNN Main Utility Script', parents=[main_arg_parser()])
    args = parser.parse_args()
    main(args)
