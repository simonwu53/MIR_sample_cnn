import numpy as np
import random
import torch
import pandas as pd
from logging import RootLogger


def get_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def reset_all_seeds(seed: int) -> np.random.Generator:
    """
    Set global seed for "random", and "torch" modules.
    Also, get local RNG from "numpy" instead of setting the global seed.
    :param seed: seed to set
    :return: numpy rng generator
    """
    rng = get_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return rng


def load_topk_annotations(path: str, reset_index: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if reset_index:
        df = df.reset_index('clip_id')
    return df


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience: int = 5,
                 min_delta: int = 0,
                 verbose: bool = False,
                 prefix: str = '',
                 logger: RootLogger = None):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        :param verbose: if True, print in console when metric is not improving
        :param prefix: prefix text for console printing
        :param logger: logging object to print information instead of bulletin print
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        self.prefix = prefix
        self.print = logger.info if logger is not None else print
        return

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            self.print(self.prefix + f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.print(self.prefix + 'Early stopping!')
                self.early_stop = True
        return
