import numpy as np
import random
import torch


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
