from .func import reset_all_seeds
from pathlib import Path
from collections import namedtuple
import logging
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# config paths
_base_dir = Path(__file__).parent.parent.parent.absolute()
_log_dir = _base_dir.joinpath('logs')
if not _log_dir.exists():
    _log_dir.mkdir()

_dir = namedtuple('PATH_COLLECTION', ('base', 'log'))
PATH = _dir(_base_dir, _log_dir)

# logger
CONSOLE = Console(record=True)  # export using export_text(), title using rule(title='title')
handler = RichHandler(level=logging.INFO,
                      console=CONSOLE,
                      show_time=True,
                      show_level=True,
                      show_path=True,
                      markup=True,
                      rich_tracebacks=True)
logging.basicConfig(format='%(message)s',
                    datefmt='[%x %X]',
                    level=logging.INFO,
                    handlers=[handler])
LOG = logging.getLogger('sample_cnn')

# RANDOM SEEDS (with new practice, use RNG-Random_Number_Generator, which can be used locally)
_seed = 53
RNG = reset_all_seeds(_seed)

# Dataset Statistics
MTT_MEAN = -0.00016837834875982804
MTT_STD = 0.1563320988805334
