from pathlib import Path
from collections import namedtuple
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install

# config paths
_base_dir = Path(__file__).parent.parent.parent.absolute()
_log_dir = _base_dir.joinpath('logs')
if not _log_dir.exists():
    _log_dir.mkdir()
# TODO: setup data path later
_data_dir = None
_data_train_dir = None
_data_valid_dir = None
_data_test_dir = None

_dir = namedtuple('PATH_COLLECTION', ('base', 'log', 'database', 'data_train', 'data_valid', 'data_test'))
PATH = _dir(_base_dir, _log_dir, _data_dir, _data_train_dir, _data_valid_dir, _data_test_dir)

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
