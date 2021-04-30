try:
    # run global
    from src.utils import LOG, CONSOLE
    from pathlib import Path, PosixPath
except ModuleNotFoundError as e:
    # run local
    import sys
    from pathlib import Path, PosixPath
    sys.path.append(Path(__file__).parent.parent.parent.absolute().as_posix())
    from src.utils import LOG, CONSOLE, traceback_install
    traceback_install(console=CONSOLE, show_locals=True)
import numpy as np
import librosa
from typing import Union, Optional, List
import argparse


def _load_audio(path: Union[str, PosixPath], sample_rate: int = 22050) -> np.ndarray:
    path = path.as_posix() if isinstance(path, PosixPath) else path
    y, sr = librosa.load(path, sr=sample_rate)
    return y


def _segment_audio(audio: np.ndarray, n_samples: int = 59049, center: bool = False) -> List[np.ndarray]:
    total_samples = audio.shape[0]
    n_segment = total_samples // n_samples

    if center:
        # take center samples
        residual = total_samples % n_samples
        audio = audio[residual // 2: -residual // 2]

    # split into segments
    segments = [audio[i*n_samples:(i+1)*n_samples] for i in range(n_segment)]
    return segments


def MTT_statistics(args):

    return


def data_arg_parser(p: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if not p:
        p = argparse.ArgumentParser('Audio Data Statistics Calculation', add_help=False)
    p.add_argument('--p_data', default='./dataset/raw', type=str)
    p.add_argument('--sr', default=22050, type=int)
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Audio Data Statistics Calculation Script', parents=[data_arg_parser()])
    config = parser.parse_args()

    CONSOLE.rule("Dataset Statistics Calculation")
    MTT_statistics(config)
