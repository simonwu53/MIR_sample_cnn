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
    n_segment = total_samples // n_samples  # MTT has 10 segments when sample size is 59049

    if center:
        # take center samples
        residual = total_samples % n_samples
        audio = audio[residual // 2: -residual // 2]

    # split into segments
    segments = [audio[i*n_samples:(i+1)*n_samples] for i in range(n_segment)]
    return segments


def MTT_statistics(args):
    from src.data.dataset import MTTDataset
    from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TextColumn

    # initialize statistics
    dataset = MTTDataset(path=args.p_data, split=args.split)
    total_segments = len(dataset)
    n_samples_per_segment = args.n_samples
    sum_x, sum_x2 = 0, 0
    n_samples_total = total_segments * n_samples_per_segment

    # iterating over dataset
    with Progress("[progress.description]{task.description}",
                  BarColumn(),
                  "[progress.percentage]{task.percentage:>3.0f}%",
                  TimeRemainingColumn(),
                  TextColumn("/"),
                  TimeElapsedColumn(),
                  "{task.completed} of {task.total} steps",
                  expand=False, console=CONSOLE, refresh_per_second=5) as progress:
        task = progress.add_task(description='[Scanning Dataset] ', total=total_segments)

        for i in range(total_segments):
            sample, label = dataset[i]
            sum_x += np.sum(sample)
            sum_x2 += np.sum(sample**2)

            progress.update(task, advance=1)

    LOG.info("Calculating final results...")
    mean = sum_x / n_samples_total
    std = np.sqrt((sum_x2 - sum_x**2/n_samples_total)/(n_samples_total-1))
    LOG.info(f"Mean: {mean}\nStddev: {std}")
    return


def data_arg_parser(p: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if not p:
        p = argparse.ArgumentParser('Audio Data Statistics Calculation', add_help=False)
    p.add_argument('--p_data', default='./dataset/processed', type=str)
    p.add_argument('--split', default='train', choices=['train', 'val', 'test'], type=str)
    p.add_argument('--sr', default=22050, type=int)
    p.add_argument('--n_samples', default=59049, type=int)
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Audio Data Statistics Calculation Script', parents=[data_arg_parser()])
    config = parser.parse_args()

    CONSOLE.rule("Dataset Statistics Calculation")
    MTT_statistics(config)
