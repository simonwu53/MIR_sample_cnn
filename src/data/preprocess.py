from pathlib import Path, PosixPath
try:
    # run global
    from src.utils import LOG, CONSOLE
    from src.data.audio import _load_audio, _segment_audio
except ModuleNotFoundError as e:
    # run local
    import sys
    sys.path.append(Path(__file__).parent.parent.parent.absolute().as_posix())
    from src.utils import LOG, CONSOLE, traceback_install
    from src.data.audio import _load_audio, _segment_audio
    traceback_install(console=CONSOLE, show_locals=True)
import numpy as np
import pandas as pd
import argparse
import shutil
import time
from multiprocessing import Process
from typing import Optional


def _process_audio_files(worker_id: int,
                         tasks: pd.DataFrame,
                         p_out: PosixPath,
                         p_raw: PosixPath,
                         n_samples: int = 59049,
                         sample_rate: int = 22050,
                         topk: int = 50,
                         file_pattern: str = 'clip-{}-seg-{}-of-{}') -> None:
    n_tasks = tasks.shape[0]
    t_start = time.time()
    n_parts = n_tasks // 10
    idx = 0
    LOG.info(f"[Worker {worker_id:02d}]: Received {n_tasks} tasks.")

    for i, t in tasks.iterrows():
        # find output dir
        split = t.split
        out_dir = p_out.joinpath(split)

        # process audio file
        try:
            segments = _segment_audio(_load_audio(p_raw.joinpath(t.mp3_path), sample_rate=sample_rate),
                                      n_samples=n_samples,
                                      center=False)
            loaded = True
        except (RuntimeError, EOFError) as e:
            LOG.warning(f"[Worker {worker_id:02d}]: Failed load audio: {t.mp3_path}. Ignored.")
            loaded = False

        # save label and segments to npy files
        if loaded:
            labels = t[t.index.tolist()[:topk]].values.astype(bool)
            n_segments = len(segments)
            for j, seg in enumerate(segments):
                np.savez_compressed(out_dir.joinpath(file_pattern.format(t.clip_id, j+1, n_segments)).as_posix(), data=seg, labels=labels)

        # report progress
        idx += 1
        if idx == n_tasks:
            LOG.info(f"[Worker {worker_id:02d}]: Job finished. Quit. (time usage: {(time.time() - t_start) / 60:.02f} min)")
        elif idx % n_parts == 0:
            LOG.info(f"[Worker {worker_id:02d}]: {idx//n_parts*10}% tasks done. (time usage: {(time.time() - t_start) / 60:.02f} min)")
    return


def process_MTT_annotations(p_anno: str = 'annotations_final.csv',
                            p_info: str = 'clip_info_final.csv',
                            delimiter: str = '\t',
                            n_top: int = 50) -> pd.DataFrame:
    """
    Reads annotation file, takes top N tags, and splits data samples

    Results 55 (top50_tags + [clip_id, mp3_path, split, shard]) columns:
    ['guitar', 'classical', 'slow', 'techno', 'strings',
    'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient',
    'beat', 'violin', 'vocal', 'synth', 'female', 'indian',
    'opera', 'male', 'singing', 'vocals', 'no vocals',
    'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male
    vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man',
    'classic', 'choir', 'voice', 'new age', 'dance', 'female
    vocal', 'male voice', 'beats', 'harp', 'cello', 'no voice',
    'weird', 'country', 'female voice', 'metal', 'choral',
    'clip_id', 'mp3_path', 'split', 'title', 'artist']

    NOTE: This will exclude audios which have only zero-tags. Therefore, number of
    each split will be 15250 / 1529 / 4332 (training / validation / test).

    :param p_anno:               A path to annotation CSV file
    :param p_info:               A path to song info CSV file
    :param delimiter:            csv delimiter
    :param n_top:                Number of the most popular tags to take
    :return:                     A DataFrame contains information of audios
                                 Schema:
                                    <tags>: 0 or 1
                                    clip_id:    clip_id of the original dataset
                                    mp3_path:   A path to a mp3 audio file
                                    split:      A split of dataset (training / validation / test).
                                                The split is determined by its directory (0, 1, ... , f).
                                                First 12 directories (0 ~ b) are used for training,
                                                1 (c) for validation, and 3 (d ~ f) for test.
                                    title:      title of the song that the clip is in
                                    artist:     artist of the song that the clip is in
    """
    def split_by_directory(mp3_path: str) -> str:
        directory = mp3_path.split('/')[0]
        part = int(directory, 16)

        if part in range(12):
            return 'train'
        elif part == 12:
            return 'val'
        elif part in range(13, 16):
            return 'test'

    LOG.info(f"Starting pre-processing MTT annotations, "
             f"keeping {n_top} top tags, "
             f"and merging song info.")

    # read csv annotation
    df_anno = pd.read_csv(p_anno, delimiter=delimiter)
    df_info = pd.read_csv(p_info, delimiter=delimiter)

    LOG.info(f"Loaded annotations from [bold]{p_anno}[/], "
             f"loaded song info from [bold]{p_info}[/], "
             f"which contains {df_anno.shape[0]} songs.")

    # get top50 tags
    top50 = df_anno.drop(['clip_id', 'mp3_path'], axis=1)\
        .sum()\
        .sort_values(ascending=False)[:n_top]\
        .index\
        .tolist()

    LOG.info(f"TOP 50 Tags:\n{top50}")

    # remove low frequency tags
    df_anno = df_anno[top50 + ['clip_id', 'mp3_path']]

    # remove songs that have 0 tag
    df_anno = df_anno[df_anno[top50].sum(axis=1) != 0]

    # creating train/val/test splits
    df_anno['split'] = df_anno['mp3_path'].transform(split_by_directory)

    # show splits
    for split in ['train', 'val', 'test']:
        LOG.info(f"{split} set size (#audio): {sum(df_anno['split'] == split)}.")

    # merge annotations and song info
    df_merge = pd.merge(df_anno, df_info[['clip_id', 'title', 'artist']], on='clip_id')

    LOG.info(f"Final quantity of songs: {df_merge.shape[0]}\nFinal columns ({df_merge.columns.size}) in the DataFrame:\n{df_merge.columns.tolist()}")
    return df_merge


def prepare_MTT_dataset(args):
    CONSOLE.rule("Pre-processing MTT Annotations and Data for Machine Learning")

    # --- get dirs ---
    # create out dir if not exists
    p_out = Path(args.p_out).absolute()
    while True:
        if p_out.exists():
            res = CONSOLE.input(f"Output folder exists ({p_out.as_posix()})! Do you want to remove it first? "
                                f"(You can also clean it manually now and hit enter key to retry) [y/n]: ")
            if res.lower() in ['y', 'yes']:
                # delete target folder
                shutil.rmtree(p_out)
                # create new one
                p_out.mkdir()
                LOG.info(f"Target folder removed, and new empty folder created.")
                break
            elif res.lower() in ['n', 'no']:
                LOG.error(f"Output folder exists! Creating folder failed. Target: {p_out.as_posix()} exists.")
                raise FileExistsError(f"Output folder exists! Creating folder failed. Target: {p_out.as_posix()} exists.")
            else:
                continue
        else:
            p_out.mkdir()
            LOG.info(f"Target folder ({p_out.as_posix()}) created.")
            break
    # train/val/test dirs
    p_out.joinpath('train').mkdir()
    p_out.joinpath('val').mkdir()
    p_out.joinpath('test').mkdir()
    # check raw data
    p_raw = Path(args.p_raw).absolute()
    assert len(list(p_raw.glob('[0-9, a-z]'))) == 16, "MTT Raw data should have 16 directories from 0-9 and a-f."

    # --- parsing and processing annotations ---
    annotations = process_MTT_annotations(p_anno=args.p_anno,
                                          p_info=args.p_info,
                                          delimiter=args.delimiter,
                                          n_top=args.n_topk)
    # save processed annotations
    annotations.to_csv(Path(args.p_anno).parent.joinpath(f'annotations_top{args.n_topk}.csv').as_posix(), index=False)
    # save topk labels
    with open(Path(args.p_anno).parent.joinpath('labels.txt').as_posix(), 'w') as f:
        f.write(','.join(annotations.columns.tolist()[:args.n_topk]))

    CONSOLE.rule("Audio Preprocessing")
    LOG.info(f"MTT annotations processed. Now segmenting audios based on annotations for machine learning...")

    # --- process audio files based on annotations ---
    avg = annotations.shape[0] // args.n_worker
    processes = [Process(target=_process_audio_files, args=(i,
                                                            annotations.iloc[i*avg:(i+1)*avg],
                                                            p_out,
                                                            p_raw,
                                                            args.n_samples,
                                                            args.sr,
                                                            args.n_topk))
                 if i != args.n_worker-1
                 else Process(target=_process_audio_files, args=(i,
                                                                 annotations.iloc[i*avg:],
                                                                 p_out,
                                                                 p_raw,
                                                                 args.n_samples,
                                                                 args.sr,
                                                                 args.n_topk))
                 for i in range(args.n_worker)]

    LOG.info(f"{args.n_worker} workers created.")

    # start jobs
    for p in processes:
        p.start()

    # wait jobs to finish
    for p in processes:
        p.join()

    CONSOLE.rule('MTT Dataset Preparation Done')
    return


def data_arg_parser(p: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if not p:
        p = argparse.ArgumentParser('MTT Data Pre-processing', add_help=False)
    p.add_argument('--n_topk', default=50, type=int)
    p.add_argument('--p_anno', default='./dataset/annotations_final.csv', type=str)
    p.add_argument('--p_info', default='./dataset/clip_info_final.csv', type=str)
    p.add_argument('--p_out', default='./dataset/processed/', type=str)
    p.add_argument('--p_raw', default='./dataset/raw/', type=str)
    p.add_argument('--delimiter', default='\t', type=str)
    p.add_argument('--n_samples', default=59049, type=int)
    p.add_argument('--sr', default=22050, type=int)
    p.add_argument('--n_worker', default=4, type=int)
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MTT Data Pre-processing Script', parents=[data_arg_parser()])
    config = parser.parse_args()
    prepare_MTT_dataset(config)
