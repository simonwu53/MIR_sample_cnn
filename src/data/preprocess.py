try:
    # run global
    from src.utils import LOG, CONSOLE, RNG
except ModuleNotFoundError as e:
    # run local
    import sys
    from pathlib import Path
    sys.path.append(Path(__file__).parent.parent.parent.absolute().as_posix())
    from src.utils import LOG, CONSOLE, RNG, traceback_install
    traceback_install(console=CONSOLE, show_locals=True)
import numpy as np
import pandas as pd


def process_MTT_annotations(path: str = 'annotations_final.csv',
                            delimiter: str = '\t',
                            n_top: int = 50,
                            n_audios_per_shard: int = 100) -> pd.DataFrame:
    """
    Reads annotation file, takes top N tags, and splits data samples

    Results 54 (top50_tags + [clip_id, mp3_path, split, shard]) columns:
    ['guitar', 'classical', 'slow', 'techno', 'strings',
    'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient',
    'beat', 'violin', 'vocal', 'synth', 'female', 'indian',
    'opera', 'male', 'singing', 'vocals', 'no vocals',
    'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male
    vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man',
    'classic', 'choir', 'voice', 'new age', 'dance', 'female
    vocal', 'male voice', 'beats', 'harp', 'cello', 'no voice',
    'weird', 'country', 'female voice', 'metal', 'choral',
    'clip_id', 'mp3_path', 'split', 'shard']

    NOTE: This will exclude audios which have only zero-tags. Therefore, number of
    each split will be 15250 / 1529 / 4332 (training / validation / test).

    :param path:                 A path to annotation CSV file
    :param delimiter:            csv delimiter
    :param n_top:                Number of the most popular tags to take
    :param n_audios_per_shard:   Number of audios per shard, each split has its own shards
    :return:                     A DataFrame contains information of audios
                                 Schema:
                                    <tags>: 0 or 1
                                    clip_id:    clip_id of the original dataset
                                    mp3_path:   A path to a mp3 audio file
                                    split:      A split of dataset (training / validation / test).
                                                The split is determined by its directory (0, 1, ... , f).
                                                First 12 directories (0 ~ b) are used for training,
                                                1 (c) for validation, and 3 (d ~ f) for test.
                                    shard: A shard index of the audio
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
             f"and {n_audios_per_shard} audios per shard.")

    # read csv annotation
    df = pd.read_csv(path, delimiter=delimiter)

    LOG.info(f"Loaded annotations from {path}, "
             f"which contains {df.shape[0]} songs.")

    # get top50 tags
    top50 = df.drop(['clip_id', 'mp3_path'], axis=1)\
        .sum()\
        .sort_values(ascending=False)[:n_top]\
        .index\
        .tolist()

    LOG.info(f"TOP 50 Tags:\n{top50}")

    # remove low frequency tags
    df = df[top50 + ['clip_id', 'mp3_path']]

    # remove songs that have 0 tag
    df = df[df[top50].sum(axis=1) != 0]

    # creating train/val/test splits
    df['split'] = df['mp3_path'].transform(split_by_directory)

    # creating shards for all songs
    for split in ['train', 'val', 'test']:
        n_audios = sum(df['split'] == split)
        n_shards = n_audios // n_audios_per_shard
        n_rest = n_audios % n_audios_per_shard

        LOG.info(f"{split} set size: {n_audios}.")

        shards = np.tile(np.arange(n_shards), n_audios_per_shard)
        shards = np.concatenate([shards, np.arange(n_rest)])
        shards = RNG.permutation(shards)

        df.loc[df['split']==split, 'shard'] = shards

    df['shard'] = df['shard'].astype(int)

    LOG.info(f"Final quantity of songs: {df.shape[0]}\nFinal columns in the DataFrame:\n{df.columns.tolist()}")
    return df


if __name__ == '__main__':
    CONSOLE.rule("Pre-processing MTT Annotations")
    annotations = process_MTT_annotations('./dataset/annotations_final.csv')
    CONSOLE.print(annotations)
