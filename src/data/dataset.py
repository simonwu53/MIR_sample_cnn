import numpy as np
from torch.utils.data.dataset import Dataset
from pathlib import Path
from typing import Literal, Tuple

SPLIT = Literal['train', 'val', 'test']


class MTTDataset(Dataset):
    def __init__(self, path: str, split: SPLIT):
        self.root = Path(path).absolute()
        self.dir = self.root.joinpath(split)
        self.files = list(self.dir.glob('clip-*-seg-*-of-*.npz'))
        return

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        file = np.load(self.files[idx])
        label = file['labels'].astype(np.float32)
        sample = file['data'].reshape(1, -1).astype(np.float32)
        return sample, label

    def __len__(self) -> int:
        return len(self.files)

    def calc_steps(self, bs: int) -> int:
        return self.__len__()//bs


if __name__ == '__main__':
    dataset = MTTDataset('./dataset/processed', 'train')
    print(dataset[50])
