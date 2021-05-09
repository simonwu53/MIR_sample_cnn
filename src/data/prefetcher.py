import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Union, List, Optional


def _to_torch(val: Union[np.ndarray, List[float], int], device: Union[str, torch.device]) -> torch.Tensor:
    if isinstance(val, np.ndarray):
        torch_val = torch.from_numpy(val).cuda(device=device)
    else:
        torch_val = torch.tensor(val, dtype=torch.float32).cuda(device=device)
    return torch_val


class DataPrefetcher:
    def __init__(self, loader: DataLoader, device: Union[str, torch.device],
                 mean: Optional[Union[np.ndarray, List[float], int]] = None,
                 std: Optional[Union[np.ndarray, List[float], int]] = None):
        """
        Boost up data loading while training the model, hit the maximum usage of gpu
        re

        :param loader: PyTorch DataLoader for training (should be an iterable)
        :param mean:   mean value for the data if need to normalization
        :param std:    std value for the data if need to normalization
        """
        self.device = device
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device=device)

        if mean is not None and std is not None:
            self.mean = _to_torch(mean, device)
            self.std = _to_torch(std, device)
        else:
            self.mean = None
            self.std = None

        # preload
        self.next_input = None
        self.next_target = None
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

            # self.next_input = self.next_input.float()
            if self.mean is not None:
                self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        inputs = self.next_input
        target = self.next_target
        self.preload()
        return inputs, target
