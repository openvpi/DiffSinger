import os
import pickle

import torch
from torch.utils.data import Dataset

from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset


class BaseDataset(Dataset):
    """Base class for indexed datasets.

    ``sizes`` is the metadata sequence selected by ``size_key``.
    ``num_frames(index)`` and ``size(index)`` return ``sizes[index]``.

    Override ``collater`` for task-specific padding and batching.
    """

    def __init__(self, prefix, size_key='lengths', preload=False):
        super().__init__()
        self.prefix = prefix
        self.data_dir = hparams['binary_data_dir']
        with open(os.path.join(self.data_dir, f'{self.prefix}.meta'), 'rb') as f:
            self.metadata = pickle.load(f)
        self.sizes = self.metadata[size_key]
        self._indexed_ds = IndexedDataset(self.data_dir, self.prefix)
        if preload:
            self.indexed_ds = [self._indexed_ds[i] for i in range(len(self._indexed_ds))]
            del self._indexed_ds
        else:
            self.indexed_ds = self._indexed_ds

    def __getitem__(self, index):
        return {'_idx': index, **self.indexed_ds[index]}

    def __len__(self):
        return len(self.sizes)

    def num_frames(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    def collater(self, samples):
        return {
            'size': len(samples),
            'indices': torch.LongTensor([s['_idx'] for s in samples])
        }
