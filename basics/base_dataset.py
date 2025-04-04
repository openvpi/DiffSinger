import os
import pickle

import torch
from torch.utils.data import Dataset

from utils.indexed_datasets import IndexedDataset


class BaseDataset(Dataset):
    """
        Base class for datasets.
        1. *sizes*:
            clipped length if "max_frames" is set;
        2. *num_frames*:
            unclipped length.

        Subclasses should define:
        1. *collate*:
            take the longest data, pad other data to the same length;
        2. *__getitem__*:
            the index function.
    """

    def __init__(self, config, prefix, size_key='lengths', preload=False):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.data_dir = self.config['binary_data_dir']
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
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def collater(self, samples):
        return {
            'size': len(samples),
            'indices': torch.LongTensor([s['_idx'] for s in samples])
        }
