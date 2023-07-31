import pathlib
import multiprocessing
from collections import deque

import h5py
import torch
import numpy as np


class IndexedDataset:
    def __init__(self, path, prefix, num_cache=0):
        super().__init__()
        self.path = pathlib.Path(path) / f'{prefix}.data'
        if not self.path.exists():
            raise FileNotFoundError(f'IndexedDataset not found: {self.path}')
        self.dset = None
        self.cache = deque(maxlen=num_cache)
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.dset):
            raise IndexError('index out of range')

    def __del__(self):
        if self.dset:
            self.dset.close()

    def __getitem__(self, i):
        if self.dset is None:
            self.dset = h5py.File(self.path, 'r')
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        item = {k: v[()].item() if v.shape == () else torch.from_numpy(v[()]) for k, v in self.dset[str(i)].items()}
        if self.num_cache > 0:
            self.cache.appendleft((i, item))
        return item

    def __len__(self):
        if self.dset is None:
            self.dset = h5py.File(self.path, 'r')
        return len(self.dset)


class IndexedDatasetBuilder:
    def __init__(self, path, prefix, allowed_attr=None):
        self.path = pathlib.Path(path) / f'{prefix}.data'
        self.prefix = prefix
        self.dset = None
        self.counter = 0
        self.lock = multiprocessing.Lock()
        if allowed_attr is not None:
            self.allowed_attr = set(allowed_attr)
        else:
            self.allowed_attr = None

    def add_item(self, item):
        if self.dset is None:
            self.dset = h5py.File(self.path, 'w')
        if self.allowed_attr is not None:
            item = {
                k: item[k]
                for k in self.allowed_attr
                if k in item
            }
        item_no = self.counter
        self.counter += 1
        for k, v in item.items():
            if v is None:
                continue
            self.dset.create_dataset(f'{item_no}/{k}', data=v)

    def finalize(self):
        if self.dset is not None:
            self.dset.close()


if __name__ == "__main__":
    import random
    from tqdm import tqdm

    ds_path = './checkpoints/indexed_ds_example'
    size = 100
    items = [{"a": np.random.normal(size=[10000, 10]),
              "b": np.random.normal(size=[10000, 10])} for i in range(size)]
    builder = IndexedDatasetBuilder(ds_path, 'example')
    for i in tqdm(range(size)):
        builder.add_item(items[i])
    builder.finalize()
    ds = IndexedDataset(ds_path, 'example')
    for i in tqdm(range(10000)):
        idx = random.randint(0, size - 1)
        assert (ds[idx]['a'] == items[idx]['a']).all()
