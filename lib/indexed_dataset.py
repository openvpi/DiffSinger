import pathlib
from collections import deque

import h5py
import numpy
import torch


class IndexedDataset:
    def __init__(self, path, prefix, num_cache=0):
        self.path = pathlib.Path(path) / f"{prefix}.data.hdf5"
        if not self.path.exists():
            raise FileNotFoundError(f"IndexedDataset not found: {self.path}")
        self._dset = None
        self.cache = deque(maxlen=num_cache)
        self.num_cache = num_cache

    @property
    def dset(self):
        if self._dset is None:
            self._dset = h5py.File(self.path, "r")
        return self._dset

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dset"] = None
        return state

    def __del__(self):
        if self._dset:
            self._dset.close()

    def __getitem__(self, i):
        if i < 0 or i >= len(self.dset):
            raise IndexError("index out of range")
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        item = {}
        for k, v in self.dset[str(i)].items():
            v = v[()]
            if v.ndim == 0:  # scalars saved as numpy.ndarray but loaded with numpy scalar types
                v = torch.tensor(v)
            else:
                v = torch.from_numpy(v)
            item[k] = v
        if self.num_cache > 0:
            self.cache.appendleft((i, item))
        return item

    def __len__(self):
        return len(self.dset)


class IndexedDatasetBuilder:
    def __init__(self, path, prefix, allowed_attr=None, auto_increment=True):
        self.path = pathlib.Path(path) / f"{prefix}.data.hdf5"
        self.prefix = prefix
        self.dset = h5py.File(self.path, "w")
        self.counter = 0
        self.auto_increment = auto_increment
        if allowed_attr is not None:
            self.allowed_attr = set(allowed_attr)
        else:
            self.allowed_attr = None

    def add_item(self, item, item_no=None):
        if self.auto_increment and item_no is not None or not self.auto_increment and item_no is None:
            raise ValueError("auto_increment and provided item_no are mutually exclusive")
        if self.allowed_attr is not None:
            item = {
                k: item[k]
                for k in self.allowed_attr
                if k in item
            }
        for k, v in item.items():
            if not isinstance(v, numpy.ndarray):
                raise TypeError(f"Value type of key '{k}' is not a NumPy array")
        if self.auto_increment:
            item_no = self.counter
            self.counter += 1
        for k, v in item.items():
            if v is None:
                continue
            self.dset.create_dataset(f"{item_no}/{k}", data=v)
        return item_no

    def finalize(self):
        self.dset.close()
