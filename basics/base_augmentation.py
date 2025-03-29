class BaseAugmentation:
    """
    Base class for data augmentation.
    All methods of this class should be thread-safe.
    1. *process_item*:
        Apply augmentation to one piece of data.
    """
    def __init__(self, config, data_dirs: list, augmentation_args: dict):
        self.raw_data_dirs = data_dirs
        self.augmentation_args = augmentation_args
        self.timestep = self.config['hop_size'] / self.config['audio_sample_rate']

    def process_item(self, item: dict, **kwargs) -> dict:
        raise NotImplementedError()


def require_same_keys(func):
    def run(*args, **kwargs):
        item: dict = args[1]
        res: dict = func(*args, **kwargs)
        assert set(item.keys()) == set(res.keys()), 'Item keys mismatch after augmentation.\n' \
                                                    f'Before: {sorted(item.keys())}\n' \
                                                    f'After: {sorted(res.keys())}'
        return res
    return run
