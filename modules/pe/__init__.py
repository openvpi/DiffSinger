from functools import lru_cache

import torch

from basics.base_pe import BasePE

from .pm import ParselmouthPE
from .pw import HarvestPE
from .rmvpe import RMVPE


# Kept at module level instead of on the binarizer instances: pickling torch tensors to
# worker processes goes through shared memory, which fails on Windows.
@lru_cache(maxsize=2)
def _get_pitch_extractor(pe: str, pe_ckpt: str, device: torch.device) -> BasePE:
    """Build and cache the pitch extractor for the given configuration."""
    if pe == 'parselmouth':
        return ParselmouthPE()
    elif pe == 'rmvpe':
        return RMVPE(pe_ckpt, device=device)
    elif pe == 'harvest':
        return HarvestPE()
    else:
        raise ValueError(f" [x] Unknown f0 extractor: {pe}")


def get_pitch_extractor(pe: str, pe_ckpt: str, device=None) -> BasePE:
    """Resolve the device before looking up the cached pitch extractor."""
    if pe != 'rmvpe':
        device = 'cpu'
    elif device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return _get_pitch_extractor(pe, pe_ckpt, device)
