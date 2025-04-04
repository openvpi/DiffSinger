from .pm import ParselmouthPE
from .pw import HarvestPE
from .rmvpe import RMVPE


def initialize_pe(config):
    pe = config['pe']
    pe_ckpt = config['pe_ckpt']
    if pe == 'parselmouth':
        return ParselmouthPE()
    elif pe == 'rmvpe':
        return RMVPE(pe_ckpt)
    elif pe == 'harvest':
        return HarvestPE()
    else:
        raise ValueError(f" [x] Unknown f0 extractor: {pe}")
