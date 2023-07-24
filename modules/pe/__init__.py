from utils import hparams

from .pm import ParselmouthPE
from .rmvpe import RMVPE

def initialize_pe():
    pe = hparams.get('pe', 'parselmouth')
    pe_ckpt = hparams.get('pe_ckpt', '')
    if pe == 'parselmouth':
        return ParselmouthPE()
    elif pe == 'rmvpe':
        return RMVPE(pe_ckpt)
    else:
        raise ValueError(f" [x] Unknown f0 extractor: {pe}")