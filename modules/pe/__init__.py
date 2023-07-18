from utils import hparams

from .pm import ParselmouthPE


def initialize_pe():
    pe = hparams.get('pe', 'parselmouth')
    if pe == 'parselmouth':
        return ParselmouthPE()
