import numpy as np
from utils import hparams
from .pw import DecomposedWaveformPyWorld
from .vr import DecomposedWaveformVocalRemover

class DecomposedWaveform:
    def __init__(self, waveform, samplerate, f0, *, hop_size=None, fft_size=None, win_size=None):
        hnsep = hparams['hnsep']
        hnsep_ckpt = hparams['hnsep_ckpt']
        if hnsep == 'world':
            self.dec_waveform = DecomposedWaveformPyWorld(
                waveform=waveform, samplerate=samplerate, f0=f0,
                hop_size=hop_size, fft_size=fft_size, win_size=win_size
            )
        elif hnsep == 'vr':
            self.dec_waveform = DecomposedWaveformVocalRemover(
                waveform=waveform, samplerate=samplerate, f0=f0, model_path=hnsep_ckpt,
                hop_size=hop_size, fft_size=fft_size, win_size=win_size
            )            
        else:
            raise ValueError(f" [x] Unknown harmonic-noise separator: {hnsep}")

    @property
    def samplerate(self):
        return self.dec_waveform.samplerate

    @property
    def hop_size(self):
        return self.dec_waveform.hop_size

    @property
    def fft_size(self):
        return self.dec_waveform.fft_size

    @property
    def win_size(self):
        return self.dec_waveform.win_size
        
    def harmonic(self, k: int = None) -> np.ndarray:
        return self.dec_waveform.harmonic(k)
    
    def aperiodic(self) -> np.ndarray:
        return self.dec_waveform.aperiodic()
    
