import numpy as np
from basics.base_pe import BasePE
from modules.rmvpe.inference import RMVPE as rmvpe
from utils.pitch_utils import interp_f0

class RMVPE(BasePE):
    def __init__(self, model_path):
        self.rmvpe = rmvpe(model_path, hop_length=160)
        
    def get_pitch(self, waveform, length, hparams, interp_uv=False, speed=1):
        hop_size = int(np.round(hparams['hop_size'] * speed))
        time_step = hop_size / hparams['audio_sample_rate']
        f0 = self.rmvpe.infer_from_audio(waveform, sample_rate=hparams['audio_sample_rate'])
        f0 = np.array([f0[int(min(int(np.round(n * time_step / 0.01)), len(f0) - 1))] for n in range(length)])
        uv = f0 == 0
        if interp_uv:
            f0, uv = interp_f0(f0, uv)
        return f0, uv
