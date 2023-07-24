import numpy as np
from basics.base_pe import BasePE
from modules.rmvpe.inference import RMVPE as rmvpe
from utils.pitch_utils import interp_f0
from utils.infer_utils import resample_align_curve

class RMVPE(BasePE):
    def __init__(self, model_path):
        self.rmvpe = rmvpe(model_path, hop_length=160)
        
    def get_pitch(self, waveform, length, hparams, interp_uv=False, speed=1):
        f0 = self.rmvpe.infer_from_audio(waveform, sample_rate=hparams['audio_sample_rate'])
        uv = f0 == 0
        f0, uv = interp_f0(f0, uv)
        
        hop_size = int(np.round(hparams['hop_size'] * speed))
        time_step = hop_size / hparams['audio_sample_rate']
        f0_res = resample_align_curve(f0, 0.01, time_step, length)
        uv_res = resample_align_curve(uv.astype(float), 0.01, time_step, length) > 0.5
        if not interp_uv:
            f0_res[uv_res] = 0
        return f0_res, uv_res
