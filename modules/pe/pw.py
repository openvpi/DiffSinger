from basics.base_pe import BasePE
import numpy as np
import pyworld as pw
from utils.pitch_utils import interp_f0

def pad_frames(frames, hop_size, n_samples, n_expect):
    n_frames = frames.shape[0]
    lpad = (int(n_samples // hop_size) - n_frames + 1) // 2
    rpad = n_expect - n_frames - lpad
    if rpad < 0:
        frames = frames[:rpad]
        rpad = 0
    if lpad > 0 or rpad > 0:
        frames = np.pad(frames, (lpad, rpad), mode='constant', constant_values=(frames[0], frames[-1]))
    return frames

class HarvestPE(BasePE):
    def get_pitch(self, waveform, length, hparams, interp_uv=False, speed=1):
        hop_size = int(np.round(hparams['hop_size'] * speed))

        time_step = 1000 * hop_size / hparams['audio_sample_rate']
        f0_floor = hparams['f0_min']
        f0_ceil = hparams['f0_max']

        f0, _ = pw.harvest(waveform.astype(np.float64), hparams['audio_sample_rate'], f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=time_step)
        f0 = pad_frames(f0.astype(np.float32), hop_size, waveform.shape[0], length)
        uv = f0 == 0

        if interp_uv:
            f0, uv = interp_f0(f0, uv)
        return f0, uv
    
class DioPE(BasePE):
    def get_pitch(self, waveform, length, hparams, interp_uv=False, speed=1):
        hop_size = int(np.round(hparams['hop_size'] * speed))

        time_step = 1000 * hop_size / hparams['audio_sample_rate']
        f0_floor = hparams['f0_min']
        f0_ceil = hparams['f0_max']

        wav64 = waveform.astype(np.float64)
        f0, t = pw.dio(wav64, hparams['audio_sample_rate'], f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=time_step)
        f0 = pw.stonemask(wav64, f0, t, hparams['audio_sample_rate']).astype(np.float32)
        f0 = pad_frames(f0, hop_size, waveform.shape[0], length)
        uv = f0 == 0

        if interp_uv:
            f0, uv = interp_f0(f0, uv)
        return f0, uv
