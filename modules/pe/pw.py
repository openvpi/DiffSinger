from basics.base_pe import BasePE
import numpy as np
import pyworld as pw
from utils.pitch_utils import interp_f0


class HarvestPE(BasePE):
    def get_pitch(
            self, waveform, samplerate, length,
            *, hop_size, f0_min=65, f0_max=1100,
            speed=1, interp_uv=False
    ):
        hop_size = int(np.round(hop_size * speed))
        time_step = 1000 * hop_size / samplerate

        f0, _ = pw.harvest(
            waveform.astype(np.float64), samplerate,
            f0_floor=f0_min, f0_ceil=f0_max, frame_period=time_step
        )
        f0 = f0.astype(np.float32)

        if f0.size < length:
            f0 = np.pad(f0, (0, length - f0.size))
        f0 = f0[:length]
        uv = f0 == 0

        if interp_uv:
            f0, uv = interp_f0(f0, uv)
        return f0, uv
