from basics.base_pe import BasePE
from lib.feature.binarizer_utils import get_pitch_parselmouth


class ParselmouthPE(BasePE):
    def get_pitch(
            self,waveform, samplerate, length,
            *, hop_size, f0_min=65, f0_max=1100,
            speed=1, interp_uv=False
    ):
        return get_pitch_parselmouth(
            waveform, samplerate=samplerate, length=length,
            hop_size=hop_size, f0_min=f0_min, f0_max=f0_max,
            speed=speed, interp_uv=interp_uv
        )
