from basics.base_pe import BasePE
from utils.binarizer_utils import get_pitch_parselmouth


class ParselmouthPE(BasePE):
    def get_pitch(self, waveform, length, hparams, interp_uv=False, speed=1):
        return get_pitch_parselmouth(waveform, length, hparams, speed=speed, interp_uv=interp_uv)
