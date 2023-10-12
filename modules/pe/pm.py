from basics.base_pe import BasePE
from utils.binarizer_utils import get_pitch_parselmouth


class ParselmouthPE(BasePE):
    def get_pitch(self, waveform, length, hparams, interp_uv=False, speed=1):
        return get_pitch_parselmouth(
            waveform, samplerate=hparams['audio_sample_rate'], length=length,
            hop_size=hparams['hop_size'], speed=speed, interp_uv=interp_uv
        )
