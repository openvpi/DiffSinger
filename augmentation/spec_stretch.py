from copy import deepcopy

import librosa
import numpy as np
import torch

from basics.base_augmentation import BaseAugmentation, require_same_keys
from basics.base_pe import BasePE
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.fastspeech.tts_modules import LengthRegulator
from utils.binarizer_utils import get_mel_torch, get_mel2ph_torch
from utils.hparams import hparams
from utils.infer_utils import resample_align_curve


class SpectrogramStretchAugmentation(BaseAugmentation):
    """
    This class contains methods for frequency-domain and time-domain stretching augmentation.
    """

    def __init__(self, data_dirs: list, augmentation_args: dict, pe: BasePE = None):
        super().__init__(data_dirs, augmentation_args)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.lr = LengthRegulator().to(self.device)
        self.pe = pe

    @require_same_keys
    def process_item(self, item: dict, key_shift=0., speed=1., replace_spk_id=None) -> dict:
        aug_item = deepcopy(item)
        waveform, _ = librosa.load(aug_item['wav_fn'], sr=hparams['audio_sample_rate'], mono=True)
        mel = get_mel_torch(
            waveform, hparams['audio_sample_rate'], num_mel_bins=hparams['audio_num_mel_bins'],
            hop_size=hparams['hop_size'], win_size=hparams['win_size'], fft_size=hparams['fft_size'],
            fmin=hparams['fmin'], fmax=hparams['fmax'], mel_base=hparams['mel_base'],
            keyshift=key_shift, speed=speed, device=self.device
        )

        aug_item['mel'] = mel

        if speed != 1. or hparams['use_speed_embed']:
            aug_item['length'] = mel.shape[0]
            aug_item['speed'] = int(np.round(hparams['hop_size'] * speed)) / hparams['hop_size']  # real speed
            aug_item['seconds'] /= aug_item['speed']
            aug_item['ph_dur'] /= aug_item['speed']
            aug_item['mel2ph'] = get_mel2ph_torch(
                self.lr, torch.from_numpy(aug_item['ph_dur']), aug_item['length'], self.timestep, device=self.device
            ).cpu().numpy()

            f0, _ = self.pe.get_pitch(
                waveform, samplerate=hparams['audio_sample_rate'], length=aug_item['length'],
                hop_size=hparams['hop_size'], f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
                speed=speed, interp_uv=True
            )
            aug_item['f0'] = f0.astype(np.float32)

            # NOTE: variance curves are directly resampled according to speed,
            # despite how frequency-domain features change after the augmentation.
            # For acoustic models, this can bring more (but not much) difficulty
            # to learn how variance curves affect the mel spectrograms, since
            # they must realize how the augmentation causes the mismatch.
            #
            # This is a simple way to combine augmentation and variances. However,
            # dealing variance curves like this will decrease the accuracy of
            # variance controls. In most situations, not being ~100% accurate
            # will not ruin the user experience. For example, it does not matter
            # if the energy does not exactly equal the RMS; it is just fine
            # as long as higher energy can bring higher loudness and strength.
            # The neural networks itself cannot be 100% accurate, though.
            #
            # There are yet other choices to simulate variance curves:
            #   1. Re-extract the features from resampled waveforms;
            #   2. Re-extract the features from re-constructed waveforms using
            #      the transformed mel spectrograms through the vocoder.
            # But there are actually no perfect ways to make them all accurate
            # and stable.
            for v_name in VARIANCE_CHECKLIST:
                if v_name in item:
                    aug_item[v_name] = resample_align_curve(
                        aug_item[v_name],
                        original_timestep=self.timestep,
                        target_timestep=self.timestep * aug_item['speed'],
                        align_length=aug_item['length']
                    )

        if key_shift != 0. or hparams['use_key_shift_embed']:
            if replace_spk_id is None:
                aug_item['key_shift'] = key_shift
            else:
                aug_item['spk_id'] = replace_spk_id
            aug_item['f0'] *= 2 ** (key_shift / 12)

        return aug_item
