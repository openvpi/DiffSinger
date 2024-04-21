import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from librosa.filters import mel as librosa_mel_fn

from basics.base_vocoder import BaseVocoder
from modules.vocoders.registry import register_vocoder
from utils.hparams import hparams


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_model(model_path: pathlib.Path, device='cpu'):
    config_file = model_path.with_name('config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)

    # load model
    print(' [Loading] ' + str(model_path))
    model = torch.jit.load(model_path, map_location=torch.device(device))
    model.eval()

    return model, args


@register_vocoder
class DDSP(BaseVocoder):
    def __init__(self, device='cpu'):
        self.device = device
        model_path = pathlib.Path(hparams['vocoder_ckpt'])
        assert model_path.exists(), 'DDSP model file is not found!'
        self.model, self.args = load_model(model_path, device=self.device)

    def to_device(self, device):
        pass

    def get_device(self):
        return self.device

    def spec2wav_torch(self, mel, f0):  # mel: [B, T, bins] f0: [B, T]
        if self.args.data.sampling_rate != hparams['audio_sample_rate']:
            print('Mismatch parameters: hparams[\'audio_sample_rate\']=', hparams['audio_sample_rate'], '!=',
                  self.args.data.sampling_rate, '(vocoder)')
        if self.args.data.n_mels != hparams['audio_num_mel_bins']:
            print('Mismatch parameters: hparams[\'audio_num_mel_bins\']=', hparams['audio_num_mel_bins'], '!=',
                  self.args.data.n_mels, '(vocoder)')
        if self.args.data.n_fft != hparams['fft_size']:
            print('Mismatch parameters: hparams[\'fft_size\']=', hparams['fft_size'], '!=', self.args.data.n_fft,
                  '(vocoder)')
        if self.args.data.win_length != hparams['win_size']:
            print('Mismatch parameters: hparams[\'win_size\']=', hparams['win_size'], '!=', self.args.data.win_length,
                  '(vocoder)')
        if self.args.data.block_size != hparams['hop_size']:
            print('Mismatch parameters: hparams[\'hop_size\']=', hparams['hop_size'], '!=', self.args.data.block_size,
                  '(vocoder)')
        if self.args.data.mel_fmin != hparams['fmin']:
            print('Mismatch parameters: hparams[\'fmin\']=', hparams['fmin'], '!=', self.args.data.mel_fmin,
                  '(vocoder)')
        if self.args.data.mel_fmax != hparams['fmax']:
            print('Mismatch parameters: hparams[\'fmax\']=', hparams['fmax'], '!=', self.args.data.mel_fmax,
                  '(vocoder)')
        with torch.no_grad():
            mel = mel.to(self.device)
            mel_base = hparams.get('mel_base', 10)
            if mel_base != 'e':
                assert mel_base in [10, '10'], "mel_base must be 'e', '10' or 10."
            else:
                # log mel to log10 mel
                mel = 0.434294 * mel
            f0 = f0.unsqueeze(-1).to(self.device)
            signal, _, (s_h, s_n) = self.model(mel, f0)
            signal = signal.view(-1)
        return signal

    def spec2wav(self, mel, f0):
        if self.args.data.sampling_rate != hparams['audio_sample_rate']:
            print('Mismatch parameters: hparams[\'audio_sample_rate\']=', hparams['audio_sample_rate'], '!=',
                  self.args.data.sampling_rate, '(vocoder)')
        if self.args.data.n_mels != hparams['audio_num_mel_bins']:
            print('Mismatch parameters: hparams[\'audio_num_mel_bins\']=', hparams['audio_num_mel_bins'], '!=',
                  self.args.data.n_mels, '(vocoder)')
        if self.args.data.n_fft != hparams['fft_size']:
            print('Mismatch parameters: hparams[\'fft_size\']=', hparams['fft_size'], '!=', self.args.data.n_fft,
                  '(vocoder)')
        if self.args.data.win_length != hparams['win_size']:
            print('Mismatch parameters: hparams[\'win_size\']=', hparams['win_size'], '!=', self.args.data.win_length,
                  '(vocoder)')
        if self.args.data.block_size != hparams['hop_size']:
            print('Mismatch parameters: hparams[\'hop_size\']=', hparams['hop_size'], '!=', self.args.data.block_size,
                  '(vocoder)')
        if self.args.data.mel_fmin != hparams['fmin']:
            print('Mismatch parameters: hparams[\'fmin\']=', hparams['fmin'], '!=', self.args.data.mel_fmin,
                  '(vocoder)')
        if self.args.data.mel_fmax != hparams['fmax']:
            print('Mismatch parameters: hparams[\'fmax\']=', hparams['fmax'], '!=', self.args.data.mel_fmax,
                  '(vocoder)')
        with torch.no_grad():
            mel = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
            mel_base = hparams.get('mel_base', 10)
            if mel_base != 'e':
                assert mel_base in [10, '10'], "mel_base must be 'e', '10' or 10."
            else:
                # log mel to log10 mel
                mel = 0.434294 * mel
            f0 = torch.FloatTensor(f0).unsqueeze(0).unsqueeze(-1).to(self.device)
            signal, _, (s_h, s_n) = self.model(mel, f0)
            signal = signal.view(-1)
        wav_out = signal.cpu().numpy()
        return wav_out
