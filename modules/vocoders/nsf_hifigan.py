import pathlib

import torch

try:
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ModuleNotFoundError:
    rank_zero_info = print

from modules.nsf_hifigan.models import load_model
from basics.base_vocoder import BaseVocoder
from modules.vocoders.registry import register_vocoder


@register_vocoder
class NsfHifiGAN(BaseVocoder):
    def __init__(self, config):
        self.config = config
        model_path = pathlib.Path(config['vocoder_ckpt'])
        if not model_path.exists():
            raise FileNotFoundError(
                f'NSF-HiFiGAN vocoder model is not found at \'{model_path}\'. '
                'Please follow instructions in docs/BestPractices.md#vocoders to get one.'
            )
        rank_zero_info(f'| Load HifiGAN: {model_path}')
        self.model, self.h = load_model(model_path)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to_device(self, device):
        self.model.to(device)

    def get_device(self):
        return self.device

    def spec2wav_torch(self, mel, **kwargs):  # mel: [B, T, bins]
        if self.h.sampling_rate != self.config['audio_sample_rate']:
            print('Mismatch parameters: config[\'audio_sample_rate\']=', self.config['audio_sample_rate'], '!=',
                  self.h.sampling_rate, '(vocoder)')
        if self.h.num_mels != self.config['audio_num_mel_bins']:
            print('Mismatch parameters: config[\'audio_num_mel_bins\']=', self.config['audio_num_mel_bins'], '!=',
                  self.h.num_mels, '(vocoder)')
        if self.h.n_fft != self.config['fft_size']:
            print('Mismatch parameters: config[\'fft_size\']=', self.config['fft_size'], '!=', self.h.n_fft, '(vocoder)')
        if self.h.win_size != self.config['win_size']:
            print('Mismatch parameters: config[\'win_size\']=', self.config['win_size'], '!=', self.h.win_size,
                  '(vocoder)')
        if self.h.hop_size != self.config['hop_size']:
            print('Mismatch parameters: config[\'hop_size\']=', self.config['hop_size'], '!=', self.h.hop_size,
                  '(vocoder)')
        if self.h.fmin != self.config['fmin']:
            print('Mismatch parameters: config[\'fmin\']=', self.config['fmin'], '!=', self.h.fmin, '(vocoder)')
        if self.h.fmax != self.config['fmax']:
            print('Mismatch parameters: config[\'fmax\']=', self.config['fmax'], '!=', self.h.fmax, '(vocoder)')
        with torch.no_grad():
            c = mel.transpose(2, 1)  # [B, T, bins]
            mel_base = self.config.get('mel_base', 10)
            if mel_base != 'e':
                assert mel_base in [10, '10'], "mel_base must be 'e', '10' or 10."
                # log10 to log mel
                c = 2.30259 * c
            f0 = kwargs.get('f0')  # [B, T]
            if f0 is not None:
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        return y

    def spec2wav(self, mel, **kwargs):
        if self.h.sampling_rate != self.config['audio_sample_rate']:
            print('Mismatch parameters: config[\'audio_sample_rate\']=', self.config['audio_sample_rate'], '!=',
                  self.h.sampling_rate, '(vocoder)')
        if self.h.num_mels != self.config['audio_num_mel_bins']:
            print('Mismatch parameters: config[\'audio_num_mel_bins\']=', self.config['audio_num_mel_bins'], '!=',
                  self.h.num_mels, '(vocoder)')
        if self.h.n_fft != self.config['fft_size']:
            print('Mismatch parameters: config[\'fft_size\']=', self.config['fft_size'], '!=', self.h.n_fft, '(vocoder)')
        if self.h.win_size != self.config['win_size']:
            print('Mismatch parameters: config[\'win_size\']=', self.config['win_size'], '!=', self.h.win_size,
                  '(vocoder)')
        if self.h.hop_size != self.config['hop_size']:
            print('Mismatch parameters: config[\'hop_size\']=', self.config['hop_size'], '!=', self.h.hop_size,
                  '(vocoder)')
        if self.h.fmin != self.config['fmin']:
            print('Mismatch parameters: config[\'fmin\']=', self.config['fmin'], '!=', self.h.fmin, '(vocoder)')
        if self.h.fmax != self.config['fmax']:
            print('Mismatch parameters: config[\'fmax\']=', self.config['fmax'], '!=', self.h.fmax, '(vocoder)')
        with torch.no_grad():
            c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(self.device)
            mel_base = self.config.get('mel_base', 10)
            if mel_base != 'e':
                assert mel_base in [10, '10'], "mel_base must be 'e', '10' or 10."
                # log10 to log mel
                c = 2.30259 * c
            f0 = kwargs.get('f0')
            if f0 is not None:
                f0 = torch.FloatTensor(f0[None, :]).to(self.device)
                y = self.model(c, f0).view(-1)
            else:
                y = self.model(c).view(-1)
        wav_out = y.cpu().numpy()
        return wav_out
