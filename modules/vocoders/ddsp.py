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


class Audio2Mel(torch.nn.Module):
    def __init__(
            self,
            hop_length,
            sampling_rate,
            n_mel_channels,
            win_length,
            n_fft=None,
            mel_fmin=0,
            mel_fmax=None,
            clamp=1e-5
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1):
        '''
              audio: B x C x T
        log_mel_spec: B x T_ x C x n_mel 
        '''
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        keyshift_key = str(keyshift) + '_' + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)

        B, C, T = audio.shape
        audio = audio.reshape(B * C, T)
        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=True,
            return_complex=False)
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new

        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=self.clamp))

        # log_mel_spec: B x C, M, T
        T_ = log_mel_spec.shape[-1]
        log_mel_spec = log_mel_spec.reshape(B, C, self.n_mel_channels, T_)
        log_mel_spec = log_mel_spec.permute(0, 3, 1, 2)

        # print('og_mel_spec:', log_mel_spec.shape)
        log_mel_spec = log_mel_spec.squeeze(2)  # mono
        return log_mel_spec


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
            f0 = f0.unsqueeze(-1)
            signal, _, (s_h, s_n) = self.model(mel.to(self.device), f0.to(self.device))
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
            f0 = torch.FloatTensor(f0).unsqueeze(0).unsqueeze(-1).to(self.device)
            signal, _, (s_h, s_n) = self.model(mel.to(self.device), f0.to(self.device))
            signal = signal.view(-1)
        wav_out = signal.cpu().numpy()
        return wav_out
