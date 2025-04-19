import os

os.environ["LRU_CACHE_CAPACITY"] = "3"
import torch
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


class STFT:
    def __init__(
            self, sr=22050,
            n_mels=80, n_fft=1024, win_size=1024, hop_length=256,
            fmin=20, fmax=11025, clip_val=1e-5,
            device=None
    ):
        self.target_sr = sr

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.mel_basis = torch.from_numpy(mel_basis).float().to(device)

    def get_mel(self, y, keyshift=0, speed=1, center=False):

        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        window = torch.hann_window(win_size_new, device=self.device)

        y = torch.nn.functional.pad(y.unsqueeze(1), (
            (win_size_new - hop_length_new) // 2,
            (win_size_new - hop_length_new + 1) // 2
        ), mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(
            y, n_fft_new, hop_length=hop_length_new,
            win_length=win_size_new, window=window,
            center=center, pad_mode='reflect',
            normalized=False, onesided=True, return_complex=True
        ).abs()
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * self.win_size / win_size_new

        spec = torch.matmul(self.mel_basis, spec)
        spec = dynamic_range_compression_torch(spec, clip_val=self.clip_val)

        return spec
