import os

os.environ["LRU_CACHE_CAPACITY"] = "3"
import torch
import torch.utils.data
import numpy as np
import librosa
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F


def load_wav_to_torch(full_path, target_sr=None):
    data, sr = librosa.load(full_path, sr=target_sr, mono=True)
    return torch.from_numpy(data), sr


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


class STFT:
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025,
                 clip_val=1e-5):
        self.target_sr = sr

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False):
        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val

        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))

        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        mel_basis_key = str(fmax) + '_' + str(y.device)
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        keyshift_key = str(keyshift) + '_' + str(y.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)

        y = torch.nn.functional.pad(y.unsqueeze(1),
                                    ((win_size_new - hop_length_new) // 2, (win_size_new - hop_length_new + 1) // 2),
                                    mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(
            y, n_fft_new, hop_length=hop_length_new,
            win_length=win_size_new, window=self.hann_window[keyshift_key],
            center=center, pad_mode='reflect',
            normalized=False, onesided=True, return_complex=True
        ).abs()
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))
            spec = spec[:, :size, :] * win_size / win_size_new

        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)

        return spec

    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(0)).squeeze(0)
        return spect
