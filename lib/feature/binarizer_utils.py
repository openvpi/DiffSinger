from typing import Union

import librosa
import numpy as np
import parselmouth
import pyworld
import torch
from torch.nn import functional as F

from lib.feature.decomposed_waveform import DecomposedWaveform
from lib.feature.nvSTFT import STFT
from utils.pitch_utils import interp_f0


def get_mel_torch(
        waveform, samplerate,
        *,
        num_mel_bins=128, hop_size=512, win_size=2048, fft_size=2048,
        fmin=40, fmax=16000,
        keyshift=0, speed=1, device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stft = STFT(samplerate, num_mel_bins, fft_size, win_size, hop_size, fmin, fmax, device=device)
    with torch.no_grad():
        wav_torch = torch.from_numpy(waveform).to(device)
        mel_torch = stft.get_mel(wav_torch.unsqueeze(0), keyshift=keyshift, speed=speed).squeeze(0).T
        return mel_torch.cpu().numpy()


def get_pitch_parselmouth(
        waveform, samplerate, length,
        *, hop_size, f0_min=65, f0_max=1100,
        speed=1, interp_uv=False
):
    """

    :param waveform: [T]
    :param samplerate: sampling rate
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param f0_min: Minimum f0 in Hz
    :param f0_max: Maximum f0 in Hz
    :param speed: Change the speed
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, uv
    """
    hop_size = int(np.round(hop_size * speed))
    time_step = hop_size / samplerate

    l_pad = int(np.ceil(1.5 / f0_min * samplerate))
    r_pad = hop_size * ((len(waveform) - 1) // hop_size + 1) - len(waveform) + l_pad + 1
    waveform = np.pad(waveform, (l_pad, r_pad))

    # noinspection PyArgumentList
    s = parselmouth.Sound(waveform, sampling_frequency=samplerate).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max
    )
    assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
    f0 = s.selected_array['frequency'].astype(np.float32)
    if len(f0) < length:
        f0 = np.pad(f0, (0, length - len(f0)))
    f0 = f0[: length]
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv


def get_pitch_harvest(
        waveform, samplerate, length,
        *, hop_size, f0_min=65, f0_max=1100,
        speed=1, interp_uv=False
):
    hop_size = int(np.round(hop_size * speed))
    time_step = 1000 * hop_size / samplerate

    f0, _ = pyworld.harvest(
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


def get_energy_librosa(waveform, length, *, hop_size, win_size, domain='db'):
    """
    Definition of energy: RMS of the waveform, in dB representation
    :param waveform: [T]
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param win_size: Window size, in number of samples
    :param domain: db or amplitude
    :return: energy
    """
    energy = librosa.feature.rms(y=waveform, frame_length=win_size, hop_length=hop_size)[0]
    if len(energy) < length:
        energy = np.pad(energy, (0, length - len(energy)))
    energy = energy[: length]
    if domain == 'db':
        energy = librosa.amplitude_to_db(energy)
    elif domain == 'amplitude':
        pass
    else:
        raise ValueError(f'Invalid domain: {domain}')
    return energy


def get_tension(harmonic, base_harmonic, length, *, hop_size, win_size, domain="logit"):
    energy_base_h = get_energy_librosa(
        base_harmonic, length,
        hop_size=hop_size, win_size=win_size,
        domain='amplitude'
    )
    energy_h = get_energy_librosa(
        harmonic, length,
        hop_size=hop_size, win_size=win_size,
        domain='amplitude'
    )
    tension = np.sqrt(np.clip(energy_h ** 2 - energy_base_h ** 2, a_min=0, a_max=None)) / (energy_h + 1e-5)
    if domain == 'ratio':
        tension = np.clip(tension, a_min=0, a_max=1)
    elif domain == 'db':
        tension = np.clip(tension, a_min=1e-5, a_max=1)
        tension = librosa.amplitude_to_db(tension)
    elif domain == 'logit':
        tension = np.clip(tension, a_min=1e-4, a_max=1 - 1e-4)
        tension = np.log(tension / (1 - tension))
    return tension


def world_analyze(waveform, f0, *, samplerate, hop_size, fft_size) -> tuple[np.ndarray, np.ndarray]:  # [sp, ap]
    # Add a tiny noise to the signal to avoid NaN results of D4C in rare edge cases
    # References:
    #   - https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/issues/50
    #   - https://github.com/mmorise/World/issues/116
    x = waveform.astype(np.double) + np.random.randn(*waveform.shape) * 1e-5
    f0 = f0.astype(np.double)

    wav_frames = (x.shape[0] + hop_size - 1) // hop_size
    f0_frames = f0.shape[0]
    if f0_frames < wav_frames:
        f0 = np.pad(f0, (0, wav_frames - f0_frames), mode="edge")
    elif f0_frames > wav_frames:
        f0 = f0[:wav_frames]

    time_step = hop_size / samplerate
    t = np.arange(0, wav_frames) * time_step
    sp = pyworld.cheaptrick(x, f0, t, samplerate, fft_size=fft_size)  # extract smoothed spectrogram
    ap = pyworld.d4c(x, f0, t, samplerate, fft_size=fft_size)  # extract aperiodicity
    return sp, ap


def world_synthesize(f0, sp, ap, *, samplerate, time_step) -> np.ndarray:
    f0 = f0.astype(np.double)
    f0_frames = f0.shape[0]
    sp_frames = sp.shape[0]
    if f0_frames < sp_frames:
        f0 = np.pad(f0, (0, sp_frames - f0_frames), mode="edge")
    elif f0_frames > sp_frames:
        f0 = f0[:sp_frames]
    waveform = pyworld.synthesize(
        f0, sp, ap,
        samplerate, frame_period=time_step * 1000
    ).astype(np.float32)
    return waveform


def world_synthesize_harmonics(f0, sp, ap, *, samplerate, time_step) -> np.ndarray:
    return world_synthesize(
        f0,
        np.clip(sp * (1 - ap * ap), a_min=1e-16, a_max=None),  # clip to avoid zeros
        np.zeros_like(ap),
        samplerate=samplerate, time_step=time_step
    )  # synthesize the harmonic part using the parameters


def world_synthesize_aperiodic(f0, sp, ap, *, samplerate, time_step) -> np.ndarray:
    return world_synthesize(
        f0, sp * ap * ap, np.ones_like(ap),
        samplerate=samplerate, time_step=time_step
    )  # synthesize the harmonic part using the parameters


def get_kth_harmonic(waveform, f0, k: int, *, samplerate, hop_size, win_size, kth_harmonic_radius=3.5, device="cpu"):
    batched = waveform.ndim > 1
    if not batched:
        waveform = waveform[None]
        f0 = f0[None]
    waveform = torch.from_numpy(waveform).to(device)  # [B, n_samples]
    n_samples = waveform.shape[1]
    f0 = f0 * (k + 1)
    pad_size = int(n_samples // hop_size) - len(f0) + 1
    if pad_size > 0:
        f0 = np.pad(f0, ((0, 0), (0, pad_size)), mode="edge")

    f0 = torch.from_numpy(f0).to(device)[..., None]  # [B, n_frames, 1]
    n_f0_frames = f0.shape[1]

    phase = torch.arange(win_size, dtype=waveform.dtype, device=device) / win_size * 2 * np.pi
    nuttall_window = (
            0.355768
            - 0.487396 * torch.cos(phase)
            + 0.144232 * torch.cos(2 * phase)
            - 0.012604 * torch.cos(3 * phase)
    )
    spec = torch.stft(
        waveform,
        n_fft=win_size,
        win_length=win_size,
        hop_length=hop_size,
        window=nuttall_window,
        center=True,
        return_complex=True
    ).permute(0, 2, 1)  # [B, n_frames, n_spec]
    n_spec_frames, n_specs = spec.shape[1:]
    idx = torch.arange(n_specs).unsqueeze(0).unsqueeze(0).to(f0)  # [1, 1, n_spec]
    center = f0 * win_size / samplerate
    start = torch.clip(center - kth_harmonic_radius, min=0)
    end = torch.clip(center + kth_harmonic_radius, max=n_specs)
    idx_mask = (center >= 1) & (idx >= start) & (idx < end)  # [B, n_frames, n_spec]
    if n_f0_frames < n_spec_frames:
        idx_mask = F.pad(idx_mask, [0, 0, 0, n_spec_frames - n_f0_frames])
    spec = spec * idx_mask[:, :n_spec_frames, :]
    kth_harmonic = torch.istft(
        spec.permute(0, 2, 1),
        n_fft=win_size,
        win_length=win_size,
        hop_length=hop_size,
        window=nuttall_window,
        center=True,
        length=n_samples
    ).cpu().numpy()
    if batched:
        return kth_harmonic
    else:
        return kth_harmonic.squeeze(0)


def get_breathiness(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None
):
    """
    Definition of breathiness: RMS of the aperiodic part, in dB representation
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :return: breathiness
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_ap = waveform.aperiodic()
    breathiness = get_energy_librosa(
        waveform_ap, length=length,
        hop_size=waveform.hop_size, win_size=waveform.win_size
    )
    return breathiness


def get_voicing(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None
):
    """
    Definition of voicing: RMS of the harmonic part, in dB representation
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :return: voicing
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_sp = waveform.harmonic()
    voicing = get_energy_librosa(
        waveform_sp, length=length,
        hop_size=waveform.hop_size, win_size=waveform.win_size
    )
    return voicing


def get_tension_base_harmonic(
        waveform: Union[np.ndarray, DecomposedWaveform],
        samplerate, f0, length,
        *, hop_size=None, fft_size=None, win_size=None,
        domain='logit'
):
    """
    Definition of tension: radio of the real harmonic part (harmonic part except the base harmonic)
    to the full harmonic part.
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :param domain: The domain of the final ratio representation.
     Can be 'ratio' (the raw ratio), 'db' (log decibel) or 'logit' (the reverse function of sigmoid)
    :return: tension
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0,
            hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_h = waveform.harmonic()
    waveform_base_h = waveform.harmonic(0)
    tension = get_tension(
        waveform_h, waveform_base_h, length,
        hop_size=waveform.hop_size, win_size=waveform.win_size, domain=domain
    )
    return tension


class SinusoidalSmoothingConv1d(torch.nn.Conv1d):
    def __init__(self, kernel_size):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=False,
            padding='same',
            padding_mode='replicate'
        )
        smooth_kernel = torch.sin(torch.from_numpy(
            np.linspace(0, 1, kernel_size).astype(np.float32) * np.pi
        ))
        smooth_kernel /= smooth_kernel.sum()
        self.weight.data = smooth_kernel[None, None]
