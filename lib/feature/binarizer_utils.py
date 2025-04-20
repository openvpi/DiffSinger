from typing import Union

import librosa
import numpy as np
import parselmouth
import pyworld
import torch
import torch.nn

from lib.feature.decomposed_waveform import DecomposedWaveform
from lib.feature.mel_spec import StretchableMelSpectrogram
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
    wav2mel = StretchableMelSpectrogram(
        samplerate, num_mel_bins, fft_size, win_size, hop_size, fmin, fmax
    ).to(device)
    with torch.no_grad():
        wav_torch = torch.from_numpy(waveform).to(device)
        mel_torch = wav2mel(wav_torch.unsqueeze(0), key_shift=keyshift, speed=speed).squeeze(0).T
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
