from typing import Union, Dict

import librosa
import numpy as np
import parselmouth
import pyworld as pw
import torch
import torch.nn.functional as F

from utils.pitch_utils import interp_f0


@torch.no_grad()
def get_mel2ph_torch(lr, durs, length, timestep, device='cpu'):
    ph_acc = torch.round(torch.cumsum(durs.to(device), dim=0) / timestep + 0.5).long()
    ph_dur = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(device))
    mel2ph = lr(ph_dur[None])[0]
    num_frames = mel2ph.shape[0]
    if num_frames < length:
        mel2ph = torch.cat((mel2ph, torch.full((length - num_frames,), fill_value=mel2ph[-1], device=device)), dim=0)
    elif num_frames > length:
        mel2ph = mel2ph[:length]
    return mel2ph


def get_pitch_parselmouth(
        waveform, samplerate, length,
        *, hop_size, f0_min=65, f0_max=800,
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


class DecomposedWaveform:
    def __init__(
            self, waveform, samplerate, f0,  # basic parameters
            *,
            hop_size=None, fft_size=None, win_size=None, base_harmonic_radius=3.5,  # analysis parameters
            device=None  # computation parameters
    ):
        # the source components
        self._waveform = waveform
        self._samplerate = samplerate
        self._f0 = f0
        # extraction parameters
        self._hop_size = hop_size
        self._fft_size = fft_size if fft_size is not None else win_size
        self._win_size = win_size if win_size is not None else win_size
        self._time_step = hop_size / samplerate
        self._half_width = base_harmonic_radius
        self._device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # intermediate variables
        self._f0_world = None
        self._sp = None
        self._ap = None
        # final components
        self._harmonic_part: np.ndarray = None
        self._aperiodic_part: np.ndarray = None
        self._harmonics: Dict[int, np.ndarray] = {}

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def hop_size(self):
        return self._hop_size

    @property
    def fft_size(self):
        return self._fft_size

    @property
    def win_size(self):
        return self._win_size

    def _world_extraction(self):
        # Add a tiny noise to the signal to avoid NaN results of D4C in rare edge cases
        # References:
        #   - https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/issues/50
        #   - https://github.com/mmorise/World/issues/116
        x = self._waveform.astype(np.double) + np.random.randn(*self._waveform.shape) * 1e-5
        samplerate = self._samplerate
        f0 = self._f0.astype(np.double)

        hop_size = self._hop_size
        fft_size = self._fft_size

        wav_frames = (x.shape[0] + hop_size - 1) // hop_size
        f0_frames = f0.shape[0]
        if f0_frames < wav_frames:
            f0 = np.pad(f0, (0, wav_frames - f0_frames), mode='constant', constant_values=(f0[0], f0[-1]))
        elif f0_frames > wav_frames:
            f0 = f0[:wav_frames]

        time_step = hop_size / samplerate
        t = np.arange(0, wav_frames) * time_step
        self._f0_world = f0
        self._sp = pw.cheaptrick(x, f0, t, samplerate, fft_size=fft_size)  # extract smoothed spectrogram
        self._ap = pw.d4c(x, f0, t, samplerate, fft_size=fft_size)  # extract aperiodicity

    def _kth_harmonic(self, k: int) -> np.ndarray:
        """
        Extract the Kth harmonic (starting from 0) from the waveform. Author: @yxlllc
        :param k: a non-negative integer
        :return: kth_harmonic float32[T]
        """
        if k in self._harmonics:
            return self._harmonics[k]

        hop_size = self._hop_size
        win_size = self._win_size
        samplerate = self._samplerate
        half_width = self._half_width
        device = self._device

        waveform = torch.from_numpy(self.harmonic()).unsqueeze(0).to(device)  # [B, n_samples]
        n_samples = waveform.shape[1]
        pad_size = (int(n_samples // hop_size) - len(self._f0) + 1) // 2
        f0 = self._f0[pad_size:] * (k + 1)
        f0, _ = interp_f0(f0, uv=f0 == 0)
        f0 = torch.from_numpy(f0).to(device)[None, :, None]  # [B, n_frames, 1]
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
        start = torch.clip(center - half_width, min=0)
        end = torch.clip(center + half_width, max=n_specs)
        idx_mask = (center >= 1) & (idx >= start) & (idx < end)  # [B, n_frames, n_spec]
        if n_f0_frames < n_spec_frames:
            idx_mask = F.pad(idx_mask, [0, 0, 0, n_spec_frames - n_f0_frames])
        spec = spec * idx_mask[:, :n_spec_frames, :]
        self._harmonics[k] = torch.istft(
            spec.permute(0, 2, 1),
            n_fft=win_size,
            win_length=win_size,
            hop_length=hop_size,
            window=nuttall_window,
            center=True,
            length=n_samples
        ).squeeze(0).cpu().numpy()

        return self._harmonics[k]

    def harmonic(self, k: int = None) -> np.ndarray:
        """
        Extract the full harmonic part, or the Kth harmonic if `k` is not None, from the waveform.
        :param k: an integer representing the harmonic index, starting from 0
        :return: full_harmonics float32[T] or kth_harmonic float32[T]
        """
        if k is not None:
            return self._kth_harmonic(k)
        if self._harmonic_part is not None:
            return self._harmonic_part
        if self._sp is None or self._ap is None:
            self._world_extraction()
        self._harmonic_part = pw.synthesize(
            self._f0_world,
            np.clip(self._sp * (1 - self._ap * self._ap), a_min=1e-16, a_max=None),  # clip to avoid zeros
            np.zeros_like(self._ap),
            self._samplerate, frame_period=self._time_step * 1000
        ).astype(np.float32)  # synthesize the harmonic part using the parameters
        return self._harmonic_part

    def aperiodic(self) -> np.ndarray:
        """
        Extract the aperiodic part from the waveform.
        :return: aperiodic_part float32[T]
        """
        if self._aperiodic_part is not None:
            return self._aperiodic_part
        if self._sp is None or self._ap is None:
            self._world_extraction()
        self._aperiodic_part = pw.synthesize(
            self._f0_world, self._sp * self._ap * self._ap, np.ones_like(self._ap),
            self._samplerate, frame_period=self._time_step * 1000
        ).astype(np.float32)  # synthesize the aperiodic part using the parameters
        return self._aperiodic_part


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


def get_breathiness_pyworld(
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
    energy_base_h = get_energy_librosa(
        waveform_base_h, length,
        hop_size=waveform.hop_size, win_size=waveform.win_size,
        domain='amplitude'
    )
    energy_h = get_energy_librosa(
        waveform_h, length,
        hop_size=waveform.hop_size, win_size=waveform.win_size,
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
