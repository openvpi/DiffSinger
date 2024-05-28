from typing import Dict

import numpy as np
import pyworld as pw
import torch
from torch.nn import functional as F

from modules.hnsep.vr import load_sep_model
from utils import hparams
from utils.pitch_utils import interp_f0


class DecomposedWaveform:
    def __new__(
            cls, waveform, samplerate, f0,
            *,
            hop_size=None, fft_size=None, win_size=None,
            algorithm='world', device=None
    ):
        if algorithm == 'world':
            obj = object.__new__(DecomposedWaveformPyWorld)
            # noinspection PyProtectedMember
            obj._init(
                waveform=waveform, samplerate=samplerate, f0=f0,
                hop_size=hop_size, fft_size=fft_size, win_size=win_size,
                device=device
            )
        elif algorithm == 'vr':
            obj = object.__new__(DecomposedWaveformVocalRemover)
            hnsep_ckpt = hparams['hnsep_ckpt']
            # noinspection PyProtectedMember
            obj._init(
                waveform=waveform, samplerate=samplerate, f0=f0, hop_size=hop_size,
                fft_size=fft_size, win_size=win_size, model_path=hnsep_ckpt,
                device=device
            )
        else:
            raise ValueError(f" [x] Unknown harmonic-noise separator: {algorithm}")
        return obj

    @property
    def samplerate(self):
        raise NotImplementedError()

    @property
    def hop_size(self):
        raise NotImplementedError()

    @property
    def fft_size(self):
        raise NotImplementedError()

    @property
    def win_size(self):
        raise NotImplementedError()

    def harmonic(self, k: int = None) -> np.ndarray:
        raise NotImplementedError()

    def aperiodic(self) -> np.ndarray:
        raise NotImplementedError()


class DecomposedWaveformPyWorld(DecomposedWaveform):
    def _init(
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
        # noinspection PyAttributeOutsideInit
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
        # noinspection PyAttributeOutsideInit
        self._aperiodic_part = pw.synthesize(
            self._f0_world, self._sp * self._ap * self._ap, np.ones_like(self._ap),
            self._samplerate, frame_period=self._time_step * 1000
        ).astype(np.float32)  # synthesize the aperiodic part using the parameters
        return self._aperiodic_part


SEP_MODEL = None


class DecomposedWaveformVocalRemover(DecomposedWaveformPyWorld):
    def _init(
            self, waveform, samplerate, f0,
            hop_size=None, fft_size=None, win_size=None, base_harmonic_radius=3.5,
            model_path=None, device=None
    ):
        super()._init(
            waveform, samplerate, f0, hop_size=hop_size, fft_size=fft_size,
            win_size=win_size, base_harmonic_radius=base_harmonic_radius, device=device
        )
        global SEP_MODEL
        if SEP_MODEL is None:
            SEP_MODEL = load_sep_model(model_path, self._device)
        self.sep_model = SEP_MODEL

    def _infer(self):
        with torch.no_grad():
            x = torch.from_numpy(self._waveform).to(self._device).reshape(1, 1, -1)
            if not self.sep_model.is_mono:
                x = x.repeat(1, 2, 1)
            x = self.sep_model.predict_from_audio(x)
            x = torch.mean(x, dim=1)
            self._harmonic_part = x.squeeze().cpu().numpy()
            self._aperiodic_part = self._waveform - self._harmonic_part

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
        self._infer()
        return self._harmonic_part

    def aperiodic(self) -> np.ndarray:
        """
        Extract the aperiodic part from the waveform.
        :return: aperiodic_part float32[T]
        """
        if self._aperiodic_part is not None:
            return self._aperiodic_part
        self._infer()
        return self._aperiodic_part
