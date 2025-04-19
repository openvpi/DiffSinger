from typing import Dict

import numpy as np
import torch

from lib.feature.binarizer_utils import (
    world_analyze,
    world_synthesize_harmonics,
    world_synthesize_aperiodic,
    get_kth_harmonic
)
from modules.vr import load_sep_model
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

    def waveform(self) -> np.ndarray:
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
        self._sp, self._ap = world_analyze(
            self._waveform, self._f0,
            samplerate=self._samplerate, hop_size=self._hop_size, fft_size=self._fft_size
        )

    def _kth_harmonic(self, k: int) -> np.ndarray:
        """
        Extract the Kth harmonic (starting from 0) from the waveform. Author: @yxlllc
        :param k: a non-negative integer
        :return: kth_harmonic float32[T]
        """
        if k in self._harmonics:
            return self._harmonics[k]

        f0, _ = interp_f0(self._f0, uv=self._f0 == 0)
        self._harmonics[k] = get_kth_harmonic(
            self.harmonic(), f0, k,
            samplerate=self._samplerate, hop_size=self._hop_size, win_size=self._win_size,
            kth_harmonic_radius=self._half_width, device=self._device
        )

        return self._harmonics[k]

    def waveform(self) -> np.ndarray:
        """
        Get the original waveform.
        :return: waveform float32[T]
        """
        return self._waveform

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
        self._harmonic_part = world_synthesize_harmonics(
            self._f0, self._sp, self._ap,
            samplerate=self._samplerate, time_step=self._time_step
        )
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
        self._aperiodic_part = world_synthesize_aperiodic(
            self._f0, self._sp, self._ap,
            samplerate=self._samplerate, time_step=self._time_step
        )
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
