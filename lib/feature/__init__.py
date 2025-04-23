import librosa
import numpy as np
import torch
import torch.nn


def get_energy(waveform, length, *, hop_size, win_size, domain="db"):
    """
    Definition of energy: RMS of the waveform, in dB representation.
    Other energy-based parameters: breathiness (energy of aperiodic part), voicing (energy of harmonic part).
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
    if domain == "db":
        energy = librosa.amplitude_to_db(energy)
    elif domain == "amplitude":
        pass
    else:
        raise ValueError(f"Invalid domain: {domain}")
    return energy


def get_tension(harmonic, base_harmonic, length, *, hop_size, win_size, domain="logit"):
    """
    Definition of tension: radio of the real harmonic part (harmonic part except the base harmonic).
    to the full harmonic part.
    :param harmonic: The full harmonic part
    :param base_harmonic: The base harmonic part
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param win_size: Window size, in number of samples
    :param domain: The domain of the final ratio representation.
     Can be 'ratio' (the raw ratio), 'db' (log decibel) or 'logit' (the reverse function of sigmoid)
    :return: tension
    """
    energy_base_h = get_energy(
        base_harmonic, length,
        hop_size=hop_size, win_size=win_size,
        domain="amplitude"
    )
    energy_h = get_energy(
        harmonic, length,
        hop_size=hop_size, win_size=win_size,
        domain="amplitude"
    )
    tension = np.sqrt(np.clip(energy_h ** 2 - energy_base_h ** 2, a_min=0, a_max=None)) / (energy_h + 1e-5)
    if domain == "ratio":
        tension = np.clip(tension, a_min=0, a_max=1)
    elif domain == "db":
        tension = np.clip(tension, a_min=1e-5, a_max=1)
        tension = librosa.amplitude_to_db(tension)
    elif domain == "logit":
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
            padding="same",
            padding_mode="replicate"
        )
        smooth_kernel = torch.sin(torch.from_numpy(
            np.linspace(0, 1, kernel_size).astype(np.float32) * np.pi
        ))
        smooth_kernel /= smooth_kernel.sum()
        self.weight.data = smooth_kernel[None, None]
