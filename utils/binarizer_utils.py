import librosa
import numpy as np
import parselmouth
import pyworld as pw
import torch

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


def pad_frames(frames, hop_size, n_samples, n_expect):
    n_frames = frames.shape[0]
    lpad = (int(n_samples // hop_size) - n_frames + 1) // 2
    rpad = n_expect - n_frames - lpad
    if rpad < 0:
        frames = frames[:rpad]
        rpad = 0
    if lpad > 0 or rpad > 0:
        frames = np.pad(frames, (lpad, rpad), mode='constant', constant_values=(frames[0], frames[-1]))
    return frames


def get_pitch_parselmouth(wav_data, length, hparams, speed=1, interp_uv=False):
    """

    :param wav_data: [T]
    :param length: Expected number of frames
    :param hparams:
    :param speed: Change the speed
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, uv
    """
    hop_size = int(np.round(hparams['hop_size'] * speed))

    time_step = hop_size / hparams['audio_sample_rate']
    f0_min = 65
    f0_max = 800

    # noinspection PyArgumentList
    f0 = parselmouth.Sound(wav_data, sampling_frequency=hparams['audio_sample_rate']).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max
    ).selected_array['frequency'].astype(np.float32)
    f0 = pad_frames(f0, hop_size, wav_data.shape[0], length)
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv


def get_energy_librosa(wav_data, length, hparams):
    """

    :param wav_data: [T]
    :param length: Expected number of frames
    :param hparams:
    :return: energy
    """
    hop_size = hparams['hop_size']
    win_size = hparams['win_size']

    energy = librosa.feature.rms(y=wav_data, frame_length=win_size, hop_length=hop_size)[0]
    energy = pad_frames(energy, hop_size, wav_data.shape[0], length)
    energy_db = librosa.amplitude_to_db(energy)
    return energy_db


def get_breathiness_pyworld(wav_data, f0, length, hparams):
    """

    :param wav_data: [T]
    :param f0: reference f0
    :param length: Expected number of frames
    :param hparams:
    :return: breathiness
    """
    sample_rate = hparams['audio_sample_rate']
    hop_size = hparams['hop_size']
    fft_size = hparams['fft_size']

    x = wav_data.astype(np.double)
    f0 = f0.astype(np.double)
    wav_frames = (x.shape[0] + hop_size - 1) // hop_size
    f0_frames = f0.shape[0]
    if f0_frames < wav_frames:
        f0 = np.pad(f0, (0, wav_frames - f0_frames), mode='constant', constant_values=(f0[0], f0[-1]))
    elif f0_frames > wav_frames:
        f0 = f0[:wav_frames]

    time_step = hop_size / sample_rate
    t = np.arange(0, wav_frames) * time_step
    sp = pw.cheaptrick(x, f0, t, sample_rate, fft_size=fft_size)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, sample_rate, fft_size=fft_size)  # extract aperiodicity
    y = pw.synthesize(
        f0, sp * ap * ap, np.ones_like(ap), sample_rate,
        frame_period=time_step * 1000
    ).astype(np.float32)  # synthesize the aperiodic part using the parameters
    breathiness = get_energy_librosa(y, length, hparams)
    return breathiness


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
