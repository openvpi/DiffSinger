import numpy as np
import pyworld
import torch
from torch.nn import functional as F


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
