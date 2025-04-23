import numpy as np
import parselmouth
import pyworld


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


def norm_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = np.log2(f0 + uv)  # avoid arithmetic error
    f0[uv] = -np.inf
    return f0


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0, uv)
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0
