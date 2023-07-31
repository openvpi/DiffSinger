import librosa
import numpy as np
import torch

from .constants import *


def to_local_average_f0(hidden, center=None, thred=0.03):
    idx = torch.arange(N_CLASS, device=hidden.device)[None, None, :]  # [B=1, T=1, N]
    idx_cents = idx * 20 + CONST  # [B=1, N]
    if center is None:
        center = torch.argmax(hidden, dim=2, keepdim=True)  # [B, T, 1]
    start = torch.clip(center - 4, min=0)  # [B, T, 1]
    end = torch.clip(center + 5, max=N_CLASS)  # [B, T, 1]
    idx_mask = (idx >= start) & (idx < end)  # [B, T, N]
    weights = hidden * idx_mask  # [B, T, N]
    product_sum = torch.sum(weights * idx_cents, dim=2)  # [B, T]
    weight_sum = torch.sum(weights, dim=2)  # [B, T]
    cents = product_sum / (weight_sum + (weight_sum == 0))  # avoid dividing by zero, [B, T]
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thred  # [B, T]
    f0 = f0 * ~uv
    return f0.squeeze(0).cpu().numpy()


def to_viterbi_f0(hidden, thred=0.03):
    # Create viterbi transition matrix
    if not hasattr(to_viterbi_f0, 'transition'):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_f0.transition = transition

    # Convert to probability
    prob = hidden.squeeze(0).cpu().numpy()
    prob = prob.T
    prob = prob / prob.sum(axis=0)

    # Perform viterbi decoding
    path = librosa.sequence.viterbi(prob, to_viterbi_f0.transition).astype(np.int64)
    center = torch.from_numpy(path).unsqueeze(0).unsqueeze(-1).to(hidden.device)

    return to_local_average_f0(hidden, center=center, thred=thred)
