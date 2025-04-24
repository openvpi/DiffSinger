import torch
from torch import nn

__all__ = [
    "FeatureNormalizer",
]


class FeatureNormalizer(nn.Module):
    def __init__(
            self, num_channels: int, num_features: int = 1, num_repeats: int = None,
            squeezed_channel_dim: bool = False,
            norm_mins: list[float] = None, norm_maxs: list[float] = None,
            clip_mins: list[float] = None, clip_maxs: list[float] = None,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_repeats = num_repeats
        self.squeezed_channel_dim = squeezed_channel_dim
        if self.num_channels > 1 and self.squeezed_channel_dim:
            raise ValueError("squeeze_channel_dim cannot be True if num_channels > 1.")
        dims = 2 if self.squeezed_channel_dim else 3
        if norm_mins is not None:
            if norm_maxs is None:
                raise ValueError("norm_maxs must be provided if norm_mins is provided.")
            if len(norm_mins) != num_features:
                raise ValueError(f"len(norm_mins) != num_features: {len(norm_mins)} != {num_features}")
            self.register_buffer("norm_min", insert_dims(torch.FloatTensor(norm_mins), dims), persistent=False)
        else:
            self.norm_min = None
        if norm_maxs is not None:
            if norm_mins is None:
                raise ValueError("norm_mins must be provided if norm_maxs is provided.")
            if len(norm_maxs) != num_features:
                raise ValueError(f"len(norm_maxs) != num_features: {len(norm_maxs)} != {num_features}")
            self.register_buffer("norm_max", insert_dims(torch.FloatTensor(norm_maxs), dims), persistent=False)
        else:
            self.norm_max = None
        if clip_mins is not None:
            if len(clip_mins) != num_features:
                raise ValueError(f"len(clip_mins) != num_features: {len(clip_mins)} != {num_features}")
            clip_mins = [float("-inf") if x is None else x for x in clip_mins]
            self.register_buffer("clip_min", insert_dims(torch.FloatTensor(clip_mins), dims), persistent=False)
        else:
            self.clip_min = None
        if clip_maxs is not None:
            if len(clip_maxs) != num_features:
                raise ValueError(f"len(clip_maxs) != num_features: {len(clip_maxs)} != {num_features}")
            clip_maxs = [float("+inf") if x is None else x for x in clip_maxs]
            self.register_buffer("clip_max", insert_dims(torch.FloatTensor(clip_maxs), dims), persistent=False)
        else:
            self.clip_max = None

    def norm(self, *features):
        # in: N x [B, T, C] or [B, T]
        # out: [B, T, C x N x R]
        x = torch.stack(features, dim=-1)  # [B, T, C, N] or [B, T, N]
        if self.clip_min is not None or self.clip_max is not None:
            x = x.clamp(min=self.clip_min, max=self.clip_max)
        if self.norm_min is not None:
            x = (x - self.norm_min) / (self.norm_max - self.norm_min) * 2 - 1
        if self.num_repeats is not None:
            x = x.repeat_interleave(repeats=self.num_repeats, dim=-1)  # [B, T, C, N x R] or [B, T, N x R]
        x = x.flatten(start_dim=2)  # [B, T, C x N x R]
        return x

    def denorm(self, x):
        # in: [B, T, C x N x R]
        # out: N x [B, T, C] or [B, T]
        T = x.shape[1]
        if self.num_repeats is None:
            x = x.reshape(-1, T, self.num_channels, self.num_features)  # [B, T, C, N]
        else:
            x = x.reshape(-1, T, self.num_channels, self.num_features, self.num_repeats)  # [B, T, C, N, R]
            x = x.mean(dim=-1)  # [B, T, C, N]
        if self.squeezed_channel_dim:
            x = x.squeeze(2)  # [B, T, N]
        if self.norm_min is not None:
            x = (x + 1) / 2 * (self.norm_max - self.norm_min) + self.norm_min
        if self.clip_min is not None or self.clip_max is not None:
            x = x.clamp(min=self.clip_min, max=self.clip_max)
        if self.num_features == 1:
            features = x.squeeze(-1)  # [B, T, C] or [B, T]
        else:
            features = x.unbind(dim=-1)  # N x [B, T, C] or [B, T]
        return features


def insert_dims(x, n_dims: int):
    # Insert n_dim dimensions to the tensor
    for _ in range(n_dims):
        x = x.unsqueeze(0)
    return x
