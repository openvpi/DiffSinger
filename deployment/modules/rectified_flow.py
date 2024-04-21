from __future__ import annotations

from typing import List, Tuple

import torch

from modules.core import (
    RectifiedFlow, PitchRectifiedFlow, MultiVarianceRectifiedFlow
)


class RectifiedFlowONNX(RectifiedFlow):
    @property
    def backbone(self):
        return self.velocity_fn

    # We give up the setter for the property `backbone` because this will cause TorchScript to fail
    # @backbone.setter
    @torch.jit.unused
    def set_backbone(self, value):
        self.velocity_fn = value

    def sample_euler(self, x, t, dt: float, cond):
        x += self.velocity_fn(x, t * self.time_scale_factor, cond) * dt
        return x

    def norm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.
        b = (self.spec_max + self.spec_min) / 2.
        return (x - b) / k

    def denorm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.
        b = (self.spec_max + self.spec_min) / 2.
        return x * k + b

    def forward(self, condition, x_end=None, depth=None, steps: int = 10):
        condition = condition.transpose(1, 2)  # [1, T, H] => [1, H, T]
        device = condition.device
        n_frames = condition.shape[2]
        noise = torch.randn((1, self.num_feats, self.out_dims, n_frames), device=device)
        if x_end is None:
            t_start = 0.
            x = noise
        else:
            t_start = torch.max(1 - depth, torch.tensor(self.t_start, dtype=torch.float32, device=device))
            x_end = self.norm_spec(x_end).transpose(-2, -1)
            if self.num_feats == 1:
                x_end = x_end[:, None, :, :]
            if t_start <= 0.:
                x = noise
            elif t_start >= 1.:
                x = x_end
            else:
                x = t_start * x_end + (1 - t_start) * noise

        t_width = 1. - t_start
        if t_width >= 0.:
            dt = t_width / max(1, steps)
            for t in torch.arange(steps, dtype=torch.long, device=device)[:, None].float() * dt + t_start:
                x = self.sample_euler(x, t, dt, condition)

        if self.num_feats == 1:
            x = x.squeeze(1).permute(0, 2, 1)  # [B, 1, M, T] => [B, T, M]
        else:
            x = x.permute(0, 1, 3, 2)  # [B, F, M, T] => [B, F, T, M]
        x = self.denorm_spec(x)
        return x


class PitchRectifiedFlowONNX(RectifiedFlowONNX, PitchRectifiedFlow):
    def __init__(self, vmin: float, vmax: float,
                 cmin: float, cmax: float, repeat_bins,
                 time_scale_factor=1000,
                 backbone_type=None, backbone_args=None):
        self.vmin = vmin
        self.vmax = vmax
        self.cmin = cmin
        self.cmax = cmax
        super(PitchRectifiedFlow, self).__init__(
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type, backbone_args=backbone_args
        )

    def clamp_spec(self, x):
        return x.clamp(min=self.cmin, max=self.cmax)

    def denorm_spec(self, x):
        d = (self.spec_max - self.spec_min) / 2.
        m = (self.spec_max + self.spec_min) / 2.
        x = x * d + m
        x = x.mean(dim=-1)
        return x


class MultiVarianceRectifiedFlowONNX(RectifiedFlowONNX, MultiVarianceRectifiedFlow):
    def __init__(
            self, ranges: List[Tuple[float, float]],
            clamps: List[Tuple[float | None, float | None] | None],
            repeat_bins, time_scale_factor=1000,
            backbone_type=None, backbone_args=None
    ):
        assert len(ranges) == len(clamps)
        self.clamps = clamps
        vmin = [r[0] for r in ranges]
        vmax = [r[1] for r in ranges]
        if len(vmin) == 1:
            vmin = vmin[0]
        if len(vmax) == 1:
            vmax = vmax[0]
        super(MultiVarianceRectifiedFlow, self).__init__(
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type, backbone_args=backbone_args
        )

    def denorm_spec(self, x):
        d = (self.spec_max - self.spec_min) / 2.
        m = (self.spec_max + self.spec_min) / 2.
        x = x * d + m
        x = x.mean(dim=-1)
        return x
