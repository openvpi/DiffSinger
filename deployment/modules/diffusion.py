from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

from modules.core import (
    GaussianDiffusion, PitchDiffusion, MultiVarianceDiffusion
)


def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))


# noinspection PyMethodOverriding
class GaussianDiffusionONNX(GaussianDiffusion):
    @property
    def backbone(self):
        return self.denoise_fn

    # We give up the setter for the property `backbone` because this will cause TorchScript to fail
    # @backbone.setter
    @torch.jit.unused
    def set_backbone(self, value):
        self.denoise_fn = value

    def q_sample(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t) * noise
        )

    def p_sample(self, x, t, cond):
        x_pred = self.denoise_fn(x, t, cond)
        x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t) * x -
                extract(self.sqrt_recipm1_alphas_cumprod, t) * x_pred
        )
        # This is previously inherited from original DiffSinger repository
        # and disabled due to some loudness issues when speedup = 1.
        # x_recon = torch.clamp(x_recon, min=-1., max=1.)

        model_mean = (
                extract(self.posterior_mean_coef1, t) * x_recon +
                extract(self.posterior_mean_coef2, t) * x
        )
        model_log_variance = extract(self.posterior_log_variance_clipped, t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = ((t > 0).float()).reshape(1, 1, 1, 1)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_ddim(self, x, t, interval: int, cond):
        a_t = extract(self.alphas_cumprod, t)
        t_prev = t - interval
        a_prev = extract(self.alphas_cumprod, t_prev * (t_prev > 0))

        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_prev = a_prev.sqrt() * (
                x / a_t.sqrt() + (((1 - a_prev) / a_prev).sqrt() - ((1 - a_t) / a_t).sqrt()) * noise_pred
        )
        return x_prev

    def plms_get_x_pred(self, x, noise_t, t, t_prev):
        a_t = extract(self.alphas_cumprod, t)
        a_prev = extract(self.alphas_cumprod, t_prev)
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

        x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (
                a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x + x_delta

        return x_pred

    def p_sample_plms(self, x_prev, t, interval: int, cond, noise_list: List[Tensor], stage: int):
        noise_pred = self.denoise_fn(x_prev, t, cond)
        t_prev = t - interval
        t_prev = t_prev * (t_prev > 0)
        if stage == 0:
            x_pred = self.plms_get_x_pred(x_prev, noise_pred, t, t_prev)
            noise_pred_prev = self.denoise_fn(x_pred, t_prev, cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2.
        elif stage == 1:
            noise_pred_prime = (3. * noise_pred - noise_list[-1]) / 2.
        elif stage == 2:
            noise_pred_prime = (23. * noise_pred - 16. * noise_list[-1] + 5. * noise_list[-2]) / 12.
        else:
            noise_pred_prime = (55. * noise_pred - 59. * noise_list[-1] + 37.
                                * noise_list[-2] - 9. * noise_list[-3]) / 24.
        x_prev = self.plms_get_x_pred(x_prev, noise_pred_prime, t, t_prev)
        return noise_pred, x_prev

    def norm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.
        b = (self.spec_max + self.spec_min) / 2.
        return (x - b) / k

    def denorm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.
        b = (self.spec_max + self.spec_min) / 2.
        return x * k + b

    def forward(self, condition, x_start=None, depth=None, steps: int = 10):
        condition = condition.transpose(1, 2)  # [1, T, H] => [1, H, T]
        device = condition.device
        n_frames = condition.shape[2]

        noise = torch.randn((1, self.num_feats, self.out_dims, n_frames), device=device)
        if x_start is None:
            speedup = max(1, self.timesteps // steps)
            speedup = self.timestep_factors[torch.sum(self.timestep_factors <= speedup) - 1]
            step_range = torch.arange(0, self.k_step, speedup, dtype=torch.long, device=device).flip(0)[:, None]
            x = noise
        else:
            depth_int64 = min(torch.round(depth * self.timesteps).long(), self.k_step)
            speedup = max(1, depth_int64 // steps)
            depth_int64 = depth_int64 // speedup * speedup  # make depth_int64 a multiple of speedup
            step_range = torch.arange(0, depth_int64, speedup, dtype=torch.long, device=device).flip(0)[:, None]
            x_start = self.norm_spec(x_start).transpose(-2, -1)
            if self.num_feats == 1:
                x_start = x_start[:, None, :, :]
            if depth_int64 >= self.timesteps:
                x = noise
            elif depth_int64 > 0:
                x = self.q_sample(
                    x_start, torch.full((1,), depth_int64 - 1, device=device, dtype=torch.long), noise
                )
            else:
                x = x_start

        if speedup > 1:
            for t in step_range:
                x = self.p_sample_ddim(x, t, interval=speedup, cond=condition)
            # plms_noise_stage: int = 0
            # noise_list: List[Tensor] = []
            # for t in step_range:
            #     noise_pred, x = self.p_sample_plms(
            #         x, t, interval=speedup, cond=condition,
            #         noise_list=noise_list, stage=plms_noise_stage
            #     )
            #     if plms_noise_stage == 0:
            #         noise_list = [noise_pred]
            #         plms_noise_stage = plms_noise_stage + 1
            #     else:
            #         if plms_noise_stage >= 3:
            #             noise_list.pop(0)
            #         else:
            #             plms_noise_stage = plms_noise_stage + 1
            #         noise_list.append(noise_pred)
        else:
            for t in step_range:
                x = self.p_sample(x, t, cond=condition)

        if self.num_feats == 1:
            x = x.squeeze(1).permute(0, 2, 1)  # [B, 1, M, T] => [B, T, M]
        else:
            x = x.permute(0, 1, 3, 2)  # [B, F, M, T] => [B, F, T, M]
        x = self.denorm_spec(x)
        return x


class PitchDiffusionONNX(GaussianDiffusionONNX, PitchDiffusion):
    def __init__(self, config: dict,
                 vmin: float, vmax: float,
                 cmin: float, cmax: float, repeat_bins,
                 timesteps=1000, k_step=1000,
                 backbone_type=None, backbone_args=None,
                 betas=None):
        self.vmin = vmin
        self.vmax = vmax
        self.cmin = cmin
        self.cmax = cmax
        super(PitchDiffusion, self).__init__(
            config=config,
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            timesteps=timesteps, k_step=k_step,
            backbone_type=backbone_type, backbone_args=backbone_args,
            betas=betas
        )

    def clamp_spec(self, x):
        return x.clamp(min=self.cmin, max=self.cmax)

    def denorm_spec(self, x):
        d = (self.spec_max - self.spec_min) / 2.
        m = (self.spec_max + self.spec_min) / 2.
        x = x * d + m
        x = x.mean(dim=-1)
        return x


class MultiVarianceDiffusionONNX(GaussianDiffusionONNX, MultiVarianceDiffusion):
    def __init__(
            self, config: dict,
            ranges: List[Tuple[float, float]],
            clamps: List[Tuple[float | None, float | None] | None],
            repeat_bins, timesteps=1000, k_step=1000,
            backbone_type=None, backbone_args=None,
            betas=None
    ):
        assert len(ranges) == len(clamps)
        self.clamps = clamps
        vmin = [r[0] for r in ranges]
        vmax = [r[1] for r in ranges]
        if len(vmin) == 1:
            vmin = vmin[0]
        if len(vmax) == 1:
            vmax = vmax[0]
        super(MultiVarianceDiffusion, self).__init__(
            config=config,
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            timesteps=timesteps, k_step=k_step,
            backbone_type=backbone_type, backbone_args=backbone_args,
            betas=betas
        )

    def denorm_spec(self, x):
        d = (self.spec_max - self.spec_min) / 2.
        m = (self.spec_max + self.spec_min) / 2.
        x = x * d + m
        x = x.mean(dim=-1)
        return x
