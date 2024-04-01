from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.hparams import hparams
from modules.diffusion.wavenet import WaveNet
from inspect import isfunction

DIFF_DENOISERS = {
    'wavenet': WaveNet
}


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


class RectifiedFlow(nn.Module):
    def __init__(self, out_dims, num_feats=1, timesteps=1000, k_step=1000,
                 denoiser_type=None, denoiser_args=None, betas=None,
                 spec_min=None, spec_max=None):
        super().__init__()

        self.denoise_fn: nn.Module = DIFF_DENOISERS[denoiser_type](out_dims, num_feats, **denoiser_args)
        self.out_dims = out_dims
        self.num_feats = num_feats
        self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', False)
        if self.use_shallow_diffusion:
            assert k_step <= timesteps, 'K_step should not be larger than timesteps.'
        self.timesteps = timesteps
        self.k_step = k_step if self.use_shallow_diffusion else timesteps
        self.timestep_type = hparams.get('timestep_type', 'continuous')

        # spec: [B, T, M] or [B, F, T, M]
        # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer('spec_min', spec_min)
        self.register_buffer('spec_max', spec_max)

    def get_timestep(self, B, device):
        if self.timestep_type == 'continuous':
            t_start = (self.timesteps - self.k_step) / self.timesteps
            t = t_start + (1.0 - t_start) * torch.rand((B,), device=device)
            return t, t * self.timesteps
        elif self.timestep_type == 'discrete':
            t = torch.randint(0, self.k_step , (B,), device=device).long() + (self.timesteps - self.k_step)  # todo

            return t.float() / self.timesteps, t
        else:
            raise NotImplementedError(self.timestep_type)

    def p_losses(self, x_start, t, cond):

        x_noisy = torch.randn_like(x_start)
        x_t = x_noisy + t[0][:, None, None, None] * (x_start - x_noisy)
        x_recon = self.denoise_fn(x_t, t[1], cond)

        return x_recon, x_start - x_noisy

    def forward(self, condition, gt_spec=None, src_spec=None, infer=True):
        """
            conditioning diffusion, use fastspeech2 encoder output as the condition
        """
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device

        if not infer:
            # gt_spec: [B, T, M] or [B, F, T, M]
            spec = self.norm_spec(gt_spec).transpose(-2, -1)  # [B, M, T] or [B, F, M, T]
            if self.num_feats == 1:
                spec = spec[:, None, :, :]  # [B, F=1, M, T]
            # t = torch.randint(0, self.k_step, (b,), device=device).long()
            t = self.get_timestep(b, device)
            x_recon, xv = self.p_losses(spec, t, cond=cond)
            return x_recon, xv, t[0]
        else:
            # src_spec: [B, T, M] or [B, F, T, M]
            if src_spec is not None:
                spec = self.norm_spec(src_spec).transpose(-2, -1)
                if self.num_feats == 1:
                    spec = spec[:, None, :, :]
            else:
                spec = None
            x = self.inference(cond, b=b, x_start=spec, device=device)
            return self.denorm_spec(x)

    @torch.no_grad()
    def sample_euler(self, x, t, dt, cond, model_fn):
        x += model_fn(x, self.timesteps * t, cond) * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk4(self, x, t, dt, cond, model_fn):
        k_1 = model_fn(x, self.timesteps * t, cond)
        k_2 = model_fn(x + 0.5 * k_1 * dt, self.timesteps * (t + 0.5 * dt), cond)
        k_3 = model_fn(x + 0.5 * k_2 * dt, self.timesteps * (t + 0.5 * dt), cond)
        k_4 = model_fn(x + k_3 * dt, self.timesteps * (t + dt), cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk2(self, x, t, dt, cond, model_fn):
        k_1 = model_fn(x, self.timesteps * t, cond)
        k_2 = model_fn(x + 0.5 * k_1 * dt, self.timesteps * (t + 0.5 * dt), cond)
        x += k_2 * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk5(self, x, t, dt, cond, model_fn):
        k_1 = model_fn(x, self.timesteps * t, cond)
        k_2 = model_fn(x + 0.25 * k_1 * dt, self.timesteps * (t + 0.25 * dt), cond)
        k_3 = model_fn(x + 0.125 * (k_2 + k_1) * dt, self.timesteps * (t + 0.25 * dt), cond)
        k_4 = model_fn(x + 0.5 * (-k_2 + 2 * k_3) * dt, self.timesteps * (t + 0.5 * dt), cond)
        k_5 = model_fn(x + 0.0625 * (3 * k_1 + 9 * k_4) * dt, self.timesteps * (t + 0.75 * dt), cond)
        k_6 = model_fn(x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt / 7, self.timesteps * (t + dt),
                       cond)
        x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt / 90
        t += dt
        return x, t

    @torch.no_grad()
    def sample_euler_fp64(self, x, t, dt, cond, model_fn):
        x = x.double()
        x += model_fn(x.float(), self.timesteps * t, cond).double() * dt.double()
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk4_fp64(self, x, t, dt, cond, model_fn):
        x = x.double()
        k_1 = model_fn(x.float(), self.timesteps * t, cond).double()
        k_2 = model_fn((x + 0.5 * k_1 * dt.double()).float(), self.timesteps * (t + 0.5 * dt), cond).double()
        k_3 = model_fn((x + 0.5 * k_2 * dt.double()).float(), self.timesteps * (t + 0.5 * dt), cond).double()
        k_4 = model_fn((x + k_3 * dt.double()).float(), self.timesteps * (t + dt), cond).double()
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt.double() / 6
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk2_fp64(self, x, t, dt, cond, model_fn):
        x = x.double()
        k_1 = model_fn(x.float(), self.timesteps * t, cond).double()
        k_2 = model_fn((x + 0.5 * k_1 * dt.double()).float(), self.timesteps * (t + 0.5 * dt), cond).double()
        x += k_2 * dt.double()
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk5_fp64(self, x, t, dt, cond, model_fn):
        x = x.double()
        k_1 = model_fn(x.float(), self.timesteps * t, cond).double()
        k_2 = model_fn((x + 0.25 * k_1 * dt.double()).float(), self.timesteps * (t + 0.25 * dt), cond).double()
        k_3 = model_fn((x + 0.125 * (k_2 + k_1) * dt.double()).float(), self.timesteps * (t + 0.25 * dt), cond).double()
        k_4 = model_fn((x + 0.5 * (-k_2 + 2 * k_3) * dt.double()).float(), self.timesteps * (t + 0.5 * dt),
                       cond).double()
        k_5 = model_fn((x + 0.0625 * (3 * k_1 + 9 * k_4) * dt.double()).float(), self.timesteps * (t + 0.75 * dt),
                       cond).double()
        k_6 = model_fn((x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt.double() / 7).float(),
                       self.timesteps * (t + dt),
                       cond).double()
        x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt.double() / 90
        t += dt
        return x, t

    @torch.no_grad()
    def inference(self, cond, b=1, x_start=None, device=None):
        depth = hparams.get('K_step_infer', self.k_step)
        noise = torch.randn(b, self.num_feats, self.out_dims, cond.shape[2], device=device)
        if self.use_shallow_diffusion:
            t_max = min(depth, self.k_step)
        else:
            t_max = self.k_step

        if t_max >= self.timesteps:
            x = noise
            t_start = 0.
        elif t_max > 0:
            assert x_start is not None, 'Missing shallow diffusion source.'
            # x = self.q_sample(
            #     x_start, torch.full((b,), t_max - 1, device=device, dtype=torch.long), noise
            # )
            t_start = 1. - t_max / self.timesteps  # todo
            # t_start = 1. - (t_max - 1.) / self.timesteps
            x = t_start * x_start + (1 - t_start) * noise
        else:
            assert x_start is not None, 'Missing shallow diffusion source.'
            t_start = 1.
            x = x_start
        algorithm = hparams['diff_accelerator']
        if hparams['diff_speedup'] > 1 and t_max > 0:

            infer_step = int(t_max / hparams['diff_speedup'])

        else:
            infer_step = int(t_max)
            # for i in tqdm(reversed(range(0, t_max)), desc='sample time step', total=t_max,
            #               disable=not hparams['infer'], leave=False):
            #     x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)

        if infer_step != 0:
            dt = (1.0 - t_start) / infer_step
            # t = torch.full((b,), t_start, device=device)
            algorithm_fn = {'euler': self.sample_euler, 'rk4': self.sample_rk4, 'rk2': self.sample_rk2,
                            'rk5': self.sample_rk5, 'rk5_fp64': self.sample_rk5_fp64,
                            'euler_fp64': self.sample_euler_fp64,
                            'rk4_fp64': self.sample_rk4_fp64, 'rk2_fp64': self.sample_rk2_fp64, }.get(algorithm)
            if algorithm_fn is None:
                raise NotImplementedError(algorithm)
            dts=torch.tensor(dt).to(x)
            for i in tqdm(range(infer_step), desc='sample time step', total=infer_step,
                          disable=not hparams['infer'], leave=False):
                x, _ = algorithm_fn(x,t_start+ i*dts, dt, cond, model_fn=self.denoise_fn)
            x = x.float()
        x = x.transpose(2, 3).squeeze(1)  # [B, F, M, T] => [B, T, M] or [B, F, T, M]
        return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class RepetitiveRectifiedFlow(RectifiedFlow):
    def __init__(self, vmin: float | int | list, vmax: float | int | list, repeat_bins: int,
                 timesteps=1000, k_step=1000,
                 denoiser_type=None, denoiser_args=None,
                 betas=None):
        assert (isinstance(vmin, (float, int)) and isinstance(vmin, (float, int))) or len(vmin) == len(vmax)
        num_feats = 1 if isinstance(vmin, (float, int)) else len(vmin)
        spec_min = [vmin] if num_feats == 1 else [[v] for v in vmin]
        spec_max = [vmax] if num_feats == 1 else [[v] for v in vmax]
        self.repeat_bins = repeat_bins
        super().__init__(
            out_dims=repeat_bins, num_feats=num_feats,
            timesteps=timesteps, k_step=k_step,
            denoiser_type=denoiser_type, denoiser_args=denoiser_args,
            betas=betas, spec_min=spec_min, spec_max=spec_max
        )

    def norm_spec(self, x):
        """

        :param x: [B, T] or [B, F, T]
        :return [B, T, R] or [B, F, T, R]
        """
        if self.num_feats == 1:
            repeats = [1, 1, self.repeat_bins]
        else:
            repeats = [1, 1, 1, self.repeat_bins]
        return super().norm_spec(x.unsqueeze(-1).repeat(repeats))

    def denorm_spec(self, x):
        """

        :param x: [B, T, R] or [B, F, T, R]
        :return [B, T] or [B, F, T]
        """
        return super().denorm_spec(x).mean(dim=-1)


class PitchRectifiedFlow(RepetitiveRectifiedFlow):
    def __init__(self, vmin: float, vmax: float,
                 cmin: float, cmax: float, repeat_bins,
                 timesteps=1000, k_step=1000,
                 denoiser_type=None, denoiser_args=None,
                 betas=None):
        self.vmin = vmin  # norm min
        self.vmax = vmax  # norm max
        self.cmin = cmin  # clip min
        self.cmax = cmax  # clip max
        super().__init__(
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            timesteps=timesteps, k_step=k_step,
            denoiser_type=denoiser_type, denoiser_args=denoiser_args,
            betas=betas
        )

    def norm_spec(self, x):
        return super().norm_spec(x.clamp(min=self.cmin, max=self.cmax))

    def denorm_spec(self, x):
        return super().denorm_spec(x).clamp(min=self.cmin, max=self.cmax)


class MultiVarianceRectifiedFlow(RepetitiveRectifiedFlow):
    def __init__(
            self, ranges: List[Tuple[float, float]],
            clamps: List[Tuple[float | None, float | None] | None],
            repeat_bins, timesteps=1000, k_step=1000,
            denoiser_type=None, denoiser_args=None,
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
        super().__init__(
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            timesteps=timesteps, k_step=k_step,
            denoiser_type=denoiser_type, denoiser_args=denoiser_args,
            betas=betas
        )

    def clamp_spec(self, xs: list | tuple):
        clamped = []
        for x, c in zip(xs, self.clamps):
            if c is None:
                clamped.append(x)
                continue
            clamped.append(x.clamp(min=c[0], max=c[1]))
        return clamped

    def norm_spec(self, xs: list | tuple):
        """

        :param xs: sequence of [B, T]
        :return: [B, F, T] => super().norm_spec(xs) => [B, F, T, R]
        """
        assert len(xs) == self.num_feats
        clamped = self.clamp_spec(xs)
        xs = torch.stack(clamped, dim=1)  # [B, F, T]
        if self.num_feats == 1:
            xs = xs.squeeze(1)  # [B, T]
        return super().norm_spec(xs)

    def denorm_spec(self, xs):
        """

        :param xs: [B, T, R] or [B, F, T, R] => super().denorm_spec(xs) => [B, T] or [B, F, T]
        :return: sequence of [B, T]
        """
        xs = super().denorm_spec(xs)
        if self.num_feats == 1:
            xs = [xs]
        else:
            xs = xs.unbind(dim=1)
        assert len(xs) == self.num_feats
        return self.clamp_spec(xs)
