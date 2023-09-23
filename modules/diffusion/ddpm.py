from __future__ import annotations

from collections import deque
from functools import partial
from inspect import isfunction
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from modules.diffusion.wavenet import WaveNet
from utils.hparams import hparams

DIFF_DENOISERS = {
    'wavenet': WaveNet
}


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def linear_beta_schedule(timesteps, max_beta=hparams.get('max_beta', 0.01)):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


class GaussianDiffusion(nn.Module):
    def __init__(self, out_dims, num_feats=1, timesteps=1000, k_step=1000,
                 denoiser_type=None, denoiser_args=None, betas=None,
                 spec_min=None, spec_max=None):
        super().__init__()
        self.denoise_fn: nn.Module = DIFF_DENOISERS[denoiser_type](out_dims, num_feats, **denoiser_args)
        self.out_dims = out_dims
        self.num_feats = num_feats

        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = beta_schedule[hparams['schedule_type']](timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', False)
        if self.use_shallow_diffusion:
            assert k_step <= timesteps, 'K_step should not be larger than timesteps.'
        else:
            assert k_step == timesteps, 'K_step must equal timesteps if use_shallow_diffusion is False.'
        self.timesteps = timesteps
        self.k_step = k_step
        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # spec: [B, T, M] or [B, F, T, M]
        # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer('spec_min', spec_min)
        self.register_buffer('spec_max', spec_max)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_ddim(self, x, t, interval, cond):
        a_t = extract(self.alphas_cumprod, t, x.shape)
        a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)

        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_prev = a_prev.sqrt() * (
                x / a_t.sqrt() + (((1 - a_prev) / a_prev).sqrt() - ((1 - a_t) / a_t).sqrt()) * noise_pred
        )
        return x_prev

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from
        [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (
                    a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)

        return x_recon, noise

    def inference(self, cond, b=1, x_start=None, device=None):
        depth = hparams.get('K_step_infer', self.k_step)
        noise = torch.randn(b, self.num_feats, self.out_dims, cond.shape[2], device=device)
        if self.use_shallow_diffusion:
            t_max = min(depth, self.k_step)
        else:
            t_max = self.k_step

        if t_max >= self.timesteps:
            x = noise
        elif t_max > 0:
            assert x_start is not None, 'Missing shallow diffusion source.'
            x = self.q_sample(
                x_start, torch.full((b,), t_max - 1, device=device, dtype=torch.long), noise
            )
        else:
            assert x_start is not None, 'Missing shallow diffusion source.'
            x = x_start

        if hparams.get('pndm_speedup') and hparams['pndm_speedup'] > 1 and t_max > 0:
            algorithm = hparams.get('diff_accelerator', 'ddim')
            if algorithm == 'dpm-solver':
                from inference.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
                # 1. Define the noise schedule.
                noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t_max])

                # 2. Convert your discrete-time `model` to the continuous-time
                # noise prediction model. Here is an example for a diffusion model
                # `model` with the noise prediction type ("noise") .
                def my_wrapper(fn):
                    def wrapped(x, t, **kwargs):
                        ret = fn(x, t, **kwargs)
                        self.bar.update(1)
                        return ret

                    return wrapped

                model_fn = model_wrapper(
                    my_wrapper(self.denoise_fn),
                    noise_schedule,
                    model_type="noise",  # or "x_start" or "v" or "score"
                    model_kwargs={"cond": cond}
                )

                # 3. Define dpm-solver and sample by singlestep DPM-Solver.
                # (We recommend singlestep DPM-Solver for unconditional sampling)
                # You can adjust the `steps` to balance the computation
                # costs and the sample quality.
                dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

                steps = t_max // hparams["pndm_speedup"]
                self.bar = tqdm(desc="sample time step", total=steps, disable=not hparams['infer'], leave=False)
                x = dpm_solver.sample(
                    x,
                    steps=steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
                self.bar.close()
            elif algorithm == 'unipc':
                from inference.uni_pc import NoiseScheduleVP, model_wrapper, UniPC
                # 1. Define the noise schedule.
                noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas[:t_max])

                # 2. Convert your discrete-time `model` to the continuous-time
                # noise prediction model. Here is an example for a diffusion model
                # `model` with the noise prediction type ("noise") .
                def my_wrapper(fn):
                    def wrapped(x, t, **kwargs):
                        ret = fn(x, t, **kwargs)
                        self.bar.update(1)
                        return ret

                    return wrapped

                model_fn = model_wrapper(
                    my_wrapper(self.denoise_fn),
                    noise_schedule,
                    model_type="noise",  # or "x_start" or "v" or "score"
                    model_kwargs={"cond": cond}
                )

                # 3. Define uni_pc and sample by multistep UniPC.
                # You can adjust the `steps` to balance the computation
                # costs and the sample quality.
                uni_pc = UniPC(model_fn, noise_schedule, variant='bh2')

                steps = t_max // hparams["pndm_speedup"]
                self.bar = tqdm(desc="sample time step", total=steps, disable=not hparams['infer'], leave=False)
                x = uni_pc.sample(
                    x,
                    steps=steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
                self.bar.close()
            elif algorithm == 'pndm':
                self.noise_list = deque(maxlen=4)
                iteration_interval = hparams['pndm_speedup']
                for i in tqdm(
                        reversed(range(0, t_max, iteration_interval)), desc='sample time step',
                        total=t_max // iteration_interval, disable=not hparams['infer'], leave=False
                ):
                    x = self.p_sample_plms(
                        x, torch.full((b,), i, device=device, dtype=torch.long),
                        iteration_interval, cond=cond
                    )
            elif algorithm == 'ddim':
                iteration_interval = hparams['pndm_speedup']
                for i in tqdm(
                        reversed(range(0, t_max, iteration_interval)), desc='sample time step',
                        total=t_max // iteration_interval, disable=not hparams['infer'], leave=False
                ):
                    x = self.p_sample_ddim(
                        x, torch.full((b,), i, device=device, dtype=torch.long),
                        iteration_interval, cond=cond
                    )
            else:
                raise NotImplementedError(algorithm)
        else:
            for i in tqdm(reversed(range(0, t_max)), desc='sample time step', total=t_max,
                          disable=not hparams['infer'], leave=False):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
        x = x.transpose(2, 3).squeeze(1)  # [B, F, M, T] => [B, T, M] or [B, F, T, M]
        return x

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
            t = torch.randint(0, self.k_step, (b,), device=device).long()
            return self.p_losses(spec, t, cond=cond)
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

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class RepetitiveDiffusion(GaussianDiffusion):
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


class PitchDiffusion(RepetitiveDiffusion):
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


class MultiVarianceDiffusion(RepetitiveDiffusion):
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
