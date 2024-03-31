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

    def reflow_loss(self, x_1, t, cond, loss_type=None):
        x_0 = torch.randn_like(x_1)
        x_t = x_0 + t[:, None, None, None] * (x_1 - x_0)
        v_pred = self.velocity_fn(x_t, 1000 * t, cond)

        if loss_type is None:
            loss_type = self.loss_type
        else:
            loss_type = loss_type

        if loss_type == 'l1':
            loss = (x_1 - x_0 - v_pred).abs().mean()
        elif loss_type == 'l2':
            loss = F.mse_loss(x_1 - x_0, v_pred)
        elif loss_type == 'l2_lognorm':
            weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / (1 - t)) ** 2)
            loss = torch.mean(weights[:, None, None, None] * F.mse_loss(x_1 - x_0, v_pred, reduction='none'))
        else:
            raise NotImplementedError()

        return loss

    def get_timestep(self, B, device):
        if self.timestep_type == 'continuous':
            t_start = 1 - (self.timesteps - self.k_step) / self.timesteps
            t = t_start + (1.0 - t_start) * torch.rand((B,), device=device)
            return t, t * self.timesteps
        elif self.timestep_type == 'discrete':
            t = torch.randint(0, self.k_step, (B,), device=device).long() + (self.timesteps - self.k_step)

            return t.float() / self.timesteps, t

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
            for i in tqdm(reversed(range(0, t_max)), desc='sample time step', total=t_max,
                          disable=not hparams['infer'], leave=False):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)

        if infer_step != 0:
            dt = (1.0 - t_start) / infer_step
            t = torch.full((b,), t_start, device=device)
            algorithm_fn = {'euler': self.sample_euler, 'rk4': self.sample_rk4}.get(algorithm)
            if algorithm_fn is None:
                raise NotImplementedError(algorithm)

            for _ in tqdm(range(infer_step), desc='sample time step', total=infer_step,
                          disable=not hparams['infer'], leave=False):
                x, t = algorithm_fn(x, t, dt, cond,model_fn=self.denoise_fn)

        x = x.transpose(2, 3).squeeze(1)  # [B, F, M, T] => [B, T, M] or [B, F, T, M]
        return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
