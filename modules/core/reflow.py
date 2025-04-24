import torch
from torch import nn
from tqdm import tqdm

__all__ = [
    "RectifiedFlow",
]


class RectifiedFlow(nn.Module):
    def __init__(
            self, sample_dim: int, backbone: nn.Module,
            t_start=0., time_scale_factor=1000
    ):
        super().__init__()
        self.sample_dim = sample_dim
        self.velocity_fn = backbone
        self.t_start = t_start
        self.time_scale_factor = time_scale_factor

    def p_losses(self, x_end, t, cond):
        x_start = torch.randn_like(x_end)
        x_t = x_start + t[:, None, None, None] * (x_end - x_start)
        v_pred = self.velocity_fn(x_t, t * self.time_scale_factor, cond)

        return v_pred, x_end - x_start

    def forward(self, cond, x_src=None, x_gt=None, infer=True, sampling_algorithm="euler", sampling_steps=20):
        # cond: [B, T, H]
        # x_src, x_gt: [B, T, C]
        cond = cond.transpose(1, 2)
        b, device = cond.shape[0], cond.device

        if infer:
            if x_src is not None:
                x_src = x_src.transpose(1, 2).unsqueeze(1)  # [B, 1, C, T]
            x = self.inference(
                cond, b=b, x_end=x_src, device=device,
                algorithm=sampling_algorithm,
                num_steps=sampling_steps
            )
            return x
        else:
            x_gt = x_gt.transpose(1, 2).unsqueeze(1)  # [B, 1, C, T]
            t = self.t_start + (1.0 - self.t_start) * torch.rand((b,), device=device)
            v_pred, v_gt = self.p_losses(x_end=x_gt, t=t, cond=cond)
            return v_pred, v_gt, t

    @torch.no_grad()
    def sample_euler(self, x, t, dt, cond):
        x += self.velocity_fn(x, self.time_scale_factor * t, cond) * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk2(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale_factor * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        x += k_2 * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk4(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale_factor * t, cond)
        k_2 = self.velocity_fn(x + 0.5 * k_1 * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        k_3 = self.velocity_fn(x + 0.5 * k_2 * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        k_4 = self.velocity_fn(x + k_3 * dt, self.time_scale_factor * (t + dt), cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk5(self, x, t, dt, cond):
        k_1 = self.velocity_fn(x, self.time_scale_factor * t, cond)
        k_2 = self.velocity_fn(x + 0.25 * k_1 * dt, self.time_scale_factor * (t + 0.25 * dt), cond)
        k_3 = self.velocity_fn(x + 0.125 * (k_2 + k_1) * dt, self.time_scale_factor * (t + 0.25 * dt), cond)
        k_4 = self.velocity_fn(x + 0.5 * (-k_2 + 2 * k_3) * dt, self.time_scale_factor * (t + 0.5 * dt), cond)
        k_5 = self.velocity_fn(x + 0.0625 * (3 * k_1 + 9 * k_4) * dt, self.time_scale_factor * (t + 0.75 * dt), cond)
        k_6 = self.velocity_fn(x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt / 7,
                               self.time_scale_factor * (t + dt),
                               cond)
        x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt / 90
        t += dt
        return x, t

    @torch.no_grad()
    def inference(self, cond, b=1, x_end=None, device=None, algorithm="euler", num_steps=20):
        noise = torch.randn(b, 1, self.sample_dim, cond.shape[2], device=device)
        if x_end is None:
            if self.t_start > 0.:
                raise ValueError("Missing shallow diffusion source.")
            x = noise
        else:
            x = self.t_start * x_end + (1 - self.t_start) * noise

        if self.t_start < 1.:
            dt = (1. - self.t_start) / max(1, num_steps)
            algorithm_fn = {
                'euler': self.sample_euler,
                'rk2': self.sample_rk2,
                'rk4': self.sample_rk4,
                'rk5': self.sample_rk5,
            }[algorithm]
            dts = torch.tensor([dt]).to(x)
            for i in tqdm(
                    range(num_steps), desc="sample time step", total=num_steps,
                    disable=False, leave=False
            ):
                x, _ = algorithm_fn(x, self.t_start + i * dts, dt, cond)
            x = x.float()
        x = x.transpose(2, 3).squeeze(1)  # [B, 1, M, T] => [B, T, M]
        return x
