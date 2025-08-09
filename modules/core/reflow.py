from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from modules.backbones import build_backbone
from utils.hparams import hparams


class RectifiedFlow(nn.Module):
    def __init__(self, out_dims, num_feats=1, t_start=0., time_scale_factor=1000,
                 backbone_type=None, backbone_args=None,
                 spec_min=None, spec_max=None):
        super().__init__()
        self.velocity_fn: nn.Module = build_backbone(out_dims, num_feats, backbone_type, backbone_args)
        self.out_dims = out_dims
        self.num_feats = num_feats
        self.use_shallow_diffusion = hparams.get('use_shallow_diffusion', False)
        if self.use_shallow_diffusion:
            assert 0. <= t_start <= 1., 'T_start should be in [0, 1].'
        else:
            t_start = 0.
        self.t_start = t_start
        self.time_scale_factor = time_scale_factor

        # spec: [B, T, M] or [B, F, T, M]
        # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
        spec_min = torch.FloatTensor(spec_min)[None, None, :out_dims].transpose(-3, -2)
        spec_max = torch.FloatTensor(spec_max)[None, None, :out_dims].transpose(-3, -2)
        self.register_buffer('spec_min', spec_min, persistent=False)
        self.register_buffer('spec_max', spec_max, persistent=False)

    def p_losses(self, x_end, t, cond):
        x_start = torch.randn_like(x_end)
        x_t = x_start + t[:, None, None, None] * (x_end - x_start)
        v_pred = self.velocity_fn(x_t, t * self.time_scale_factor, cond)

        return v_pred, x_end - x_start

    def forward(self, condition, gt_spec=None, src_spec=None, infer=True):
        cond = condition.transpose(1, 2)
        b, device = condition.shape[0], condition.device

        if not infer:
            # gt_spec: [B, T, M] or [B, F, T, M]
            spec = self.norm_spec(gt_spec).transpose(-2, -1)  # [B, M, T] or [B, F, M, T]
            if self.num_feats == 1:
                spec = spec[:, None, :, :]  # [B, F=1, M, T]
            t = self.t_start + (1.0 - self.t_start) * torch.rand((b,), device=device)
            v_pred, v_gt = self.p_losses(spec, t, cond=cond)
            return v_pred, v_gt, t
        else:
            # src_spec: [B, T, M] or [B, F, T, M]
            if src_spec is not None:
                spec = self.norm_spec(src_spec).transpose(-2, -1)
                if self.num_feats == 1:
                    spec = spec[:, None, :, :]
            else:
                spec = None
            x = self.inference(cond, b=b, x_end=spec, device=device)
            return self.denorm_spec(x)

    @torch.no_grad()
    def _get_velocity(self, x, t, cond, noise, base, expr, is_guidance):
        v_pred = self.velocity_fn(x, t, cond)
        if not is_guidance:
            return v_pred
        
        v_guidance = base - noise
        return expr * v_pred + (1 - expr) * v_guidance

    @torch.no_grad()
    def sample_euler(self, x, t, dt, cond, noise, base, expr, is_guidance):
        x += self._get_velocity(x, self.time_scale_factor * t, cond, noise, base, expr, is_guidance) * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk2(self, x, t, dt, cond, noise, base, expr, is_guidance):
        k_1 = self._get_velocity(x, self.time_scale_factor * t, cond, noise, base, expr, is_guidance)
        k_2 = self._get_velocity(x + 0.5 * k_1 * dt, self.time_scale_factor * (t + 0.5 * dt), cond, noise, base, expr, is_guidance)
        x += k_2 * dt
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk4(self, x, t, dt, cond, noise, base, expr, is_guidance):
        k_1 = self._get_velocity(x, self.time_scale_factor * t, cond, noise, base, expr, is_guidance)
        k_2 = self._get_velocity(x + 0.5 * k_1 * dt, self.time_scale_factor * (t + 0.5 * dt), cond, noise, base, expr, is_guidance)
        k_3 = self._get_velocity(x + 0.5 * k_2 * dt, self.time_scale_factor * (t + 0.5 * dt), cond, noise, base, expr, is_guidance)
        k_4 = self._get_velocity(x + k_3 * dt, self.time_scale_factor * (t + dt), cond)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt / 6
        t += dt
        return x, t

    @torch.no_grad()
    def sample_rk5(self, x, t, dt, cond, noise, base, expr, is_guidance):
        k_1 = self._get_velocity(x, self.time_scale_factor * t, cond, noise, base, expr, is_guidance)
        k_2 = self._get_velocity(x + 0.25 * k_1 * dt, self.time_scale_factor * (t + 0.25 * dt), cond, noise, base, expr, is_guidance)
        k_3 = self._get_velocity(x + 0.125 * (k_2 + k_1) * dt, self.time_scale_factor * (t + 0.25 * dt), cond, noise, base, expr, is_guidance)
        k_4 = self._get_velocity(x + 0.5 * (-k_2 + 2 * k_3) * dt, self.time_scale_factor * (t + 0.5 * dt), cond, noise, base, expr, is_guidance)
        k_5 = self._get_velocity(x + 0.0625 * (3 * k_1 + 9 * k_4) * dt, self.time_scale_factor * (t + 0.75 * dt), cond, noise, base, expr, is_guidance)
        k_6 = self._get_velocity(x + (-3 * k_1 + 2 * k_2 + 12 * k_3 - 12 * k_4 + 8 * k_5) * dt / 7,
                               self.time_scale_factor * (t + dt),
                               cond)
        x += (7 * k_1 + 32 * k_3 + 12 * k_4 + 32 * k_5 + 7 * k_6) * dt / 90
        t += dt
        return x, t

    @torch.no_grad()
    def inference(self, cond, b=1, x_end=None, device=None, input_mel=None, inpaint_mask=None, inpaint_weight=None, base=None, expr=1.0, temperature=1.0):
        # 在这里进行inpainting机制开启的判断和输入的处理
        # input_mel与inference结果对齐（[B, T, M] or [B, F, T, M]），调整到与noise对齐（[B, F, M, T]）
        # inpaint_mask是一个一维布尔值（[B, T]），**与retake对齐，True为mask部分**，调整到与时间维度对齐（[B, 1, 1, T]）
        # inpaint_weight在这里定义为一个帧级数值（[B，F, T] or [B, T]），调整到与时间维度对齐（[B, F, 1, T]），取值范围为0~1
        is_inpaint = inpaint_mask is not None and input_mel is not None and inpaint_weight is not None
        
        if is_inpaint:
            inpaint_mask = inpaint_mask.float().to(device).unsqueeze(-2) # [B, F, 1, T] or [B, 1, T]
            inpaint_weight = inpaint_weight.float().to(device).unsqueeze(-2) # [B, F, 1, T] or [B, 1, T]
            input_mel = self.norm_spec(input_mel).transpose(-2, -1) # [B, F, M, T] or [B, M, T]
            if self.num_feats == 1:
                inpaint_mask = inpaint_mask[:, None, :, :] # [B, 1, 1, T]
                inpaint_weight = inpaint_weight[:, None, :, :] # [B, 1, 1, T]
                input_mel = input_mel[:, None, :, :] # [B, 1, M, T]

        # Training-Free Guidance
        # base:[B, T]
        is_guidance = base is not None and expr < 1.0
        
        if is_guidance:
            base = self.norm_spec(base).transpose(-2, -1).unsqueeze(-2)
            if self.num_feats == 1:
                base = base[:, None, :, :] # [B, 1, 1, T]

        # 在这里noise要乘上temperature，temperature在默认情况下为1.0，降低temperature会降低结果的多样性，反之亦然
        # temperature ≠ 1.0时，与训练不对齐，理论上会降低质量，实践中因为数据质量分布差异调节可能会有改善
        noise = torch.randn(b, self.num_feats, self.out_dims, cond.shape[2], device=device) * temperature
        t_start = hparams.get('T_start_infer', self.t_start)
        if self.use_shallow_diffusion and t_start > 0:
            assert x_end is not None, 'Missing shallow diffusion source.'
            # shallow diffusion的情况下，在这里构造x_end，把input_mel和前级的输出进行拼接
            # 也就是说对于保留部分，渲染的起点也是input_mel
            # 起点不考虑inpaint_weight
            if is_inpaint:
                x_end = x_end * inpaint_mask + input_mel * (1 - inpaint_mask)
            if t_start >= 1.:
                t_start = 1.
                x = x_end
            else:
                x = t_start * x_end + (1 - t_start) * noise
        else:
            # 考虑直接对input_mel进行shallow diffusion的情况, 也就是说对于全扩散模型也要考虑渲染深度问题
            if is_inpaint:
                if t_start >= 1.:
                    t_start = 1.
                    x = input_mel
                else:
                    x = t_start * input_mel + (1 - t_start) * noise
            else:
                t_start = 0.
                x = noise
        
        algorithm = hparams['sampling_algorithm']
        infer_step = hparams['sampling_steps']

        if t_start < 1:
            dt = (1.0 - t_start) / max(1, infer_step)
            algorithm_fn = {
                'euler': self.sample_euler,
                'rk2': self.sample_rk2,
                'rk4': self.sample_rk4,
                'rk5': self.sample_rk5,
            }.get(algorithm)
            if algorithm_fn is None:
                raise ValueError(f'Unsupported algorithm for Rectified Flow: {algorithm}.')
            dts = torch.tensor([dt]).to(x)
            for i in tqdm(range(infer_step), desc='sample time step', total=infer_step,
                          disable=not hparams['infer'], leave=False):
                ti = t_start + i * dts
                x, _ = algorithm_fn(x, ti, dt, cond, noise, base, expr, is_guidance)
                # **关键**，这里每一步要把去噪的结果修正到保留部分+对应噪声的结果
                # 根据inpaint_weight修正到要保留的程度
                if is_inpaint:
                    weight = (1 - inpaint_mask) * inpaint_weight
                    x = x * (1 - weight) + (input_mel * ti + noise * (1 - ti)) * weight
            x = x.float()
        x = x.transpose(2, 3).squeeze(1)  # [B, F, M, T] => [B, T, M] or [B, F, T, M]
        return x

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class RepetitiveRectifiedFlow(RectifiedFlow):
    def __init__(self, vmin: float | int | list, vmax: float | int | list,
                 repeat_bins: int, time_scale_factor=1000,
                 backbone_type=None, backbone_args=None):
        assert (isinstance(vmin, (float, int)) and isinstance(vmin, (float, int))) or len(vmin) == len(vmax)
        num_feats = 1 if isinstance(vmin, (float, int)) else len(vmin)
        spec_min = [vmin] if num_feats == 1 else [[v] for v in vmin]
        spec_max = [vmax] if num_feats == 1 else [[v] for v in vmax]
        self.repeat_bins = repeat_bins
        super().__init__(
            out_dims=repeat_bins, num_feats=num_feats,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type, backbone_args=backbone_args,
            spec_min=spec_min, spec_max=spec_max
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
                 time_scale_factor=1000,
                 backbone_type=None, backbone_args=None):
        self.vmin = vmin  # norm min
        self.vmax = vmax  # norm max
        self.cmin = cmin  # clip min
        self.cmax = cmax  # clip max
        super().__init__(
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type, backbone_args=backbone_args
        )

    def norm_spec(self, x):
        return super().norm_spec(x.clamp(min=self.cmin, max=self.cmax))

    def denorm_spec(self, x):
        return super().denorm_spec(x).clamp(min=self.cmin, max=self.cmax)


class MultiVarianceRectifiedFlow(RepetitiveRectifiedFlow):
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
        super().__init__(
            vmin=vmin, vmax=vmax, repeat_bins=repeat_bins,
            time_scale_factor=time_scale_factor,
            backbone_type=backbone_type, backbone_args=backbone_args
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
