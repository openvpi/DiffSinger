# refer to： 
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/model_conformer_naive.py
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/naive_v2_diff.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb
from utils.hparams import hparams


class SwiGLU(nn.Module):
    # Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return out * F.silu(gate)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

def zeroinit(mdl):
    nn.init.zeros_(mdl.weight)
    nn.init.zeros_(mdl.bias)
    return mdl

class LYNXConvModule(nn.Module):
    @staticmethod
    def calc_same_padding(kernel_size):
        pad = kernel_size // 2
        return pad, pad - (kernel_size + 1) % 2

    def __init__(self, dim, expansion_factor, kernel_size=31, activation='PReLU', dropout=0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        activation_classes = {
            'SiLU': nn.SiLU,
            'ReLU': nn.ReLU,
            'PReLU': lambda: nn.PReLU(inner_dim)
        }
        activation = activation if activation is not None else 'PReLU'
        if activation not in activation_classes:
            raise ValueError(f'{activation} is not a valid activation')
        _activation = activation_classes[activation]()
        padding = self.calc_same_padding(kernel_size)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            SwiGLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding[0], groups=inner_dim),
            _activation,
            zeroinit(nn.Conv1d(inner_dim, dim, 1)),
            Transpose((1, 2)),
            _dropout
        )

    def forward(self, x):
        return self.net(x)

# Residual connection are removed. Don't add residual connection or the network will fail.
class LYNXNetResidualLayer(nn.Module):
    def __init__(self, dim_cond, dim, expansion_factor, kernel_size=31, activation='PReLU', dropout=0.):
        super().__init__()
        self.diffusion_projection = nn.Conv1d(dim * 4, dim, 1)
        self.conditioner_projection = nn.Conv1d(dim_cond, dim, 1)
        self.convmodule = LYNXConvModule(dim=dim, expansion_factor=expansion_factor, kernel_size=kernel_size,
                                         activation=activation, dropout=dropout)

    def forward(self, x, conditioner, diffusion_step):
        # res_x = x.transpose(1, 2)
        x = x + self.diffusion_projection(diffusion_step) + self.conditioner_projection(conditioner)
        x = x.transpose(1, 2)
        x = self.convmodule(x)  # (#batch, dim, length)
        # x = x + res_x
        x = x.transpose(1, 2)

        return x  # (#batch, length, dim)

class RevNetKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cond, step, layers):
        with torch.no_grad():
            x1, x2 = torch.split(x, [x.size(1) // 2, x.size(1) // 2], dim=1)
            for layer in layers:
                x1 = x1 + layer(x2, cond, step)
                x1, x2 = x2, x1
        ctx.save_for_backward(x1, x2, cond, step)
        ctx.layers = layers
        return torch.cat([x1, x2], dim=1)
    @staticmethod
    def backward(ctx, grad_x):
        grad_x1, grad_x2 = torch.split(grad_x, [grad_x.size(1) // 2, grad_x.size(1) // 2], dim=1)
        x1, x2, cond, step = ctx.saved_tensors
        grad_cond = torch.full_like(cond, 0.0)
        grad_step = torch.full_like(step, 0.0)
        for layer in ctx.layers[::-1]:
            x1, x2 = x2, x1
            grad_x1, grad_x2 = grad_x2, grad_x1
            with torch.enable_grad():
                x2_4g = x2.clone().detach().requires_grad_(True)
                cond_4g = cond.clone().detach().requires_grad_(True)
                step_4g = step.clone().detach().requires_grad_(True)
                dx1 = layer(x2_4g, cond_4g, step_4g)
                dx1.backward(grad_x1)
            grad_x2 += x2_4g.grad
            grad_cond += cond_4g.grad
            grad_step += step_4g.grad
            x1 -= dx1
        return torch.cat([grad_x1, grad_x2], dim=1), grad_cond, grad_step, None

def RevNetKernel_eval(x, cond, step, layers):
    x1, x2 = torch.split(x, [x.size(1) // 2, x.size(1) // 2], dim=1)
    for layer in layers:
        x1 = x1 + layer(x2, cond, step)
        x1, x2 = x2, x1
    return torch.cat([x1, x2], dim=1)

class AsivoNet(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=21, num_channels=512, expansion_factor=2, kernel_size=31,
                 activation='PReLU', dropout=0.):
        """
        AsivoNet (RevNet Improved LYNXNet)
        TIPS:You can control the style of the generated results by modifying the 'activation', 
            - 'PReLU'(default) : Similar to WaveNet
            - 'SiLU' : Voice will be more pronounced, not recommended for use under DDPM
            - 'ReLU' : Contrary to 'SiLU', Voice will be weakened
        """
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Conv1d(in_dims * n_feats, num_channels * 2, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU()
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNetResidualLayer(
                    dim_cond=hparams['hidden_size'],
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    activation=activation,
                    dropout=dropout
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels * 2)
        self.output_projection = nn.Conv1d(num_channels * 2, in_dims * n_feats, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """

        if self.n_feats == 1:
            x = spec[:, 0]  # [B, M, T]
        else:
            x = spec.flatten(start_dim=1, end_dim=2)  # [B, F x M, T]

        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.gelu(x)
        
        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)

        if not torch.onnx.is_in_onnx_export():
            x = x.to(torch.float32)
            diffusion_step = diffusion_step.to(torch.float32)
            cond = cond.to(torch.float32)

        if self.training:
            x = RevNetKernel.apply(x, cond, diffusion_step, self.residual_layers)
        else:
            with torch.no_grad():
                x = RevNetKernel_eval(x, cond, diffusion_step, self.residual_layers)

        # post-norm
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # MLP and GLU
        x = self.output_projection(x)  # [B, 128, T]

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
