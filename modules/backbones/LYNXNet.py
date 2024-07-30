# refer to： 
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/model_conformer_naive.py
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/naive_v2_diff.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hparams import hparams


class SwiGLU(nn.Module):
    ## Swish-Applies the gated linear unit function.
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        # out, gate = x.chunk(2, dim=self.dim)
        # Using torch.split instead of chunk for ONNX export compatibility.
        out, gate = torch.split(x, x.size(self.dim) // 2, dim=self.dim)
        return out * F.silu(gate)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class LYNXConvModule(nn.Module):
    @staticmethod
    def calc_same_padding(kernel_size):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)

    def __init__(self, dim, expansion_factor, kernel_size=31, in_norm=False, activation='PReLU', dropout=0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        _normalize = nn.LayerNorm(dim) if in_norm or dim > 512 else nn.Identity()
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
            _normalize,
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            SwiGLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding[0], groups=inner_dim),
            _activation,
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            _dropout
        )

    def forward(self, x):
        return self.net(x)


class LYNXNetResidualLayer(nn.Module):
    def __init__(self, dim_cond, dim, expansion_factor, kernel_size=31, in_norm=False, activation='PReLU', dropout=0.):
        super().__init__()
        self.diffusion_projection = nn.Conv1d(dim, dim, 1)
        self.conditioner_projection = nn.Conv1d(dim_cond, dim, 1)
        self.convmodule = LYNXConvModule(dim=dim, expansion_factor=expansion_factor, kernel_size=kernel_size, in_norm=in_norm, activation=activation, dropout=dropout)

    def forward(self, x, conditioner, diffusion_step):
        res_x = x.transpose(1, 2)
        x = x + self.diffusion_projection(diffusion_step) + self.conditioner_projection(conditioner)
        x = x.transpose(1, 2)
        x = self.convmodule(x)  # (#batch, dim, length)
        x = x + res_x
        x = x.transpose(1, 2)

        return x  # (#batch, length, dim)


class LYNXNet(nn.Module):
    def __init__(self, in_dims, n_feats, *, n_layers=6, n_chans=512, n_dilates=2, in_norm=False, activation='PReLU', dropout=0.):
        """
        LYNXNet(Linear Gated Depthwise Separable Convolution Network)
        TIPS:You can control the style of the generated results by modifying the 'activation', 
            - 'PReLU'(default) : Similar to WaveNet
            - 'SiLU' : Voice will be more pronounced, not recommended for use under DDPM
            - 'ReLU' : Contrary to 'SiLU', Voice will be weakened
        """
        super().__init__()
        self.input_projection = nn.Conv1d(in_dims * n_feats, n_chans, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(n_chans),
            nn.Linear(n_chans, n_chans * 4),
            nn.GELU(),
            nn.Linear(n_chans * 4, n_chans),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNetResidualLayer(
                    dim_cond=hparams['hidden_size'], 
                    dim=n_chans, 
                    expansion_factor=n_dilates, 
                    kernel_size=31, 
                    in_norm=in_norm, 
                    activation=activation, 
                    dropout=dropout
                )
                for i in range(n_layers)
            ]
        )
        self.output_projection = nn.Conv1d(n_chans, in_dims * n_feats, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        
    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """
        
        # To keep compatibility with DiffSVC, [B, 1, M, T]
        x = spec
        use_4_dim = False
        if x.dim() == 4:
            x = x[:, 0]
            use_4_dim = True

        assert x.dim() == 3, f"mel must be 3 dim tensor, but got {x.dim()}"

        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.gelu(x)
        
        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)
        
        for layer in self.residual_layers:
            x = layer(x, cond, diffusion_step)
        
        # MLP and GLU
        x = self.output_projection(x)  # [B, 128, T]
        
        return x[:, None] if use_4_dim else x
