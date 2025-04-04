# refer to： 
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/model_conformer_naive.py
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/naive_v2_diff.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb, SwiGLU


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


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
        return pad, pad - (kernel_size + 1) % 2

    def __init__(self, dim, expansion_factor, kernel_size=31, activation='PReLU', dropout=0.0):
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
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            _dropout
        )

    def forward(self, x):
        return self.net(x)


class LYNXNetResidualLayer(nn.Module):
    def __init__(self, dim_cond, dim, expansion_factor, kernel_size=31, activation='PReLU', dropout=0.0):
        super().__init__()
        self.diffusion_projection = nn.Conv1d(dim, dim, 1)
        self.conditioner_projection = nn.Conv1d(dim_cond, dim, 1)
        self.convmodule = LYNXConvModule(dim=dim, expansion_factor=expansion_factor, kernel_size=kernel_size,
                                         activation=activation, dropout=dropout)

    def forward(self, x, conditioner, diffusion_step, front_cond_inject=False):
        if front_cond_inject:
            x = x + self.conditioner_projection(conditioner)
            res_x = x
        else:
            res_x = x
            x = x + self.conditioner_projection(conditioner)
        x = x + self.diffusion_projection(diffusion_step)
        x = x.transpose(1, 2)
        x = self.convmodule(x)  # (#batch, dim, length) 
        x = x.transpose(1, 2) + res_x
        return x  # (#batch, length, dim)


class LYNXNet(nn.Module):
    def __init__(self, in_dims, n_feats, hidden_size, *, num_layers=6, num_channels=512, expansion_factor=2, kernel_size=31,
                 activation='PReLU', dropout=0.0, strong_cond=False):
        """
        LYNXNet(Linear Gated Depthwise Separable Convolution Network)
        TIPS:You can control the style of the generated results by modifying the 'activation', 
            - 'PReLU'(default) : Similar to WaveNet
            - 'SiLU' : Voice will be more pronounced, not recommended for use under DDPM
            - 'ReLU' : Contrary to 'SiLU', Voice will be weakened
        """
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = Conv1d(in_dims * n_feats, num_channels, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNetResidualLayer(
                    dim_cond=hidden_size,
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    activation=activation,
                    dropout=dropout
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = Conv1d(num_channels, in_dims * n_feats, kernel_size=1)
        self.strong_cond = strong_cond
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
        if not self.strong_cond:
            x = F.gelu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(-1)

        for layer in self.residual_layers:
            x = layer(x, cond, diffusion_step, front_cond_inject=self.strong_cond)

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
