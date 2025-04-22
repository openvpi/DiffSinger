import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb, SwiGLU, Transpose
from utils.hparams import hparams


class LYNXNet2Block(nn.Module):
    def __init__(self, dim, expansion_factor, kernel_size=31, dropout=0.):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        if float(dropout) > 0.:
            _dropout = nn.Dropout(dropout)
        else:
            _dropout = nn.Identity()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
            Transpose((1, 2)),
            nn.Linear(dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, inner_dim * 2),
            SwiGLU(),
            nn.Linear(inner_dim, dim),
            _dropout
        )

    def forward(self, x):
        return x + self.net(x)


class LYNXNet2(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, expansion_factor=1, kernel_size=31,
                 dropout=0.0):
        """
        LYNXNet2(Linear Gated Depthwise Separable Convolution Network Version 2)
        """
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)
        self.conditioner_projection = nn.Linear(hparams['hidden_size'], num_channels)
        # It may need to be modified at some point to be compatible with the condition cache
        # self.conditioner_projection = nn.Conv1d(hparams['hidden_size'], num_channels, 1)
        self.diffusion_embedding = nn.Sequential(
            SinusoidalPosEmb(num_channels),
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        self.residual_layers = nn.ModuleList(
            [
                LYNXNet2Block(
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, in_dims * n_feats)
        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.conditioner_projection.weight)
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

        x = self.input_projection(x.transpose(1, 2)) # [B, T, F x M]
        x = x + self.conditioner_projection(cond.transpose(1, 2))
        # It may need to be modified at some point to be compatible with the condition cache
        # x = x + self.conditioner_projection(cond.transpose(1, 2))
        x = x + self.diffusion_embedding(diffusion_step).unsqueeze(1)

        for layer in self.residual_layers:
            x = layer(x)

        # post-norm
        x = self.norm(x)

        # output projection
        x = self.output_projection(x).transpose(1, 2)  # [B, 128, T]

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=1, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
