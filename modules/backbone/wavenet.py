import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb


class Conv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        self.conditioner_projection = nn.Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    def __init__(
            self, sample_dim: int, condition_dim: int,
            *,
            num_layers=20, num_channels=256, dilation_cycle_length=4
    ):
        super().__init__()
        self.sample_dim = sample_dim
        self.input_projection = Conv1d(sample_dim, num_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.Mish(),
            nn.Linear(num_channels * 4, num_channels)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                encoder_hidden=condition_dim,
                residual_channels=num_channels,
                dilation=2 ** (i % dilation_cycle_length)
            )
            for i in range(num_layers)
        ])
        self.skip_projection = Conv1d(num_channels, num_channels, 1)
        self.output_projection = Conv1d(num_channels, sample_dim, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """
        x = spec.squeeze(1)  # [B, M, T]
        x = self.input_projection(x)  # [B, C, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, M, T]
        x = x[:, None, :, :]

        return x
