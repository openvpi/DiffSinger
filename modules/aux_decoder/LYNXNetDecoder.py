# refer to： 
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/model_conformer_naive.py
# https://github.com/CNChTu/Diffusion-SVC/blob/v2.0_dev/diffusion/naive_v2/naive_v2_diff.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.backbones.LYNXNet import LYNXConvModule


class LYNXNetDecoderLayer(nn.Module):
    """
    LYNXNet Decoder Layer

    Args:
        dim (int): Dimension of model
        expansion_factor (int): Expansion factor of conv module, default 2
        kernel_size (int): Kernel size of conv module, default 31
        in_norm (bool): Whether to use norm
        activation (str): Activation Function for conv module
    """

    def __init__(self, dim, expansion_factor, kernel_size=31, in_norm=False, activation='SiLU'):
        super().__init__()
        self.convmodule = LYNXConvModule(dim=dim, expansion_factor=expansion_factor, kernel_size=kernel_size, in_norm=in_norm, activation=activation)

    def forward(self, x) -> torch.Tensor:
        residual = x
        x = self.convmodule(x)
        x = residual + x
        
        return x


class LYNXNetDecoder(nn.Module):
    def __init__(
            self, in_dims, out_dims, /, *,
            num_channels=512, num_layers=6, kernel_size=31, dropout_rate=None
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(in_dims, num_channels, 1)
        self.encoder_layers = nn.ModuleList(
                LYNXNetDecoderLayer(
                    dim=num_channels, 
                    expansion_factor=2, 
                    kernel_size=kernel_size, 
                    in_norm=False, 
                    activation='SiLU') for _ in range(num_layers)
        )
        self.output_projection = nn.Conv1d(num_channels, out_dims, kernel_size=1)

    def forward(self, x, infer=False):
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, in_dims)
        return:
            torch.Tensor: Output tensor (#batch, length, out_dims)
        """
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.transpose(1, 2)
        x = self.output_projection(x)
        x = x.transpose(1, 2)
        
        return x