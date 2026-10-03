from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import AdamWConv1d, AdamWDWConv1d, MixedPrecisionLayerNorm, in_export_or_trace

# Above this many frames the channels-last depthwise-conv path is slower than Conv1d
# in training (cuDNN depthwise backward regresses on long sequences). Export always
# uses the channels-last path: ORT's CPU kernel for 1D depthwise Conv is far slower
# than the 2D one, and DirectML is indifferent.
_NHWC_MAX_T = 4096


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
            self,
            dim: int,
            intermediate_dim: int,
            layer_scale_init_value: Optional[float] = None, drop_out: float = 0.0

    ):
        super().__init__()
        self.dwconv = AdamWDWConv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv

        self.norm = MixedPrecisionLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.dropout = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, nhwc: bool = False) -> torch.Tensor:
        if not nhwc:
            # original layout: x (B, C, T)
            residual = x
            x = self.dwconv(x)
            x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
            x = self.dropout(x)

            x = residual + self.drop_path(x)
            return x
        # channels-last layout: x (B, T, C), fed by ConvNeXtDecoder._forward_nhwc
        residual = x
        x = self._dwconv_nhwc(x)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = self.dropout(x)

        x = residual + self.drop_path(x)
        return x

    def _dwconv_nhwc(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise conv over [B, T, C] as channels-last Conv2d over zero-copy views,
        # keeping the [B, T, C] layout the surrounding Linears consume.
        dw = self.dwconv
        w = dw.weight.view(dw.weight.size(0), 1, 1, dw.weight.size(2))
        w = w.contiguous(memory_format=torch.channels_last)
        h = F.conv2d(x.permute(0, 2, 1).unsqueeze(2), w, dw.bias,
                     padding=(0, dw.padding[0]), groups=dw.groups)
        return h.squeeze(2).permute(0, 2, 1)


class ConvNeXtDecoder(nn.Module):
    def __init__(
            self, in_dims, out_dims, /, *,
            num_channels=512, num_layers=6, kernel_size=7, dropout_rate=0.1
    ):
        super().__init__()
        self.inconv = nn.Conv1d(
            in_dims, num_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )
        self.conv = nn.ModuleList(
            ConvNeXtBlock(
                dim=num_channels, intermediate_dim=num_channels * 4,
                layer_scale_init_value=1e-6, drop_out=dropout_rate
            ) for _ in range(num_layers)
        )
        self.outconv = AdamWConv1d(
            num_channels, out_dims, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2
        )

    # noinspection PyUnusedLocal
    def forward(self, x, infer=False):
        # in_export_or_trace() short-circuits the shape check during graph capture, so
        # the exported graph always takes the channels-last path without shape guards.
        if in_export_or_trace() or x.shape[1] <= _NHWC_MAX_T:
            return self._forward_nhwc(x)
        x = x.transpose(1, 2)
        x = self.inconv(x)
        for conv in self.conv:
            x = conv(x)
        x = self.outconv(x)
        x = x.transpose(1, 2)
        return x

    def _forward_nhwc(self, x):
        # Keeps [B, T, C] layout through the whole stack: inconv/outconv run as
        # channels-last Conv2d over zero-copy views, blocks take nhwc=True.
        inconv = self.inconv
        w = inconv.weight.view(inconv.weight.size(0), inconv.weight.size(1), 1, inconv.weight.size(2))
        w = w.contiguous(memory_format=torch.channels_last)
        h = F.conv2d(x.permute(0, 2, 1).unsqueeze(2), w, inconv.bias,
                     padding=(0, inconv.padding[0]))
        h = h.squeeze(2).permute(0, 2, 1)
        for conv in self.conv:
            h = conv(h, nhwc=True)
        outconv = self.outconv
        w = outconv.weight.view(outconv.weight.size(0), outconv.weight.size(1), 1, outconv.weight.size(2))
        w = w.contiguous(memory_format=torch.channels_last)
        h = F.conv2d(h.permute(0, 2, 1).unsqueeze(2), w, outconv.bias,
                     padding=(0, outconv.padding[0]))
        return h.squeeze(2).permute(0, 2, 1)
