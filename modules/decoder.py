from dataclasses import dataclass

import torch
from torch import nn

from lib.conf.schema import DiffusionDecoderConfig
from utils import filter_kwargs
from .aux_decoder import ConvNeXtDecoder
from .backbone import WaveNet, LYNXNet
from .core import RectifiedFlow

__all__ = [
    "DiffusionDecoder",
    "ShallowDiffusionOutput",
]

from .normalizer import FeatureNormalizer

AUX_DECODERS = {
    "convnext": ConvNeXtDecoder,
}
BACKBONES = {
    'wavenet': WaveNet,
    'lynxnet': LYNXNet,
}


@dataclass
class ShallowDiffusionOutput:
    """
    Output of shallow diffusion decoder.
    """
    aux_out: torch.Tensor
    diff_out: torch.Tensor
    norm_gt: torch.Tensor = None


class DiffusionDecoder(nn.Module):
    def __init__(
            self, sample_dim: int, condition_dim: int,
            normalizer: FeatureNormalizer, config: DiffusionDecoderConfig
    ):
        super().__init__()
        self.normalizer = normalizer
        self.use_shallow_diffusion = config.use_shallow_diffusion
        if self.use_shallow_diffusion:
            self.aux_decoder_grad = config.aux_decoder_grad
            self.aux_decoder = (cls := AUX_DECODERS[config.aux_decoder_arch])(
                condition_dim, sample_dim, **filter_kwargs(config.aux_decoder_kwargs, cls)
            )
        self.decoder = RectifiedFlow(
            sample_dim=sample_dim,
            backbone=(cls := BACKBONES[config.backbone_arch])(
                sample_dim, condition_dim, **filter_kwargs(config.backbone_kwargs, cls)
            ),
            time_scale_factor=config.time_scale_factor
        )
        self.sampling_algorithm = config.sampling_algorithm
        self.sampling_steps = config.sampling_steps

    def forward(self, condition, sample_gt=None, infer=True):
        if self.use_shallow_diffusion:
            aux_cond = condition * self.aux_decoder_grad + condition.detach() * (1 - self.aux_decoder_grad)
            aux_out = self.aux_decoder(aux_cond)
        else:
            aux_out = None
        if infer:
            diff_out = self.decoder(
                condition, x_src=aux_out, infer=infer,
                sampling_algorithm=self.sampling_algorithm,
                sampling_steps=self.sampling_steps
            )
            aux_sample_pred = self.normalizer.denorm(aux_out) if aux_out is not None else None
            diff_sample_pred = self.normalizer.denorm(diff_out)
            return ShallowDiffusionOutput(aux_out=aux_sample_pred, diff_out=diff_sample_pred)
        else:
            if self.normalizer.num_features == 1:
                sample_gt = [sample_gt]
            norm_gt = self.normalizer.norm(*sample_gt)
            diff_out = self.decoder(condition, x_gt=norm_gt, infer=infer)
            return ShallowDiffusionOutput(
                aux_out=aux_out, diff_out=diff_out, norm_gt=norm_gt
            )
