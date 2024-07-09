import torch.nn
from torch import nn

from .convnext import ConvNeXtDecoder
from .LYNXNetDecoder import LYNXNetDecoder
from utils import filter_kwargs

AUX_DECODERS = {
    'convnext': ConvNeXtDecoder, 
    'lynxnet': LYNXNetDecoder
}
AUX_LOSSES = {
    'convnext': nn.L1Loss, 
    'lynxnet': nn.L1Loss
}


def build_aux_decoder(
        in_dims: int, out_dims: int,
        aux_decoder_arch: str, aux_decoder_args: dict
) -> torch.nn.Module:
    decoder_cls = AUX_DECODERS[aux_decoder_arch]
    kwargs = filter_kwargs(aux_decoder_args, decoder_cls)
    return AUX_DECODERS[aux_decoder_arch](in_dims, out_dims, **kwargs)


def build_aux_loss(aux_decoder_arch):
    return AUX_LOSSES[aux_decoder_arch]()


class AuxDecoderAdaptor(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, num_feats: int,
                 spec_min: list, spec_max: list,
                 aux_decoder_arch: str, aux_decoder_args: dict):
        super().__init__()
        self.decoder = build_aux_decoder(
            in_dims=in_dims, out_dims=out_dims * num_feats,
            aux_decoder_arch=aux_decoder_arch,
            aux_decoder_args=aux_decoder_args
        )
        self.out_dims = out_dims
        self.n_feats = num_feats
        if spec_min is not None and spec_max is not None:
            # spec: [B, T, M] or [B, F, T, M]
            # spec_min and spec_max: [1, 1, M] or [1, 1, F, M] => transpose(-3, -2) => [1, 1, M] or [1, F, 1, M]
            spec_min = torch.FloatTensor(spec_min)[None, None, :].transpose(-3, -2)
            spec_max = torch.FloatTensor(spec_max)[None, None, :].transpose(-3, -2)
            self.register_buffer('spec_min', spec_min, persistent=False)
            self.register_buffer('spec_max', spec_max, persistent=False)

    def norm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.
        b = (self.spec_max + self.spec_min) / 2.
        return (x - b) / k

    def denorm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.
        b = (self.spec_max + self.spec_min) / 2.
        return x * k + b

    def forward(self, condition, infer=False):
        x = self.decoder(condition, infer=infer)  # [B, T, F x C]

        if self.n_feats > 1:
            # This is the temporary solution since PyTorch 1.13
            # does not support exporting aten::unflatten to ONNX
            # x = x.unflatten(dim=2, sizes=(self.n_feats, self.in_dims))
            x = x.reshape(-1, x.shape[1], self.n_feats, self.out_dims)  # [B, T, F, C]
            x = x.transpose(1, 2)  # [B, F, T, C]
        if infer:
            x = self.denorm_spec(x)

        return x  # [B, T, C] or [B, F, T, C]
