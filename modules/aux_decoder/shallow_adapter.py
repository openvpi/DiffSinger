import torch
import torch.nn as nn

cls_map = {'fs2': 'modules.aux_decoder.fast_speech2_decoder.fs2_decode',
           'ns': 'modules.aux_decoder.noise_decoder.noise_decoder', 'ld': 'modules.aux_decoder.light_decoder.noise_decoder','att_fs2':'modules.aux_decoder.fs2_decoder.attention_fs2_decoder'
           ,'glow':'modules.aux_decoder.glow.glow_decoder','glow_convnext':'modules.aux_decoder.convnext_glow.glow_decoder_convnext','gglow':'modules.aux_decoder.gglow.glow_decoder','fast_speech2_decoders':'modules.aux_decoder.fast_speech2_decoders.fs2_decode'
           }
encoder_cls_map = {'fs2': 'modules.fastspeech.acoustic_encoder.FastSpeech2Acoustic'}


def build_object_from_class_name(cls_str, parent_cls, strict, *args, **kwargs):
    import importlib

    pkg = ".".join(cls_str.split(".")[:-1])
    cls_name = cls_str.split(".")[-1]
    cls_type = getattr(importlib.import_module(pkg), cls_name)
    if parent_cls is not None:
        assert issubclass(cls_type, parent_cls), f'| {cls_type} is not subclass of {parent_cls}.'
    if strict:
        return cls_type(*args, **kwargs)
    return cls_type(*args, **filter_kwargs(kwargs, cls_type))


def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect

    sig = inspect.signature(kwarg_obj)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys if
                     filter_key in dict_to_filter}
    return filtered_dict


class shallow_adapt(nn.Module):
    def __init__(self, parame, out_dims, vocab_size):
        super().__init__()
        self.parame = parame

        decodeparame = parame['shallow_diffusion_args']['aux_decoder_args']
        if decodeparame.get('encoder_hidden') is None:
            decodeparame['encoder_hidden'] = parame['hidden_size']
        decodeparame['out_dims'] = out_dims
        decodeparame['parame'] = parame

        encoderparame = parame['shallow_diffusion_args']['aux_encoder_args']
        encoderparame['parame'] = parame
        encoderparame['vocab_size'] = vocab_size
        self.decoder = build_object_from_class_name(cls_map[parame['shallow_diffusion_args']['aux_decoder_arch']],
                                                    nn.Module,
                                                    parame['shallow_diffusion_args']['aux_decoder_strict_hparams'],
                                                    **decodeparame)

        if not parame['shallow_diffusion_args']['aux_share_encoder']:
            # todo
            self.use_encoder = True
            self.encoder = build_object_from_class_name(
                encoder_cls_map[parame['shallow_diffusion_args']['aux_encoder_arch']],
                nn.Module,
                parame['shallow_diffusion_args']['aux_encoder_strict_hparams'],
                **encoderparame)
        else:
            self.use_encoder = False

    def forward(self, condition, infer=False, txt_tokens=None, mel2ph=None, f0=None,
                key_shift=None, speed=None,
                spk_embed_id=None,gt_mel=None,mask=None, **kwargs):

        if self.use_encoder:
            condition = self.encoder(txt_tokens=txt_tokens, mel2ph=mel2ph, f0=f0,
                                     key_shift=key_shift, speed=speed,
                                     spk_embed_id=spk_embed_id, **kwargs)

        return self.decoder(condition, infer,gt_mel,mask)

    def get_loss(self):
        return self.decoder.build_loss()
