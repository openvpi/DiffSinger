import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
    SinusoidalPosEmb,
)
from modules.fastspeech.tts_modules import FastSpeech2Encoder, mel2ph_to_dur, StretchRegulator
from utils.hparams import hparams
from utils.phoneme_utils import PAD_INDEX


class FastSpeech2Acoustic(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)
        self.use_lang_id = hparams.get('use_lang_id', False)
        if self.use_lang_id:
            self.lang_embed = Embedding(hparams['num_lang'] + 1, hparams['hidden_size'], padding_idx=0)

        self.use_stretch_embed = hparams.get('use_stretch_embed', None)
        assert self.use_stretch_embed is not None, "You may be loading an old version of the model checkpoint, which is incompatible with the new version due to some bug fixes. It is recommended to roll back to the old version (commit id: 6df0ee977c3728f14cb79c2db8b19df30b23a0bf)"
        if self.use_stretch_embed:
            self.sr = StretchRegulator()
            self.stretch_embed = nn.Sequential(
                SinusoidalPosEmb(hparams['hidden_size']),
                nn.Linear(hparams['hidden_size'], hparams['hidden_size'] * 4),
                nn.GELU(),
                nn.Linear(hparams['hidden_size'] * 4, hparams['hidden_size']),
            )
            self.stretch_embed_rnn = nn.GRU(hparams['hidden_size'], hparams['hidden_size'], 1, batch_first=True)

        self.dur_embed = Linear(1, hparams['hidden_size'])
        self.encoder = FastSpeech2Encoder(
            hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], ffn_act=hparams['ffn_act'],
            dropout=hparams['dropout'], num_heads=hparams['num_heads'],
            use_pos_embed=hparams['use_pos_embed'], rel_pos=hparams.get('rel_pos', False), 
            use_rope=hparams.get('use_rope', False), rope_interleaved=hparams.get('rope_interleaved', True),
            use_gate_attn=hparams.get('use_gate_attn', False), use_qk_norm=hparams.get('use_qk_norm', False)
        )

        self.pitch_embed = Linear(1, hparams['hidden_size'])
        self.variance_embed_list = []
        self.use_energy_embed = hparams.get('use_energy_embed', False)
        self.use_breathiness_embed = hparams.get('use_breathiness_embed', False)
        self.use_voicing_embed = hparams.get('use_voicing_embed', False)
        self.use_tension_embed = hparams.get('use_tension_embed', False)
        if self.use_energy_embed:
            self.variance_embed_list.append('energy')
        if self.use_breathiness_embed:
            self.variance_embed_list.append('breathiness')
        if self.use_voicing_embed:
            self.variance_embed_list.append('voicing')
        if self.use_tension_embed:
            self.variance_embed_list.append('tension')

        self.use_variance_embeds = len(self.variance_embed_list) > 0
        if self.use_variance_embeds:
            self.variance_embeds = nn.ModuleDict({
                v_name: Linear(1, hparams['hidden_size'])
                for v_name in self.variance_embed_list
            })

        self.use_variance_scaling = hparams.get('use_variance_scaling', False)
        if self.use_variance_scaling:
            self.variance_scaling_factor = {
                'energy': 1. / 96,
                'breathiness': 1. / 96,
                'voicing': 1. / 96,
                'tension': 0.1,
                'key_shift': 1. / 12,
                'speed': 1.
            }
        else:
            self.variance_scaling_factor = {
                'energy': 1.,
                'breathiness': 1.,
                'voicing': 1.,
                'tension': 1.,
                'key_shift': 1.,
                'speed': 1.
            }

        self.use_key_shift_embed = hparams.get('use_key_shift_embed', False)
        if self.use_key_shift_embed:
            self.key_shift_embed = Linear(1, hparams['hidden_size'])

        self.use_speed_embed = hparams.get('use_speed_embed', False)
        if self.use_speed_embed:
            self.speed_embed = Linear(1, hparams['hidden_size'])

        self.use_spk_id = hparams['use_spk_id']
        if self.use_spk_id:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])

    def forward_variance_embedding(self, condition, key_shift=None, speed=None, **variances):
        if self.use_variance_embeds:
            variance_embeds = torch.stack([
                self.variance_embeds[v_name](variances[v_name][:, :, None] * self.variance_scaling_factor[v_name])
                for v_name in self.variance_embed_list
            ], dim=-1).sum(-1)
            condition += variance_embeds

        if self.use_key_shift_embed:
            key_shift_embed = self.key_shift_embed(key_shift[:, :, None] * self.variance_scaling_factor['key_shift'])
            condition += key_shift_embed

        if self.use_speed_embed:
            speed_embed = self.speed_embed(speed[:, :, None] * self.variance_scaling_factor['speed'])
            condition += speed_embed

        return condition

    def forward(
            self, txt_tokens, mel2ph, f0,
            key_shift=None, speed=None,
            spk_embed_id=None, languages=None,
            **kwargs
    ):
        txt_embed = self.txt_embed(txt_tokens)
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1])
        if self.use_variance_scaling:
            dur_embed = self.dur_embed(torch.log(1 + dur[:, :, None].float()))
        else:
            dur_embed = self.dur_embed(dur[:, :, None].float())
        if self.use_lang_id:
            lang_embed = self.lang_embed(languages)
            extra_embed = dur_embed + lang_embed
        else:
            extra_embed = dur_embed
        encoder_out = self.encoder(txt_embed, extra_embed, txt_tokens == 0)

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(encoder_out, 1, mel2ph_)

        if self.use_stretch_embed:
            stretch = torch.round(1000 * self.sr(mel2ph, dur))
            if self.training and stretch.numel() > 1000:
                # construct a phoneme stretching index lookup table with a total of 1001 indexes (0~1000)
                table = self.stretch_embed(torch.arange(0, 1001, device=stretch.device))
                stretch_embed = torch.index_select(table, 0, stretch.view(-1).long()).view_as(condition)
            else:
                stretch_embed = self.stretch_embed(stretch)
            condition += stretch_embed
            self.stretch_embed_rnn.flatten_parameters()
            stretch_embed_rnn_out, _ = self.stretch_embed_rnn(condition)
            condition = condition + stretch_embed_rnn_out

        if self.use_spk_id:
            spk_mix_embed = kwargs.get('spk_mix_embed')
            if spk_mix_embed is not None:
                spk_embed = spk_mix_embed
            else:
                spk_embed = self.spk_embed(spk_embed_id)[:, None, :]
            condition += spk_embed

        f0_mel = (1 + f0 / 700).log()
        pitch_embed = self.pitch_embed(f0_mel[:, :, None])
        condition += pitch_embed

        condition = self.forward_variance_embedding(
            condition, key_shift=key_shift, speed=speed, **kwargs
        )

        return condition
