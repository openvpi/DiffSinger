import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from modules.fastspeech.tts_modules import FastSpeech2Encoder, mel2ph_to_dur
from utils.hparams import hparams
from utils.text_encoder import PAD_INDEX


class FastSpeech2Acoustic(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)
        self.dur_embed = Linear(1, hparams['hidden_size'])
        self.encoder = FastSpeech2Encoder(
            hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], ffn_act=hparams['ffn_act'],
            dropout=hparams['dropout'], num_heads=hparams['num_heads'],
            use_pos_embed=hparams['use_pos_embed'], rel_pos=hparams['rel_pos']
        )

        self.pitch_embed = Linear(1, hparams['hidden_size'])
        self.variance_embed_list = []
        self.use_energy_embed = hparams.get('use_energy_embed', False)
        self.use_breathiness_embed = hparams.get('use_breathiness_embed', False)
        if self.use_energy_embed:
            self.variance_embed_list.append('energy')
        if self.use_breathiness_embed:
            self.variance_embed_list.append('breathiness')

        self.use_variance_embeds = len(self.variance_embed_list) > 0
        if self.use_variance_embeds:
            self.variance_embeds = nn.ModuleDict({
                v_name: Linear(1, hparams['hidden_size'])
                for v_name in self.variance_embed_list
            })

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
                self.variance_embeds[v_name](variances[v_name][:, :, None])
                for v_name in self.variance_embed_list
            ], dim=-1).sum(-1)
            condition += variance_embeds

        if self.use_key_shift_embed:
            key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
            condition += key_shift_embed

        if self.use_speed_embed:
            speed_embed = self.speed_embed(speed[:, :, None])
            condition += speed_embed

        return condition

    def forward(
            self, txt_tokens, mel2ph, f0,
            key_shift=None, speed=None,
            spk_embed_id=None, **kwargs
    ):
        txt_embed = self.txt_embed(txt_tokens)
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
        dur_embed = self.dur_embed(dur[:, :, None])
        encoder_out = self.encoder(txt_embed, dur_embed, txt_tokens == 0)

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(encoder_out, 1, mel2ph_)

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
