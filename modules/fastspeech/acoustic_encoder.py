import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from modules.fastspeech.bbc_mask import fast_bbc_mask, fast_fast_bbc_mask
from modules.fastspeech.tts_modules import FastSpeech2Encoder, mel2ph_to_dur
from utils.hparams import hparams
from utils.phoneme_utils import PAD_INDEX


class FastSpeech2Acoustic(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)
        self.use_lang_id = hparams.get('use_lang_id', False)
        if self.use_lang_id:
            self.lang_embed = Embedding(hparams['num_lang'] + 1, hparams['hidden_size'], padding_idx=0)
        self.dur_embed = Linear(1, hparams['hidden_size'])
        self.encoder = FastSpeech2Encoder(
            hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], ffn_act=hparams['ffn_act'],
            dropout=hparams['dropout'], num_heads=hparams['num_heads'],
            use_pos_embed=hparams['use_pos_embed'], rel_pos=hparams.get('rel_pos', False), 
            use_rope=hparams.get('use_rope', False)
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

        self.use_bbc_encoder = hparams.get('use_bbc_encoder', False)
        if self.use_bbc_encoder:
            self.bbc_mask_len = hparams['bbc_mask_len']
            self.bbc_min_segment_length=hparams['bbc_min_segment_length']
            self.bbc_mask_prob=hparams['bbc_mask_prob']
            self.bbc_mask_emb=nn.Parameter(torch.randn(hparams['hidden_size']))

    def forward_variance_embedding(self, condition, key_shift=None, speed=None, **variances):
        if self.use_variance_embeds:
            variance_embeds = torch.stack([
                self.variance_embeds[v_name](variances[v_name][:, :, None]) 
                * self.variance_scaling_factor[v_name]
                for v_name in self.variance_embed_list
            ], dim=-1).sum(-1)
            condition += variance_embeds

        if self.use_key_shift_embed:
            key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
            key_shift_embed *= self.variance_scaling_factor['key_shift']
            condition += key_shift_embed

        if self.use_speed_embed:
            speed_embed = self.speed_embed(speed[:, :, None])
            speed_embed *= self.variance_scaling_factor['speed']
            condition += speed_embed

        return condition

    def forward(
            self, txt_tokens, mel2ph, f0,
            key_shift=None, speed=None,
            spk_embed_id=None, languages=None,
            **kwargs
    ):
        txt_embed = self.txt_embed(txt_tokens)
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1]).float()
        if self.use_variance_scaling:
            dur_embed = self.dur_embed(torch.log(1 + dur[:, :, None]))
        else:
            dur_embed = self.dur_embed(dur[:, :, None])
        if self.use_lang_id:
            lang_embed = self.lang_embed(languages)
            extra_embed = dur_embed + lang_embed
        else:
            extra_embed = dur_embed
        encoder_out = self.encoder(txt_embed, extra_embed, txt_tokens == 0)
        if self.use_bbc_encoder:

            encoder_out=torch.cat([self.bbc_mask_emb.unsqueeze(0).unsqueeze(0).expand(mel2ph.shape[0],1,encoder_out.shape[-1]),encoder_out],dim=1)
            encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
            mel2ph=fast_fast_bbc_mask(mel2ph,mask_length=self.bbc_mask_len,min_segment_length=self.bbc_min_segment_length,mask_prob=self.bbc_mask_prob)
            mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
            condition = torch.gather(encoder_out, 1, mel2ph_)
        else:
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
