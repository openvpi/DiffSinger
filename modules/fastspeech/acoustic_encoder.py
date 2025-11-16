import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
    SinusoidalPosEmb,
    EncSALayer,
)
from modules.fastspeech.tts_modules import FastSpeech2Encoder, mel2ph_to_dur, StretchRegulator
from utils.hparams import hparams
from utils.phoneme_utils import PAD_INDEX


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = Linear(input_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class FiLM(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(cond_dim, input_dim)
        self.beta = nn.Linear(cond_dim, input_dim)

    def forward(self, x, cond):
        """
        x: [B, T, H]
        cond: [B, T, cond_dim]
        returns: FiLM modulated x
        """
        gamma = self.gamma(cond).unsqueeze(-1)  # [B, H, 1]
        beta = self.beta(cond).unsqueeze(-1)   # [B, H, 1]
        return gamma * x + beta


class FastSpeech2Acoustic(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        hidden_size = hparams['hidden_size']
        self.txt_embed = Embedding(vocab_size, hidden_size, PAD_INDEX)
        
        self.use_lang_id = hparams.get('use_lang_id', False)
        if self.use_lang_id:
            self.lang_embed = Embedding(hparams['num_lang'] + 1, hidden_size, padding_idx=0)

        self.use_stretch_embed = hparams.get('use_stretch_embed', None)
        assert self.use_stretch_embed is not None, "You may be loading an old version of the model checkpoint, which is incompatible with the new version due to some bug fixes. It is recommended to roll back to the old version (commit id: 6df0ee977c3728f14cb79c2db8b19df30b23a0bf)"
        if self.use_stretch_embed:
            self.sr = StretchRegulator()
            self.stretch_embed = nn.Sequential(
                SinusoidalPosEmb(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )
            self.stretch_embed_rnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)

        self.dur_embed = Linear(1, hidden_size)
        self.encoder = FastSpeech2Encoder(
            hidden_size=hidden_size, num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'], ffn_act=hparams['ffn_act'],
            dropout=hparams['dropout'], num_heads=hparams['num_heads'],
            use_pos_embed=hparams['use_pos_embed'], rel_pos=hparams.get('rel_pos', False), 
            use_rope=hparams.get('use_rope', False), use_mixln=hparams.get('use_mixln', False)
        )

        self.variance_embed_list = []
        self.use_energy_embed = hparams.get('use_energy_embed', False)
        self.use_breathiness_embed = hparams.get('use_breathiness_embed', False)
        self.use_voicing_embed = hparams.get('use_voicing_embed', False)
        self.use_tension_embed = hparams.get('use_tension_embed', False)
        if self.use_energy_embed: self.variance_embed_list.append('energy')
        if self.use_breathiness_embed: self.variance_embed_list.append('breathiness')
        if self.use_voicing_embed: self.variance_embed_list.append('voicing')
        if self.use_tension_embed: self.variance_embed_list.append('tension')
        
        self.use_variance_embeds = len(self.variance_embed_list) > 0
        self.use_key_shift_embed = hparams.get('use_key_shift_embed', False)
        self.use_speed_embed = hparams.get('use_speed_embed', False)

        self.use_spk_id = hparams['use_spk_id']
        if self.use_spk_id:
            self.use_mix_ln = hparams.get('use_mixln', False)
            if self.use_mix_ln:
                self.mix_ln_blacklist = set(hparams.get('mix_ln_blacklist', []))
                self.mix_ln_mask_id = hparams['num_spk']
                self.spk_embed = Embedding(hparams['num_spk'] + 1, hidden_size)
            else:
                self.spk_embed = Embedding(hparams['num_spk'], hidden_size)

        self.use_film = hparams.get('use_film', False)

        if self.use_film:
            self.pitch_embed_film = MLP(1, hidden_size)
            
            if self.use_variance_embeds:
                self.variance_embeds_film = nn.ModuleDict({
                    v_name: MLP(1, hidden_size) for v_name in self.variance_embed_list
                })
            
            if self.use_key_shift_embed: self.key_shift_embed_film = MLP(1, hidden_size)
            if self.use_speed_embed: self.speed_embed_film = MLP(1, hidden_size)

            cond_dim = hidden_size
            if self.use_variance_embeds: cond_dim += len(self.variance_embed_list) * hidden_size
            if self.use_key_shift_embed: cond_dim += hidden_size
            if self.use_speed_embed: cond_dim += hidden_size
            self.film = FiLM(hidden_size, cond_dim)

            self.final_encoder = nn.ModuleList([
                EncSALayer(
                    hidden_size, hparams.get('film_enc_num_heads', 2),
                    dropout=hparams['dropout'],
                    attention_dropout=hparams.get('film_enc_attn_dropout', 0.1),
                    relu_dropout=hparams.get('film_enc_relu_dropout', 0.1),
                    kernel_size=hparams.get('film_enc_kernel_size', 3),
                    act=hparams['ffn_act']
                ) for _ in range(hparams.get('film_enc_layers', 1))
            ])
            
            self.final_ln = nn.LayerNorm(hidden_size)
        else:
            self.pitch_embed = Linear(1, hidden_size)
            if self.use_variance_embeds:
                self.variance_embeds = nn.ModuleDict({
                    v_name: Linear(1, hidden_size) for v_name in self.variance_embed_list
                })
            if self.use_key_shift_embed: self.key_shift_embed = Linear(1, hidden_size)
            if self.use_speed_embed: self.speed_embed = Linear(1, hidden_size)

        self.use_variance_scaling = hparams.get('use_variance_scaling', False)
        if self.use_variance_scaling:
            self.variance_scaling_factor = {
                'energy': 1. / 96, 'breathiness': 1. / 96, 'voicing': 1. / 96,
                'tension': 0.1, 'key_shift': 1. / 12, 'speed': 1.
            }
        else:
            self.variance_scaling_factor = {
                'energy': 1., 'breathiness': 1., 'voicing': 1.,
                'tension': 1., 'key_shift': 1., 'speed': 1.
            }

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
        spk_embed=None
        mixln_mask_embed = None
        if self.use_spk_id:
            spk_mix_embed = kwargs.get('spk_mix_embed')
            if spk_mix_embed is not None: spk_embed = spk_mix_embed
            else: spk_embed = self.spk_embed(spk_embed_id)
            if self.training and self.use_mix_ln and self.mix_ln_blacklist:
                blacklist_mask = torch.tensor([sid.item() in self.mix_ln_blacklist for sid in spk_embed_id], device=spk_embed_id.device)
                if blacklist_mask.any():
                    mask_id_tensor = torch.tensor(self.mix_ln_mask_id, device=spk_embed_id.device)
                    mask_embedding_vector = self.spk_embed(mask_id_tensor)
                    mixln_mask_embed = torch.zeros_like(spk_embed)
                    mixln_mask_embed[blacklist_mask] = mask_embedding_vector
            else: mixln_mask_embed = None

        txt_embed = self.txt_embed(txt_tokens)
        dur = mel2ph_to_dur(mel2ph, txt_tokens.shape[1])
        dur_embed = self.dur_embed(torch.log(1 + dur[:, :, None].float()) if self.use_variance_scaling else dur[:, :, None].float())
        extra_embed = dur_embed + self.lang_embed(languages) if self.use_lang_id else dur_embed
        encoder_out = self.encoder(txt_embed, extra_embed, txt_tokens == 0, spk_embed, mixln_mask_embed=mixln_mask_embed)

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, encoder_out.shape[-1]])
        condition = torch.gather(encoder_out, 1, mel2ph_)

        if self.use_stretch_embed:
            stretch = torch.round(1000 * self.sr(mel2ph, dur))
            if self.training and stretch.numel() > 1000:
                table = self.stretch_embed(torch.arange(0, 1001, device=stretch.device))
                stretch_embed = torch.index_select(table, 0, stretch.view(-1).long()).view_as(condition)
            else:
                stretch_embed = self.stretch_embed(stretch)
            condition += stretch_embed
            self.stretch_embed_rnn.flatten_parameters()
            stretch_embed_rnn_out, _ = self.stretch_embed_rnn(condition)
            condition = condition + stretch_embed_rnn_out

        if self.use_film:
            film_x = condition
            all_conds = []
            
            f0_mel = (1 + f0 / 700).log()
            f0_embed = self.pitch_embed_film(f0_mel[:, :, None])
            
            if self.use_spk_id:
                spk_embed_expanded = spk_mix_embed if spk_mix_embed is not None else spk_embed[:, None, :]
                spk_embed_expanded = spk_embed_expanded.expand(-1, film_x.size(1), -1)
                spk_f0_cond = spk_embed_expanded + f0_embed
            else:
                spk_f0_cond = f0_embed
            all_conds.append(spk_f0_cond)

            if self.use_variance_embeds:
                for v_name in self.variance_embed_list:
                    variance_input = kwargs[v_name][:, :, None] * self.variance_scaling_factor[v_name]
                    all_conds.append(self.variance_embeds_film[v_name](variance_input))
            if self.use_key_shift_embed:
                key_shift_input = key_shift[:, :, None] * self.variance_scaling_factor['key_shift']
                all_conds.append(self.key_shift_embed_film(key_shift_input))
            if self.use_speed_embed:
                speed_input = speed[:, :, None] * self.variance_scaling_factor['speed']
                all_conds.append(self.speed_embed_film(speed_input))
            
            film_cond = torch.cat(all_conds, dim=-1)
            condition = self.film(film_x, film_cond)

            txt_padding_mask = (txt_tokens == 0)
            mel_padding_mask = torch.gather(F.pad(txt_padding_mask, [1, 0], value=False), 1, mel2ph)
            for layer in self.final_encoder:
                condition = layer(condition, encoder_padding_mask=mel_padding_mask)

            condition = self.final_ln(condition)
        else:
            if self.use_spk_id:
                spk_embed_expanded = spk_mix_embed if spk_mix_embed is not None else spk_embed[:, None, :]
                condition += spk_embed_expanded
            
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])
            condition += pitch_embed

            condition = self.forward_variance_embedding(condition, key_shift=key_shift, speed=speed, **kwargs)
            
        return condition
