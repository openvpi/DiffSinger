import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import NormalInitEmbedding as Embedding
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from modules.fastspeech.variance_encoder import FastSpeech2Variance
from utils.hparams import hparams
from utils.phoneme_utils import PAD_INDEX

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def uniform_attention_pooling(spk_embed, durations):
    _, T_mel, _ = spk_embed.shape
    ph_starts = torch.cumsum(torch.cat([torch.zeros_like(durations[:, :1]), durations[:, :-1]], dim=1), dim=1)
    ph_ends = ph_starts + durations
    mel_indices = torch.arange(T_mel, device=spk_embed.device).view(1, 1, T_mel)
    phoneme_to_mel_mask = (mel_indices >= ph_starts.unsqueeze(-1)) & (mel_indices < ph_ends.unsqueeze(-1))
    uniform_scores = phoneme_to_mel_mask.float()
    sum_scores = uniform_scores.sum(dim=2, keepdim=True)
    attn_weights = uniform_scores / (sum_scores + (sum_scores == 0).float())  # [B, T_ph, T_mel]
    ph_spk_embed = torch.bmm(attn_weights, spk_embed)

    return ph_spk_embed


def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    return f0_coarse


class LengthRegulator(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, dur):
        token_idx = torch.arange(1, dur.shape[1] + 1, device=dur.device)[None, :, None]
        dur_cumsum = torch.cumsum(dur, dim=1)
        dur_cumsum_prev = F.pad(dur_cumsum, (1, -1), mode='constant', value=0)
        pos_idx = torch.arange(dur.sum(dim=1).max(), device=dur.device)[None, None]
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask).sum(dim=1)
        return mel2ph


class FastSpeech2AcousticONNX(FastSpeech2Acoustic):
    def __init__(self, vocab_size, cross_lingual_token_idx=None):
        super().__init__(vocab_size=vocab_size)
        self.register_buffer(
            'cross_lingual_token_idx',
            torch.LongTensor(cross_lingual_token_idx),
            persistent=False
        )  # [N,]
        if len(cross_lingual_token_idx) == 0:
            self.use_lang_id = False

        # for temporary compatibility; will be completely removed in the future
        self.f0_embed_type = hparams.get('f0_embed_type', 'continuous')
        if self.f0_embed_type == 'discrete':
            self.pitch_embed = Embedding(300, hparams['hidden_size'], PAD_INDEX)

        self.lr = LengthRegulator()
        if hparams['use_key_shift_embed']:
            self.shift_min, self.shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
        if hparams['use_speed_embed']:
            self.speed_min, self.speed_max = hparams['augmentation_args']['random_time_stretching']['range']

    # noinspection PyMethodOverriding
    def forward(
            self, tokens, durations,
            f0, variances: dict,
            gender=None, velocity=None,
            spk_embed=None,
            languages=None,
            retake=None, gt_mel=None,
            tokens_b=None, blend=None
    ):
        durations = durations * (tokens > 0)
        mel2ph = self.lr(durations)
        _mel2ph = mel2ph
        f0 = f0 * (mel2ph > 0)
        mel2ph = mel2ph[..., None].repeat((1, 1, hparams['hidden_size']))
        if self.use_variance_scaling:
            dur_embed = self.dur_embed(torch.log(1 + durations.float())[:, :, None])
        else:
            dur_embed = self.dur_embed(durations.float()[:, :, None])
        if self.use_lang_id:
            lang_mask = torch.any(
                tokens[..., None] == self.cross_lingual_token_idx[None, None],
                dim=-1
            )
            lang_embed = self.lang_embed(languages * lang_mask)
            extra_embed = dur_embed + lang_embed
        else:
            extra_embed = dur_embed
        if hparams.get('use_mix_ln', False):
            if hasattr(self, 'frozen_spk_embed'):
                ph_spk_embed = self.frozen_spk_embed.repeat(1, tokens.shape[1], 1)
            else:
                ph_spk_embed = uniform_attention_pooling(spk_embed, durations)
        else:
            ph_spk_embed = None

        # Phoneme mix (P3 envelope, experiment): base phoneme stream + S target streams, encoded in
        #   ONE batched encoder pass, expanded to frames on the SHARED mel2ph, convex-combined PER FRAME.
        #   tokens_b: [S, n_tokens] (targets); blend: [S, n_frames] (per-target per-frame weight);
        #   base weight = 1 - sum_S(blend), clamped >=0. blend all-zeros => bit-identical to no-mix.
        #   extra_embed / spk_embed / mel2ph are the base's and are broadcast (shared) across streams.
        if tokens_b is None or blend is None:
            encoded = self.encoder(self.txt_embed(tokens), extra_embed, tokens == PAD_INDEX, spk_embed=ph_spk_embed)
            encoded = F.pad(encoded, (0, 0, 1, 0))
            condition = torch.gather(encoded, 1, mel2ph)
        else:
            n_mix = tokens_b.shape[0]
            tokens_all = torch.cat([tokens, tokens_b], dim=0)                      # [1 + S, n_tokens]
            extra_all = extra_embed.expand(1 + n_mix, -1, -1)
            spk_all = ph_spk_embed.expand(1 + n_mix, -1, -1) if ph_spk_embed is not None else None
            encoded_all = self.encoder(self.txt_embed(tokens_all), extra_all, tokens_all == PAD_INDEX, spk_embed=spk_all)
            encoded_all = F.pad(encoded_all, (0, 0, 1, 0))                         # [1 + S, n_tokens + 1, H]
            cond_all = torch.gather(encoded_all, 1, mel2ph.expand(1 + n_mix, -1, -1))   # [1 + S, n_frames, H]
            # base 权重 = 1 - Σblend。不 clamp（会引入 Clip 节点、与 diffusion 的 clamp_spec 撞边名）——
            #   凸性由 C# 保证(单槽 blend∈[0,1] ⇒ Σ≤1;N 槽 C# 归一)，故 1-Σ≥0，无需 clamp。
            base_w = 1.0 - blend.sum(dim=0, keepdim=True)                          # [1, n_frames]
            w_all = torch.cat([base_w, blend], dim=0)                             # [1 + S, n_frames]
            condition = (w_all[:, :, None] * cond_all).sum(dim=0, keepdim=True)   # [1, n_frames, H]

        if self.use_stretch_embed:
            stretch = torch.round(1000 * self.sr(_mel2ph, durations))
            table = self.stretch_embed(torch.arange(0, 1001, device=stretch.device))
            stretch_embed = torch.index_select(table, 0, stretch.view(-1).long()).view_as(condition)
            condition += stretch_embed
            stretch_embed_rnn_out, _ = self.stretch_embed_rnn(condition)
            condition += stretch_embed_rnn_out

        if self.f0_embed_type == 'discrete':
            pitch = f0_to_coarse(f0)
            pitch_embed = self.pitch_embed(pitch)
        else:
            f0_mel = (1 + f0 / 700).log()
            pitch_embed = self.pitch_embed(f0_mel[:, :, None])
        condition += pitch_embed

        if self.use_variance_embeds:
            variance_embeds = torch.stack([
                self.variance_embeds[v_name](variances[v_name][:, :, None] * self.variance_scaling_factor[v_name])
                for v_name in self.variance_embed_list
            ], dim=-1).sum(-1)
            condition += variance_embeds

        if hparams['use_key_shift_embed']:
            if hasattr(self, 'frozen_key_shift'):
                key_shift_embed = self.key_shift_embed(self.frozen_key_shift[:, None, None] * self.variance_scaling_factor['key_shift'])
            else:
                gender = torch.clip(gender, min=-1., max=1.)
                gender_mask = (gender < 0.).float()
                key_shift = gender * ((1. - gender_mask) * self.shift_max + gender_mask * abs(self.shift_min))
                key_shift_embed = self.key_shift_embed(key_shift[:, :, None] * self.variance_scaling_factor['key_shift'])
            condition += key_shift_embed

        if hparams['use_speed_embed']:
            if velocity is not None:
                velocity = torch.clip(velocity, min=self.speed_min, max=self.speed_max)
                speed_embed = self.speed_embed(velocity[:, :, None] * self.variance_scaling_factor['speed'])
            else:
                speed_embed = self.speed_embed(torch.FloatTensor([1.]).to(condition.device)[:, None, None] * self.variance_scaling_factor['speed'])
            condition += speed_embed

        if hparams['use_spk_id']:
            if hasattr(self, 'frozen_spk_embed'):
                condition += self.frozen_spk_embed
            else:
                condition += spk_embed

        condition = self.forward_retake_embedding(
            condition, _mel2ph, retake=retake, gt_mel=gt_mel
        )
        return condition


class FastSpeech2VarianceONNX(FastSpeech2Variance):
    def __init__(self, vocab_size, cross_lingual_token_idx=None):
        super().__init__(vocab_size=vocab_size)
        self.register_buffer(
            'cross_lingual_token_idx',
            torch.LongTensor(cross_lingual_token_idx),
            persistent=False
        )
        if len(cross_lingual_token_idx) == 0:
            self.use_lang_id = False
        self.lr = LengthRegulator()

    def _blend_txt_embed(self, tokens, tokens_b, blend):
        # P1-a phoneme mix (experiment): per-token convex blend of two phoneme embeddings.
        #   blend in [0, 1], shape [B, n_tokens]; 0 => plain lookup (bit-identical to no-mix export).
        if tokens_b is None or blend is None:
            return self.txt_embed(tokens)
        w = blend.unsqueeze(-1)  # [B, n_tokens, 1]
        return (1.0 - w) * self.txt_embed(tokens) + w * self.txt_embed(tokens_b)

    def forward_encoder_word(self, tokens, word_div, word_dur, languages=None, tokens_b=None, blend=None):
        txt_embed = self._blend_txt_embed(tokens, tokens_b, blend)
        ph2word = self.lr(word_div)
        onset = ph2word > F.pad(ph2word, [1, -1])
        onset_embed = self.onset_embed(onset.long())
        ph_word_dur = torch.gather(F.pad(word_dur, [1, 0]), 1, ph2word)
        word_dur_embed = self.word_dur_embed(ph_word_dur.float()[:, :, None])
        extra_embed = onset_embed + word_dur_embed
        if self.use_lang_id:
            lang_mask = torch.any(
                tokens[..., None] == self.cross_lingual_token_idx[None, None],
                dim=-1
            )
            lang_embed = self.lang_embed(languages * lang_mask)
            extra_embed += lang_embed
        x_masks = tokens == PAD_INDEX
        return self.encoder(txt_embed, extra_embed, x_masks), x_masks

    def forward_encoder_phoneme(self, tokens, ph_dur, languages=None, tokens_b=None, blend=None):
        txt_embed = self._blend_txt_embed(tokens, tokens_b, blend)
        if self.use_variance_scaling:
            ph_dur_embed = self.ph_dur_embed(torch.log(1 + ph_dur.float())[:, :, None])
        else:
            ph_dur_embed = self.ph_dur_embed(ph_dur.float()[:, :, None])
        if self.use_lang_id:
            lang_mask = torch.any(
                tokens[..., None] == self.cross_lingual_token_idx[None, None],
                dim=-1
            )
            lang_embed = self.lang_embed(languages * lang_mask)
            extra_embed = ph_dur_embed + lang_embed
        else:
            extra_embed = ph_dur_embed
        x_masks = tokens == PAD_INDEX
        return self.encoder(txt_embed, extra_embed, x_masks), x_masks

    def forward_dur_predictor(self, encoder_out, x_masks, ph_midi, spk_embed=None):
        midi_embed = self.midi_embed(ph_midi)
        dur_cond = encoder_out + midi_embed
        if hparams['use_spk_id'] and spk_embed is not None:
            dur_cond += spk_embed
        ph_dur = self.dur_predictor(dur_cond, x_masks=x_masks)
        return ph_dur

    def view_as_encoder(self):
        model = copy.deepcopy(self)
        if self.predict_dur:
            del model.dur_predictor
            model.forward = model.forward_encoder_word
        else:
            model.forward = model.forward_encoder_phoneme
        return model

    def view_as_dur_predictor(self):
        model = copy.deepcopy(self)
        del model.encoder
        model.forward = model.forward_dur_predictor
        return model
