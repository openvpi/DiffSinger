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
            languages=None
    ):
        txt_embed = self.txt_embed(tokens)
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
        encoded = self.encoder(txt_embed, extra_embed, tokens == PAD_INDEX)
        encoded = F.pad(encoded, (0, 0, 1, 0))
        condition = torch.gather(encoded, 1, mel2ph)

        if self.use_stretch_embed:
            stretch = self.sr(_mel2ph, durations.float())
            stretch_embed = self.stretch_embed(stretch * 1000)
            stretch_embed = self.stretch_embed_mlp(stretch_embed)
            condition += stretch_embed
            stretch_embed_rnn_out, _ =self.stretch_embed_rnn(condition)
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
                self.variance_embeds[v_name](variances[v_name][:, :, None]) 
                * self.variance_scaling_factor[v_name]
                for v_name in self.variance_embed_list
            ], dim=-1).sum(-1)
            condition += variance_embeds

        if hparams['use_key_shift_embed']:
            if hasattr(self, 'frozen_key_shift'):
                key_shift_embed = self.key_shift_embed(self.frozen_key_shift[:, None, None])
            else:
                gender = torch.clip(gender, min=-1., max=1.)
                gender_mask = (gender < 0.).float()
                key_shift = gender * ((1. - gender_mask) * self.shift_max + gender_mask * abs(self.shift_min))
                key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
            key_shift_embed *= self.variance_scaling_factor['key_shift']
            condition += key_shift_embed

        if hparams['use_speed_embed']:
            if velocity is not None:
                velocity = torch.clip(velocity, min=self.speed_min, max=self.speed_max)
                speed_embed = self.speed_embed(velocity[:, :, None])
            else:
                speed_embed = self.speed_embed(torch.FloatTensor([1.]).to(condition.device)[:, None, None])
            speed_embed *= self.variance_scaling_factor['speed']
            condition += speed_embed

        if hparams['use_spk_id']:
            if hasattr(self, 'frozen_spk_embed'):
                condition += self.frozen_spk_embed
            else:
                condition += spk_embed
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

    def forward_encoder_word(self, tokens, word_div, word_dur, languages=None):
        txt_embed = self.txt_embed(tokens)
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

    def forward_encoder_phoneme(self, tokens, ph_dur, languages=None):
        txt_embed = self.txt_embed(tokens)
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
