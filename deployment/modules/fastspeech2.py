import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from modules.fastspeech.variance_encoder import FastSpeech2Variance
from utils.hparams import hparams
from utils.pitch_utils import (
    f0_bin, f0_mel_min, f0_mel_max
)
from utils.text_encoder import PAD_INDEX


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
    def __init__(self, vocab_size):
        super().__init__(vocab_size=vocab_size)
        self.lr = LengthRegulator()
        if hparams.get('use_key_shift_embed', False):
            self.shift_min, self.shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
        if hparams.get('use_speed_embed', False):
            self.speed_min, self.speed_max = hparams['augmentation_args']['random_time_stretching']['range']

    # noinspection PyMethodOverriding
    def forward(self, tokens, durations, f0, variances: dict, gender=None, velocity=None, spk_embed=None):
        txt_embed = self.txt_embed(tokens)
        durations = durations * (tokens > 0)
        mel2ph = self.lr(durations)
        f0 = f0 * (mel2ph > 0)
        mel2ph = mel2ph[..., None].repeat((1, 1, hparams['hidden_size']))
        dur_embed = self.dur_embed(durations.float()[:, :, None])
        encoded = self.encoder(txt_embed, dur_embed, tokens == PAD_INDEX)
        encoded = F.pad(encoded, (0, 0, 1, 0))
        condition = torch.gather(encoded, 1, mel2ph)

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
                for v_name in self.variance_embed_list
            ], dim=-1).sum(-1)
            condition += variance_embeds

        if hparams.get('use_key_shift_embed', False):
            if hasattr(self, 'frozen_key_shift'):
                key_shift_embed = self.key_shift_embed(self.frozen_key_shift[:, None, None])
            else:
                gender = torch.clip(gender, min=-1., max=1.)
                gender_mask = (gender < 0.).float()
                key_shift = gender * ((1. - gender_mask) * self.shift_max + gender_mask * abs(self.shift_min))
                key_shift_embed = self.key_shift_embed(key_shift[:, :, None])
            condition += key_shift_embed

        if hparams.get('use_speed_embed', False):
            if velocity is not None:
                velocity = torch.clip(velocity, min=self.speed_min, max=self.speed_max)
                speed_embed = self.speed_embed(velocity[:, :, None])
            else:
                speed_embed = self.speed_embed(torch.FloatTensor([1.]).to(condition.device)[:, None, None])
            condition += speed_embed

        if hparams['use_spk_id']:
            if hasattr(self, 'frozen_spk_embed'):
                condition += self.frozen_spk_embed
            else:
                condition += spk_embed
        return condition


class FastSpeech2VarianceONNX(FastSpeech2Variance):
    def __init__(self, vocab_size):
        super().__init__(vocab_size=vocab_size)
        self.lr = LengthRegulator()

    def forward_encoder_word(self, tokens, word_div, word_dur):
        txt_embed = self.txt_embed(tokens)
        ph2word = self.lr(word_div)
        onset = ph2word > F.pad(ph2word, [1, -1])
        onset_embed = self.onset_embed(onset.long())
        ph_word_dur = torch.gather(F.pad(word_dur, [1, 0]), 1, ph2word)
        word_dur_embed = self.word_dur_embed(ph_word_dur.float()[:, :, None])
        x_masks = tokens == PAD_INDEX
        return self.encoder(txt_embed, onset_embed + word_dur_embed, x_masks), x_masks

    def forward_encoder_phoneme(self, tokens, ph_dur):
        txt_embed = self.txt_embed(tokens)
        ph_dur_embed = self.ph_dur_embed(ph_dur.float()[:, :, None])
        x_masks = tokens == PAD_INDEX
        return self.encoder(txt_embed, ph_dur_embed, x_masks), x_masks

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
