import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from modules.fastspeech.tts_modules import FastSpeech2Encoder, DurationPredictor
from utils.hparams import hparams
from utils.text_encoder import PAD_INDEX


class FastSpeech2Variance(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.predict_dur = hparams['predict_dur']
        self.linguistic_mode = 'word' if hparams['predict_dur'] else 'phoneme'

        self.txt_embed = Embedding(vocab_size, hparams['hidden_size'], PAD_INDEX)

        if self.predict_dur:
            self.onset_embed = Embedding(2, hparams['hidden_size'])
            self.word_dur_embed = Linear(1, hparams['hidden_size'])
        else:
            self.ph_dur_embed = Linear(1, hparams['hidden_size'])

        self.encoder = FastSpeech2Encoder(
            self.txt_embed, hidden_size=hparams['hidden_size'], num_layers=hparams['enc_layers'],
            ffn_kernel_size=hparams['enc_ffn_kernel_size'],
            ffn_padding=hparams['ffn_padding'], ffn_act=hparams['ffn_act'],
            dropout=hparams['dropout'], num_heads=hparams['num_heads'],
            use_pos_embed=hparams['use_pos_embed'], rel_pos=hparams['rel_pos']
        )

        dur_hparams = hparams['dur_prediction_args']
        if self.predict_dur:
            self.midi_embed = Embedding(128, hparams['hidden_size'])
            self.dur_predictor = DurationPredictor(
                in_dims=hparams['hidden_size'],
                n_chans=dur_hparams['hidden_size'],
                n_layers=dur_hparams['num_layers'],
                dropout_rate=dur_hparams['dropout'],
                padding=hparams['ffn_padding'],
                kernel_size=dur_hparams['kernel_size'],
                offset=dur_hparams['log_offset'],
                dur_loss_type=dur_hparams['loss_type']
            )

    def forward(self, txt_tokens, midi, ph2word, ph_dur=None, word_dur=None, spk_embed=None, infer=True):
        """
        :param txt_tokens: (train, infer) [B, T_ph]
        :param midi: (train, infer) [B, T_ph]
        :param ph2word: (train, infer) [B, T_ph]
        :param ph_dur: (train, [infer]) [B, T_ph]
        :param word_dur: (infer) [B, T_w]
        :param spk_embed: (train) [B, T_ph, H]
        :param infer: whether inference
        :return: encoder_out, ph_dur_pred
        """
        txt_embed = self.txt_embed(txt_tokens)
        if self.linguistic_mode == 'word':
            b = txt_tokens.shape[0]
            onset = torch.diff(ph2word, dim=1, prepend=ph2word.new_zeros(b, 1)) > 0
            onset_embed = self.onset_embed(onset.long())  # [B, T_ph, H]

            if word_dur is None or not infer:
                word_dur = ph_dur.new_zeros(b, ph2word.max() + 1).scatter_add(
                    1, ph2word, ph_dur
                )[:, 1:]  # [B, T_ph] => [B, T_w]
            word_dur = torch.gather(F.pad(word_dur, [1, 0], value=0), 1, ph2word)  # [B, T_w] => [B, T_ph]
            word_dur_embed = self.word_dur_embed(word_dur.float()[:, :, None])

            encoder_out = self.encoder(txt_embed, onset_embed + word_dur_embed, txt_tokens == 0)
        else:
            ph_dur_embed = self.ph_dur_embed(ph_dur.float()[:, :, None])
            encoder_out = self.encoder(txt_embed, ph_dur_embed, txt_tokens == 0)

        if self.predict_dur:
            midi_embed = self.midi_embed(midi)  # => [B, T_ph, H]
            dur_cond = encoder_out + midi_embed
            if spk_embed is not None:
                dur_cond += spk_embed
            ph_dur_pred = self.dur_predictor(dur_cond, x_masks=txt_tokens == PAD_INDEX, infer=infer)

            return encoder_out, ph_dur_pred
        else:
            return encoder_out, None
