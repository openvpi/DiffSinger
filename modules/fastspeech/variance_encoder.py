import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from modules.fastspeech.tts_modules import FastSpeech2Encoder, DurationPredictor
from utils.phoneme_utils import PAD_INDEX


class FastSpeech2Variance(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.predict_dur = config['predict_dur']
        self.linguistic_mode = 'word' if config['predict_dur'] else 'phoneme'
        self.use_lang_id = config['use_lang_id']

        self.txt_embed = Embedding(vocab_size, config['hidden_size'], PAD_INDEX)
        if self.use_lang_id:
            self.lang_embed = Embedding(config['num_lang'] + 1, config['hidden_size'], padding_idx=0)

        if self.predict_dur:
            self.onset_embed = Embedding(2, config['hidden_size'])
            self.word_dur_embed = Linear(1, config['hidden_size'])
        else:
            self.ph_dur_embed = Linear(1, config['hidden_size'])

        self.encoder = FastSpeech2Encoder(
            hidden_size=config['hidden_size'], num_layers=config['enc_layers'],
            ffn_kernel_size=config['enc_ffn_kernel_size'], ffn_act=config['ffn_act'],
            dropout=config['dropout'], num_heads=config['num_heads'],
            use_pos_embed=config['use_pos_embed'], rel_pos=config.get('rel_pos', False), 
            use_rope=config.get('use_rope', False)
        )

        dur_hparams = config['dur_prediction_args']
        if self.predict_dur:
            self.midi_embed = Embedding(128, config['hidden_size'])
            self.dur_predictor = DurationPredictor(
                in_dims=config['hidden_size'],
                n_chans=dur_hparams['hidden_size'],
                n_layers=dur_hparams['num_layers'],
                dropout_rate=dur_hparams['dropout'],
                kernel_size=dur_hparams['kernel_size'],
                offset=dur_hparams['log_offset'],
                dur_loss_type=dur_hparams['loss_type']
            )

    def forward(
            self, txt_tokens, midi, ph2word,
            ph_dur=None, word_dur=None,
            spk_embed=None, languages=None,
            infer=True
    ):
        """
        :param txt_tokens: (train, infer) [B, T_ph]
        :param midi: (train, infer) [B, T_ph]
        :param ph2word: (train, infer) [B, T_ph]
        :param ph_dur: (train, [infer]) [B, T_ph]
        :param word_dur: (infer) [B, T_w]
        :param spk_embed: (train) [B, T_ph, H]
        :param languages (train, infer) [B, T_ph]
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
            extra_embed = onset_embed + word_dur_embed
        else:
            ph_dur_embed = self.ph_dur_embed(ph_dur.float()[:, :, None])
            extra_embed = ph_dur_embed
        if self.use_lang_id:
            lang_embed = self.lang_embed(languages)
            extra_embed += lang_embed
        encoder_out = self.encoder(txt_embed, extra_embed, txt_tokens == 0)

        if self.predict_dur:
            midi_embed = self.midi_embed(midi)  # => [B, T_ph, H]
            dur_cond = encoder_out + midi_embed
            if spk_embed is not None:
                dur_cond += spk_embed
            ph_dur_pred = self.dur_predictor(dur_cond, x_masks=txt_tokens == PAD_INDEX, infer=infer)

            return encoder_out, ph_dur_pred
        else:
            return encoder_out, None


class MelodyEncoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        enc_hparams = config['melody_encoder_args']

        def get_hparam(key):
            return enc_hparams.get(key, config.get(key))

        # MIDI inputs
        hidden_size = get_hparam('hidden_size')
        self.note_midi_embed = Linear(1, hidden_size)
        self.note_dur_embed = Linear(1, hidden_size)

        # ornament inputs
        self.use_glide_embed = config['use_glide_embed']
        self.glide_embed_scale = config['glide_embed_scale']
        if self.use_glide_embed:
            # 0: none, 1: up, 2: down
            self.note_glide_embed = Embedding(len(config['glide_types']) + 1, hidden_size, padding_idx=0)

        self.encoder = FastSpeech2Encoder(
            hidden_size=hidden_size, num_layers=get_hparam('enc_layers'),
            ffn_kernel_size=get_hparam('enc_ffn_kernel_size'), ffn_act=get_hparam('ffn_act'),
            dropout=get_hparam('dropout'), num_heads=get_hparam('num_heads'),
            use_pos_embed=get_hparam('use_pos_embed'), rel_pos=get_hparam('rel_pos'),
            use_rope=get_hparam('use_rope')
        )
        self.out_proj = Linear(hidden_size, config['hidden_size'])

    def forward(self, note_midi, note_rest, note_dur, glide=None):
        """
        :param note_midi: float32 [B, T_n], -1: padding
        :param note_rest: bool [B, T_n]
        :param note_dur: int64 [B, T_n]
        :param glide: int64 [B, T_n]
        :return: [B, T_n, H]
        """
        midi_embed = self.note_midi_embed(note_midi[:, :, None]) * ~note_rest[:, :, None]
        dur_embed = self.note_dur_embed(note_dur.float()[:, :, None])
        ornament_embed = 0
        if self.use_glide_embed:
            ornament_embed += self.note_glide_embed(glide) * self.glide_embed_scale
        encoder_out = self.encoder(
            midi_embed, dur_embed + ornament_embed,
            padding_mask=note_midi < 0
        )
        encoder_out = self.out_proj(encoder_out)
        return encoder_out
