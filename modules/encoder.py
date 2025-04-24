import torch
from torch import nn

from lib.conf.schema import LinguisticEncoderConfig, MelodyEncoderConfig
from utils import filter_kwargs
from .commons.common_layers import (
    NormalInitEmbedding as Embedding,
    XavierUniformInitLinear as Linear,
)
from modules.commons.tts_modules import FastSpeech2Encoder

__all__ = [
    "LinguisticEncoder",
    "MelodyEncoder",
]

ENCODERS = {
    "fs2": FastSpeech2Encoder,
}


class LinguisticEncoder(nn.Module):
    def __init__(self, vocab_size, config: LinguisticEncoderConfig):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, config.hidden_size, padding_idx=0)
        self.use_lang_id = config.use_lang_id
        if self.use_lang_id:
            self.language_embedding = Embedding(config.num_lang + 1, config.hidden_size, padding_idx=0)
        self.duration_embedding = Linear(1, config.hidden_size)
        self.encoder = (cls := ENCODERS[config.arch])(
            hidden_size=config.hidden_size, **filter_kwargs(config.kwargs, cls)
        )

    def forward(self, tokens, durations, languages=None):
        txt_embed = self.token_embedding(tokens)
        dur_embed = self.duration_embedding(durations[:, :, None].float())
        if self.use_lang_id:
            lang_embed = self.language_embedding(languages)
            extra_embed = dur_embed + lang_embed
        else:
            extra_embed = dur_embed
        encoder_out = self.encoder(txt_embed, extra_embed, tokens == 0)

        return encoder_out


class MelodyEncoder(nn.Module):
    def __init__(self, config: MelodyEncoderConfig):
        super().__init__()
        # MIDI inputs
        hidden_size = config.hidden_size
        self.midi_embedding = Linear(1, hidden_size)
        self.duration_embedding = Linear(1, hidden_size)

        # ornament inputs
        self.use_glide_embed = config.use_glide_id
        self.glide_embed_scale = config.glide_embed_scale
        if self.use_glide_embed:
            # 0: none, 1: up, 2: down
            self.glide_embedding = Embedding(config.num_glide + 1, hidden_size, padding_idx=0)

        self.encoder = ENCODERS[config.arch](
            hidden_size=config.hidden_size, **config.kwargs
        )
        self.out_proj = Linear(hidden_size, config.out_size)

    def forward(self, note_midi, note_rest, note_dur, glide=None):
        """
        :param note_midi: float32 [B, T_n], -1: padding
        :param note_rest: bool [B, T_n]
        :param note_dur: int64 [B, T_n]
        :param glide: int64 [B, T_n]
        :return: [B, T_n, H]
        """
        midi_embed = self.midi_embedding(note_midi[:, :, None]) * ~note_rest[:, :, None]
        dur_embed = self.duration_embedding(note_dur.float()[:, :, None])
        ornament_embed = 0
        ornament_embeds = []
        if self.use_glide_embed:
            ornament_embeds.append(self.glide_embedding(glide) * self.glide_embed_scale)
        if ornament_embeds:
            ornament_embed = torch.stack(ornament_embeds, dim=-1).sum(dim=-1)
        encoder_out = self.encoder(
            midi_embed, dur_embed + ornament_embed,
            padding_mask=note_rest
        )
        encoder_out = self.out_proj(encoder_out)
        return encoder_out
