import torch
from torch import nn

from lib.conf.schema import EmbeddingsConfig
from .commons.common_layers import XavierUniformInitLinear as Linear

__all__ = [
    "ParameterEmbeddings",
]


class ParameterEmbeddings(nn.Module):
    def __init__(self, config: EmbeddingsConfig):
        super().__init__()
        self.config = config
        self.pitch_embedding = Linear(1, config.embedding_dim)
        self.variance_embeddings = nn.ModuleDict()
        self.transition_embeddings = nn.ModuleDict()
        if self.config.use_energy_embed:
            self.variance_embeddings['energy'] = Linear(1, config.embedding_dim)
        if self.config.use_breathiness_embed:
            self.variance_embeddings['breathiness'] = Linear(1, config.embedding_dim)
        if self.config.use_voicing_embed:
            self.variance_embeddings['voicing'] = Linear(1, config.embedding_dim)
        if self.config.use_tension_embed:
            self.variance_embeddings['tension'] = Linear(1, config.embedding_dim)
        if self.config.use_key_shift_embed:
            self.transition_embeddings['key_shift'] = Linear(1, config.embedding_dim)
        if self.config.use_speed_embed:
            self.transition_embeddings['speed'] = Linear(1, config.embedding_dim)

    def forward(self, x, f0, **kwargs):
        f0_mel = (1 + f0 / 700).log()
        x = x + self.pitch_embedding(f0_mel[:, :, None])
        variance_embeds = [
            self.variance_embeddings[v_name](kwargs[v_name][:, :, None])
            for v_name in self.variance_embeddings.keys()
        ]
        transition_embeds = [
            self.transition_embeddings[v_name](kwargs[v_name][:, :, None])
            for v_name in self.transition_embeddings.keys()
        ]
        if variance_embeds:
            x = x + torch.stack(variance_embeds, dim=-1).sum(-1)
        if transition_embeds:
            x = x + torch.stack(transition_embeds, dim=-1).sum(-1)

        return x
