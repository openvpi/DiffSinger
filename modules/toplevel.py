import torch.nn as nn

from lib.conf.schema import ModelConfig
from .decoder import DiffusionDecoder, ShallowDiffusionOutput
from .embedding import ParameterEmbeddings
from .encoder import LinguisticEncoder, MelodyEncoder
from .fastspeech.tts_modules import LocalUpsample
from .normalizer import FeatureNormalizer

__all__ = [
    "DiffSingerAcoustic",
    "DiffSingerVariance",
]


class DiffSingerAcoustic(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.linguistic_encoder = LinguisticEncoder(
            vocab_size=vocab_size,
            config=config.linguistic_encoder
        )
        self.local_upsample = LocalUpsample()  # tokens to frames
        self.use_spk_embed = config.use_spk_id
        if self.use_spk_embed:
            self.spk_embed = nn.Embedding(config.num_spk, config.condition_dim)
        self.parameter_embeddings = ParameterEmbeddings(config=config.embeddings)
        self.spec_decoder = DiffusionDecoder(
            sample_dim=config.sample_dim,
            condition_dim=config.condition_dim,
            normalizer=FeatureNormalizer(
                num_channels=config.sample_dim, num_features=1, num_repeats=None,
                squeezed_channel_dim=False,
                norm_mins=[config.normalization.spec_min],
                norm_maxs=[config.normalization.spec_max],
            ),
            config=config.spec_decoder
        )

    def forward(
            self, tokens, durations, languages, f0, spk_ids=None,
            spk_embed=None, gt_mel=None, infer=True, **kwargs
    ) -> ShallowDiffusionOutput:
        encoder_out = self.linguistic_encoder(tokens=tokens, durations=durations, languages=languages)
        cond = self.local_upsample(encoder_out, ups=durations)
        if self.use_spk_embed:
            if spk_embed is None:
                spk_embed = self.spk_embed(spk_ids)[:, None, :]
            cond = cond + spk_embed
        cond = self.parameter_embeddings(cond, f0=f0, **kwargs)
        decoder_out = self.spec_decoder(condition=cond, sample_gt=gt_mel, infer=infer)
        return decoder_out


class DiffSingerVariance(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.predict_pitch = config.prediction.predict_pitch
        variance_list = []
        var_norm_mins = []
        var_norm_maxs = []
        var_clip_mins = []
        var_clip_maxs = []
        if config.prediction.predict_energy:
            variance_list.append("energy")
            var_norm_mins.append(config.normalization.energy_db_min)
            var_norm_maxs.append(config.normalization.energy_db_max)
            var_clip_mins.append(config.normalization.energy_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_breathiness:
            variance_list.append("breathiness")
            var_norm_mins.append(config.normalization.breathiness_db_min)
            var_norm_maxs.append(config.normalization.breathiness_db_max)
            var_clip_mins.append(config.normalization.breathiness_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_voicing:
            variance_list.append("voicing")
            var_norm_mins.append(config.normalization.voicing_db_min)
            var_norm_maxs.append(config.normalization.voicing_db_max)
            var_clip_mins.append(config.normalization.voicing_db_min)
            var_clip_maxs.append(0.)
        if config.prediction.predict_tension:
            variance_list.append("tension")
            var_norm_mins.append(config.normalization.tension_logit_min)
            var_norm_maxs.append(config.normalization.tension_logit_max)
            var_clip_mins.append(config.normalization.tension_logit_min)
            var_clip_maxs.append(config.normalization.tension_logit_max)
        self.predict_variances = len(variance_list) > 0
        self.variance_list = variance_list
        if not self.predict_pitch and not self.predict_variances:
            raise ValueError("Nothing to predict.")

        self.linguistic_encoder = LinguisticEncoder(
            vocab_size=vocab_size,
            config=config.linguistic_encoder
        )
        self.local_upsample = LocalUpsample()
        self.use_spk_embed = config.use_spk_id
        if self.use_spk_embed:
            self.spk_embed = nn.Embedding(config.num_spk, config.condition_dim)
        if self.predict_pitch:
            self.melody_encoder = MelodyEncoder(config=config.melody_encoder)
            self.pitch_predictor = DiffusionDecoder(
                sample_dim=config.normalization.pitch_repeat_bins,
                condition_dim=config.condition_dim,
                normalizer=FeatureNormalizer(
                    num_channels=1, num_features=1, num_repeats=config.normalization.pitch_repeat_bins,
                    squeezed_channel_dim=True,
                    norm_mins=[config.normalization.pitd_norm_min],
                    norm_maxs=[config.normalization.pitd_norm_max],
                    clip_mins=[config.normalization.pitd_clip_min],
                    clip_maxs=[config.normalization.pitd_clip_max],
                ),
                config=config.pitch_predictor
            )
        if self.predict_variances:
            total_repeat_bins = config.normalization.variance_total_repeat_bins
            if total_repeat_bins % len(self.variance_list) != 0:
                raise ValueError(
                    f"variance_total_repeat_bins must be divisible by "
                    f"number of variances ({len(self.variance_list)})."
                )
            self.pitch_embedding = nn.Linear(1, config.condition_dim)
            self.variance_predictor = DiffusionDecoder(
                sample_dim=total_repeat_bins,
                condition_dim=config.condition_dim,
                normalizer=FeatureNormalizer(
                    num_channels=1, num_features=len(self.variance_list),
                    num_repeats=total_repeat_bins // len(self.variance_list),
                    squeezed_channel_dim=True,
                    norm_mins=var_norm_mins, norm_maxs=var_norm_maxs,
                    clip_mins=var_clip_mins, clip_maxs=var_clip_maxs,
                ),
                config=config.variance_predictor
            )

    def forward(
            self, tokens, durations, languages, spk_ids=None, spk_embed=None,
            note_midi=None, note_rest=None, note_dur=None, note_glide=None, pitch=None,
            infer=True, **kwargs
    ):
        linguistic_encoder_out = self.linguistic_encoder(tokens=tokens, durations=durations, languages=languages)
        cond = self.local_upsample(linguistic_encoder_out, ups=durations)
        if self.use_spk_embed:
            if spk_embed is None:
                spk_embed = self.spk_embed(spk_ids)[:, None, :]
            cond = cond + spk_embed
        if self.predict_pitch:
            melody_encoder_out = self.melody_encoder(
                note_midi=note_midi, note_rest=note_rest, note_dur=note_dur, glide=note_glide
            )
            # TODO: add pitch retaking and expressiveness
            pitch_cond = cond + self.local_upsample(melody_encoder_out, ups=note_dur)
            pitch_predictor_out = self.pitch_predictor(condition=pitch_cond, sample_gt=pitch, infer=infer)
            pitch_predictor_out = pitch_predictor_out.diff_out  # no shallow diffusion yet
        else:
            pitch_predictor_out = None
        if self.predict_variances:
            variance_cond = cond + self.pitch_embedding(pitch[:, :, None])
            variance_predictor_out = self.variance_predictor(
                condition=variance_cond,
                sample_gt=[kwargs.get(v_name) for v_name in self.variance_list],
                infer=infer
            )
            variance_predictor_out = variance_predictor_out.diff_out  # no shallow diffusion yet
        else:
            variance_predictor_out = None
        return pitch_predictor_out, variance_predictor_out
