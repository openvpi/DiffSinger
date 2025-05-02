import json
import math
import pathlib
from typing import Annotated, Any, Literal, Union

from pydantic import Field, field_validator

from lib.vocabulary import PhonemeDictionary
from .core import ConfigBaseModel
from .ops import (
    ConfigOperationBase, ConfigOperationContext,
    ref, this, or_, not_, in_, len_, set_, max_, map_, ctx, if_, exists, coalesce, func
)


class ConfigurationScope:
    ACOUSTIC = 0x1
    VARIANCE = 0x2
    DURATION = 0x4


class DynamicCheck:
    def __init__(self, expr: ConfigOperationBase, message=None):
        self.expr = expr
        self.message = message

    def run(self, context: ConfigOperationContext):
        if isinstance(self.expr, ConfigOperationBase):
            expr = self.expr.resolve(context)
        else:
            expr = self.expr
        if not expr:
            raise ValueError(
                f"Dynamic check failed.\n"
                f"{'.'.join(str(e) for e in context.current_path)}\n"
                f"  {self.message}"
            )


class RequiredOnGivenScope(DynamicCheck):
    def __init__(self, scope_mask: int):
        super().__init__(
            expr=if_(ctx("scope") & scope_mask, exists(this()), True),
            message="Field required."
        )


class DataSourceConfig(ConfigBaseModel):
    raw_data_dir: str = Field(...)
    speaker: str = Field(...)
    spk_id: int | None = Field(None, ge=0)
    language: str = Field(..., json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=in_(this(), ref("data.dictionaries")),
            message="Language must be one of the keys in data.dictionaries."
        )
    })
    test_prefixes: list[str] = Field([])

    @property
    def raw_data_dir_resolved(self) -> pathlib.Path:
        return pathlib.Path(self.raw_data_dir).resolve()


# noinspection PyMethodParameters
class DataConfig(ConfigBaseModel):
    dictionaries: dict[str, str] = Field(...)
    extra_phonemes: list[str] = Field(...)
    merged_phoneme_groups: list[list[str]] = Field(...)
    glide_tags: list[Literal["up", "down"]] = Field(["up", "down"], max_length=2, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    sources: list[DataSourceConfig] = Field(..., min_length=1)

    @field_validator("dictionaries")
    def check_dictionaries(cls, v):
        if not v:
            raise ValueError("At least one dictionary is required.")
        return v

    @field_validator("sources")
    def check_sources(cls, v: list[DataSourceConfig]):
        cls._check_and_build_spk_map(v)
        return v

    @property
    def lang_map(self) -> dict[str, int]:
        lang_map = {lang: i for i, lang in enumerate(sorted(self.dictionaries.keys()), start=1)}
        return lang_map

    # noinspection PyAttributeOutsideInit
    @property
    def spk_map(self) -> dict[str, int]:
        if hasattr(self, "_spk_map"):
            return self._spk_map
        self._spk_map = DataConfig._check_and_build_spk_map(self.sources)
        return self._spk_map

    @property
    def glide_map(self) -> dict[str, int]:
        glide_map = {
            "none": 0,
            **{typename: idx for idx, typename in enumerate(self.glide_tags, start=1)}
        }
        return glide_map

    # noinspection PyAttributeOutsideInit
    @property
    def phoneme_dictionary(self) -> PhonemeDictionary:
        if hasattr(self, "_phoneme_dictionary"):
            return self._phoneme_dictionary
        self._phoneme_dictionary = PhonemeDictionary(
            dictionaries=self.dictionaries,
            extra_phonemes=self.extra_phonemes,
            merged_groups=self.merged_phoneme_groups
        )
        return self._phoneme_dictionary

    @staticmethod
    def _check_and_build_spk_map(sources: list[DataSourceConfig]):
        spk_map: dict[str, int] = {}
        assigned_spk_ids = {s.spk_id for s in sources if s.spk_id is not None}
        next_spk_id = 0
        for source in sources:
            if source.spk_id is None:
                if source.speaker in spk_map:
                    source.spk_id = spk_map[source.speaker]
                else:
                    while next_spk_id in assigned_spk_ids:
                        next_spk_id += 1
                    source.spk_id = next_spk_id
                    spk_map[source.speaker] = next_spk_id
                    assigned_spk_ids.add(next_spk_id)
                    next_spk_id += 1
            elif source.speaker in spk_map and spk_map[source.speaker] != source.spk_id:
                raise ValueError(
                    f"Speaker '{source.speaker}' has conflicting spk_id: "
                    f"{spk_map[source.speaker]} and {source.spk_id}"
                )
            else:
                spk_map[source.speaker] = source.spk_id
        return spk_map


class PitchExtractionConfig(ConfigBaseModel):
    method: Literal["parselmouth", "harvest", "rmvpe"] = Field("rmvpe")
    model_path: str | None = Field(None)
    f0_min: float = Field(65, ge=0)
    f0_max: float = Field(1100, ge=0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() >= ref("binarizer.extractors.pitch_extraction.f0_min"),
            message="f0_max must be greater than or equal to f0_min."
        )
    })


class HarmonicNoiseSeparationConfig(ConfigBaseModel):
    method: Literal["world", "vr"] = Field("vr")
    model_path: str | None = Field(None)


class BinarizerExtractorsConfig(ConfigBaseModel):
    pitch_extraction: PitchExtractionConfig = Field(...)
    harmonic_noise_separation: HarmonicNoiseSeparationConfig = Field(...)


class SpectrogramConfig(ConfigBaseModel):
    type: Literal["mel"] = Field("mel")
    num_bins: int = Field(128, gt=0)
    fmin: float = Field(40, ge=0)
    fmax: float = Field(16000, ge=0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > ref("binarizer.features.spectrogram.fmin"),
            message="fmax must be greater than fmin."
        )
    })


class CurveParameterConfig(ConfigBaseModel):
    enabled: bool = Field(...)
    smooth_width: float = Field(0.12, gt=0)


class BinarizerFeaturesConfig(ConfigBaseModel):
    audio_sample_rate: int = Field(44100, gt=0)
    hop_size: int = Field(512, gt=0)
    fft_size: int = Field(2048, gt=0)
    win_size: int = Field(2048, gt=0)
    spectrogram: SpectrogramConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.ACOUSTIC)
    })
    energy: CurveParameterConfig = Field(...)
    breathiness: CurveParameterConfig = Field(...)
    voicing: CurveParameterConfig = Field(...)
    tension: CurveParameterConfig = Field(...)

    @property
    def enabled_variance_names(self):
        enabled_variance_names = []
        if self.energy.enabled:
            enabled_variance_names.append("energy")
        if self.breathiness.enabled:
            enabled_variance_names.append("breathiness")
        if self.voicing.enabled:
            enabled_variance_names.append("voicing")
        if self.tension.enabled:
            enabled_variance_names.append("tension")
        return enabled_variance_names


class BinarizerMIDIConfig(ConfigBaseModel):
    enabled: bool = Field(...)
    with_glide: bool = Field(False)
    smooth_width: float = Field(0.06, gt=0)


# noinspection PyMethodParameters
class RandomPitchShiftingConfig(ConfigBaseModel):
    enabled: bool = Field(True)
    range: list[float] = Field([-5.0, 5.0])
    scale: float = Field(0.75, ge=0)

    @field_validator("range")
    def check_range(cls, v):
        if len(v) != 2 or v[0] >= 0 or v[1] <= 0:
            raise ValueError("Pitch shifting range must be in the form of (min, max) where min < 0 < max.")
        return v


# noinspection PyMethodParameters
class RandomTimeStretchingConfig(ConfigBaseModel):
    enabled: bool = Field(True)
    range: list[float] = Field([0.5, 2.0])
    scale: float = Field(0.75, ge=0)

    @field_validator("range")
    def check_range(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] >= v[1]:
            raise ValueError("Time stretching range must be in the form of (min, max) where 0 < min < max.")
        return v


class BinarizerAugmentationConfig(ConfigBaseModel):
    random_pitch_shifting: RandomPitchShiftingConfig = Field(...)
    random_time_stretching: RandomTimeStretchingConfig = Field(...)


class BinarizerConfig(ConfigBaseModel):
    binary_data_dir: str = Field(...)
    num_workers: int = Field(0, ge=0)
    prefer_ds: bool = Field(False, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    extractors: BinarizerExtractorsConfig = Field(...)
    features: BinarizerFeaturesConfig = Field(...)
    augmentation: BinarizerAugmentationConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.ACOUSTIC)
    })
    midi: BinarizerMIDIConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.VARIANCE)
    })

    @property
    def binary_data_dir_resolved(self) -> pathlib.Path:
        return pathlib.Path(self.binary_data_dir).resolve()


def _get_vocab_size_from_ph_map(ph_map_json: pathlib.Path):
    if not ph_map_json.exists():
        return None
    with open(ph_map_json, "r", encoding="utf-8") as f:
        ph_map = json.load(f)
    return max(ph_map.values()) + 1


class LinguisticEncoderConfig(ConfigBaseModel):
    vocab_size: int = Field(None, json_schema_extra={
        "dynamic_expr": coalesce(
            this(),
            func(lambda c: _get_vocab_size_from_ph_map(c.binary_data_dir_resolved / "ph_map.json"), ref("binarizer")),
            func(lambda c: c.phoneme_dictionary.vocab_size, ref("data"))
        )
    })
    use_lang_id: bool = Field(False)
    num_lang: int = Field(..., json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() >= len_(ref("data.dictionaries")),
            message="Number of language embeddings must be greater than or equal to the number of languages."
        )
    })
    hidden_size: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("model.condition_dim")
    })
    arch: Literal["fs2"] = Field("fs2")
    kwargs: dict[str, Any] = Field({})


class MelodyEncoderConfig(ConfigBaseModel):
    use_glide_id: bool = Field(False, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=or_(ref("binarizer.midi.with_glide"), not_(this())),
            message="Glide embed is available only if MIDI is with glide."
        )
    })
    num_glide: int = Field(1, json_schema_extra={
        "dynamic_expr": len_(ref("data.glide_tags"))
    })
    glide_embed_scale: float = Field(math.sqrt(128), gt=0)
    hidden_size: int = Field(128, gt=0)
    out_size: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("model.condition_dim")
    })
    arch: Literal["fs2"] = Field("fs2")
    kwargs: dict[str, Any] = Field({})


class EmbeddingsConfig(ConfigBaseModel):
    embedding_dim: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("model.condition_dim")
    })
    use_energy_embed: bool = Field(False, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=or_(ref("binarizer.features.energy.used"), not_(this())),
            message="Energy embedding cannot be enabled if energy is not extracted. "
                    "Enable binarizer.features.energy.used and re-binarize the dataset."
        )
    })
    use_breathiness_embed: bool = Field(False, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=or_(ref("binarizer.features.breathiness.used"), not_(this())),
            message="Breathiness embedding cannot be enabled if breathiness is not extracted. "
                    "Enable binarizer.features.breathiness.used and re-binarize the dataset."
        )
    })
    use_voicing_embed: bool = Field(False, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=or_(ref("binarizer.features.voicing.used"), not_(this())),
            message="Voicing embedding cannot be enabled if voicing is not extracted. "
                    "Enable binarizer.features.voicing.used and re-binarize the dataset."
        )
    })
    use_tension_embed: bool = Field(False, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=or_(ref("binarizer.features.tension.used"), not_(this())),
            message="Tension embedding cannot be enabled if tension is not extracted. "
                    "Enable binarizer.features.tension.used and re-binarize the dataset."
        )
    })
    use_key_shift_embed: bool = Field(False, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() == ref("binarizer.augmentation.random_pitch_shifting.enabled"),
            message="This value must consist with binarizer.augmentation.random_pitch_shifting.enabled."
        )
    })
    use_speed_embed: bool = Field(False, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() == ref("binarizer.augmentation.random_time_stretching.enabled"),
            message="This value must consist with binarizer.augmentation.random_time_stretching.enabled."
        )
    })

    @property
    def embedded_variance_names(self):
        embedded_variance_names = []
        if self.use_energy_embed:
            embedded_variance_names.append("energy")
        if self.use_breathiness_embed:
            embedded_variance_names.append("breathiness")
        if self.use_voicing_embed:
            embedded_variance_names.append("voicing")
        if self.use_tension_embed:
            embedded_variance_names.append("tension")
        return embedded_variance_names


class PredictionConfig(ConfigBaseModel):
    predict_pitch: bool = Field(...)
    predict_energy: bool = Field(...)
    predict_breathiness: bool = Field(...)
    predict_voicing: bool = Field(...)
    predict_tension: bool = Field(...)

    @property
    def predicted_variance_names(self):
        predicted_variance_names = []
        if self.predict_energy:
            predicted_variance_names.append("energy")
        if self.predict_breathiness:
            predicted_variance_names.append("breathiness")
        if self.predict_voicing:
            predicted_variance_names.append("voicing")
        if self.predict_tension:
            predicted_variance_names.append("tension")
        return predicted_variance_names


class NormalizationConfig(ConfigBaseModel):
    spec_min: float = Field(-12.0, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    spec_max: float = Field(0.0, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_check": DynamicCheck(
            expr=this() > ref("model.normalization.spec_min"),
            message="spec_max must be greater than spec_min."
        )
    })
    pitch_repeat_bins: int = Field(64, gt=0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    pitd_norm_min: float = Field(-8.0, lt=0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    pitd_norm_max: float = Field(8.0, gt=0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    pitd_clip_min: float = Field(-12.0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": DynamicCheck(
            expr=this() <= ref("model.normalization.pitd_norm_min"),
            message="pitd_clip_min must be <= pitd_norm_min."
        )
    })
    pitd_clip_max: float = Field(12.0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": DynamicCheck(
            expr=this() >= ref("model.normalization.pitd_norm_max"),
            message="pitd_clip_max must be >= pitd_norm_max."
        )
    })
    variance_total_repeat_bins: int = Field(48, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    energy_db_min: float = Field(-96.0, ge=-96, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    energy_db_max: float = Field(-12.0, le=0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": DynamicCheck(
            expr=this() > ref("model.normalization.energy_db_min"),
            message="energy_db_max must be greater than energy_db_min."
        )
    })
    breathiness_db_min: float = Field(-96.0, ge=-96, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    breathiness_db_max: float = Field(-20.0, le=0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": DynamicCheck(
            expr=this() > ref("model.normalization.breathiness_db_min"),
            message="breathiness_db_max must be greater than breathiness_db_min."
        )
    })
    voicing_db_min: float = Field(-96.0, ge=-96, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    voicing_db_max: float = Field(-12.0, le=0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": DynamicCheck(
            expr=this() > ref("model.normalization.voicing_db_min"),
            message="voicing_db_max must be greater than voicing_db_min."
        )
    })
    tension_logit_min: float = Field(-10.0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE
    })
    tension_logit_max: float = Field(10.0, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": DynamicCheck(
            expr=this() > ref("model.normalization.tension_logit_min"),
            message="tension_logit_max must be greater than tension_logit_min."
        )
    })


class DiffusionDecoderConfig(ConfigBaseModel):
    use_shallow_diffusion: bool = Field(False, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=if_(not_(ctx("scope") == ConfigurationScope.ACOUSTIC), not_(this()), True),
            message="Shallow diffusion is only available for acoustic model."
        )
    })
    aux_decoder_grad: float = Field(0.1, ge=0, lt=1, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    aux_decoder_arch: Literal["convnext"] = Field("convnext", json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    aux_decoder_kwargs: dict[str, Any] = Field({}, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    diffusion_type: Literal["reflow"] = Field("reflow")
    t_start: float = Field(0., ge=0., le=1.)
    time_scale_factor: int = Field(1000, gt=0)
    sampling_algorithm: Literal["euler", "rk2", "rk4", "rk5"] = Field("euler")
    sampling_steps: int = Field(20, gt=0)
    backbone_arch: Literal["wavenet", "lynxnet"] = Field("wavenet")
    backbone_kwargs: dict[str, Any] = Field(...)


class ModelConfig(ConfigBaseModel):
    use_spk_id: bool = Field(False, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=or_(len_(set_(map_(ref("data.sources"), lambda x: x.spk_id))) <= 1, this()),
            message="Speaker embedding must be enabled if there are multiple speaker IDs in the dataset."
        )
    })
    num_spk: int = Field(..., json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > max_(map_(ref("data.sources"), lambda x: x.spk_id)),
            message="Number of speaker embeddings must be greater the maximum speaker ID."
        )
    })
    condition_dim: int = Field(256, gt=0)
    linguistic_encoder: LinguisticEncoderConfig = Field(...)
    melody_encoder: MelodyEncoderConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.VARIANCE)
    })
    embeddings: EmbeddingsConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.ACOUSTIC)
    })
    sample_dim: int = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_expr": ref("binarizer.features.spectrogram.num_bins")
    })
    prediction: PredictionConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.VARIANCE)
    })
    normalization: NormalizationConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC | ConfigurationScope.VARIANCE,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.ACOUSTIC | ConfigurationScope.VARIANCE)
    })
    spec_decoder: DiffusionDecoderConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.ACOUSTIC)
    })
    pitch_predictor: DiffusionDecoderConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.VARIANCE)
    })
    variance_predictor: DiffusionDecoderConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.VARIANCE)
    })


class DiffusionLossConfig(ConfigBaseModel):
    main_loss_type: Literal["L1", "L2"] = Field(...)
    main_loss_log_norm: bool = Field(...)
    aux_loss_type: Literal["L1", "L2"] = Field("L1", json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
    })
    aux_loss_lambda: float = Field(0.1, gt=0, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
    })


class LossConfig(ConfigBaseModel):
    spec_decoder: DiffusionLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.ACOUSTIC)
    })
    pitch_predictor: DiffusionLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.VARIANCE)
    })
    variance_predictor: DiffusionLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.VARIANCE)
    })


class DataLoaderConfig(ConfigBaseModel):
    max_batch_frames: int = Field(50000, gt=0)
    max_batch_size: int = Field(64, gt=0)
    max_val_batch_frames: int = Field(20000, gt=0)
    max_val_batch_size: int = Field(1, gt=0)
    frame_count_grid: int = Field(6, ge=1)
    num_workers: int = Field(4, ge=0)
    prefetch_factor: int = Field(2, ge=0)


class OptimizerConfig(ConfigBaseModel):
    cls: str = Field(...)
    wraps: Literal["parameters", "module"] = Field("parameters")
    kwargs: dict[str, Any] = Field(...)


class LRSchedulerConfig(ConfigBaseModel):
    cls: str = Field(...)
    kwargs: dict[str, Any] = Field(...)
    unit: Literal["step", "epoch"] = Field(...)
    monitor: str | None = Field(None)

    # noinspection PyMethodParameters
    @field_validator("kwargs")
    def check_kwargs(cls, v):
        res = {}
        for key, value in v.items():
            if isinstance(value, dict):
                if "cls" in value:
                    value.setdefault("kwargs", {})
                    value = LRSchedulerConfig.model_validate(value)
                else:
                    value = LRSchedulerConfig.check_kwargs(value)
            elif isinstance(value, list):
                value = [
                    LRSchedulerConfig.model_validate(item) if isinstance(item, dict) and "cls" in item else item
                    for item in value
                ]
            res[key] = value
        return res


class PeriodicCheckpointConfig(ConfigBaseModel):
    tag: str = Field(...)
    type: Literal["periodic"] = Field("periodic")
    unit: Literal["step", "epoch"] = Field(None, json_schema_extra={
        "dynamic_expr": coalesce(this(), ref("training.trainer.unit"))
    })
    since_m_units: int = Field(0, ge=0)
    every_n_units: int = Field(...)
    save_last_k: int = Field(1, ge=-1)
    weights_only: bool = Field(False)


class ExpressionCheckpointConfig(ConfigBaseModel):
    tag: str = Field(...)
    type: Literal["expression"] = Field("expression")
    expression: str = Field(...)
    save_top_k: int = Field(5, ge=-1)
    mode: Literal["max", "min"] = Field(...)
    weights_only: bool = Field(False)


ModelCheckpointConfig = Annotated[
    PeriodicCheckpointConfig | ExpressionCheckpointConfig,
    Field(discriminator="type")
]


class TrainerStrategyConfig(ConfigBaseModel):
    name: str = Field("auto")
    kwargs: dict[str, Any] = Field(...)


class TrainerConfig(ConfigBaseModel):
    unit: Literal["step", "epoch"] = Field(...)
    min_steps: int = Field(0)
    max_steps: int = Field(160000)
    min_epochs: int = Field(0)
    max_epochs: int = Field(1000)
    val_every_n_units: int = Field(..., ge=1)
    log_every_n_steps: int = Field(100, ge=1)
    num_sanity_val_steps: int = Field(1)
    checkpoints: list[ModelCheckpointConfig] = Field(..., min_length=1)
    accelerator: str = Field("auto")
    devices: Union[Literal["auto"], int, list[int]] = Field("auto")
    num_nodes: Literal[1] = Field(1, ge=1)
    strategy: TrainerStrategyConfig = Field(...)
    precision: str = Field("16-mixed")
    accumulate_grad_batches: int = Field(1, ge=1)
    gradient_clip_val: float = Field(1.0, gt=0)

    # noinspection PyMethodParameters
    @field_validator("checkpoints")
    def check_checkpoints(cls, v):
        tags = set()
        for checkpoint in v:
            if checkpoint.tag in tags:
                raise ValueError(f"Duplicate checkpoint tag: '{checkpoint.tag}'.")
            tags.add(checkpoint.tag)
        if all(c.weights_only for c in v):
            raise ValueError("At least one checkpoint should set weights_only to False.")
        return v


class VocoderConfig(ConfigBaseModel):
    vocoder_type: Literal["nsf-hifigan"] = Field("nsf-hifigan", json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    vocoder_path: str = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    audio_sample_rate: int = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_expr": ref("binarizer.features.audio_sample_rate")
    })
    hop_size: int = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_expr": ref("binarizer.features.hop_size")
    })
    fft_size: int = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_expr": ref("binarizer.features.fft_size")
    })
    win_size: int = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_expr": ref("binarizer.features.win_size")
    })
    spectrogram: SpectrogramConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_expr": ref("binarizer.features.spectrogram")
    })


class ValidationConfig(ConfigBaseModel):
    use_vocoder: bool = Field(True, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
    })
    spec_vmin: float = Field(-14.0, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
    })
    spec_vmax: float = Field(4.0, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_check": DynamicCheck(
            expr=this() > ref("training.validation.spec_vmin"),
            message="spec_vmax must be greater than spec_vmin."
        )
    })
    max_plots: int = Field(10, ge=0)
    vocoder: VocoderConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_expr": ref("inference.vocoder"),
    })


class FinetuningConfig(ConfigBaseModel):
    pretraining_enabled: bool = Field(False)
    pretraining_from: str | None = Field(None, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=if_(ref("training.finetuning.pretraining_enabled"), exists(this()), True),
            message="pretraining_from must be specified if pretraining_enabled is True."
        )
    })
    pretraining_include_params: list[str] = Field(["model.*"])
    pretraining_exclude_params: list[str] = Field([])
    freezing_enabled: bool = Field(False)
    frozen_params: list[str] = Field([])


class WeightAveragingConfig(ConfigBaseModel):
    ema_enabled: bool = Field(False)
    ema_decay: float = Field(0.999, gt=0, le=1)
    ema_include_params: list[str] = Field(["model.*"])
    ema_exclude_params: list[str] = Field([])


class TrainingConfig(ConfigBaseModel):
    loss: LossConfig = Field(...)
    dataloader: DataLoaderConfig = Field(...)
    optimizer: OptimizerConfig = Field(...)
    lr_scheduler: LRSchedulerConfig = Field(...)
    trainer: TrainerConfig = Field(...)
    validation: ValidationConfig = Field(...)
    finetuning: FinetuningConfig = Field(...)
    weight_averaging: WeightAveragingConfig = Field(...)


class InferenceConfig(ConfigBaseModel):
    key_shift_range: list[float] = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_expr": ref("binarizer.augmentation.random_pitch_shifting.range")
    })
    speed_range: list[float] = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_expr": ref("binarizer.augmentation.random_time_stretching.range")
    })
    vocoder: VocoderConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.ACOUSTIC)
    })
    timestep: float = Field(None, json_schema_extra={
        "scope": ConfigurationScope.VARIANCE,
        "dynamic_expr": ref("binarizer.features.hop_size") / ref("binarizer.features.audio_sample_rate")
    })


class RootConfig(ConfigBaseModel):
    data: DataConfig = Field(...)
    binarizer: BinarizerConfig = Field(...)
    model: ModelConfig = Field(...)
    training: TrainingConfig = Field(...)
    inference: InferenceConfig = Field(...)
