from typing import Annotated, Any, Dict, List, Literal, Union

from pydantic import Field, field_validator

from .core import ConfigBaseModel
from .ops import (
    ConfigOperationBase, ConfigOperationContext, split_path,
    ref, this, or_, not_, in_, len_, set_, max_, map_
)


class ConfigurationScope:
    ACOUSTIC = 0x1
    VARIANCE = 0x2


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


class DataSourceConfig(ConfigBaseModel):
    raw_data_dir: str = Field(...)
    speaker: str = Field(...)
    spk_id: int = Field(None, ge=0)
    language: str = Field(..., json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=in_(this(), ref("data.dictionaries")),
            message="Language must be one of the keys in data.dictionaries."
        )
    })
    test_prefixes: List[str] = Field([])


# noinspection PyMethodParameters
class DataConfig(ConfigBaseModel):
    dictionaries: Dict[str, str] = Field(...)
    extra_phonemes: List[str] = Field(...)
    merged_phoneme_groups: List[List[str]] = Field(...)
    sources: List[DataSourceConfig] = Field(..., min_length=1)

    @field_validator("dictionaries")
    def check_dictionaries(cls, v):
        if not v:
            raise ValueError("At least one dictionary is required.")
        return v

    @field_validator("sources")
    def check_sources(cls, v: List[DataSourceConfig]):
        cls._check_and_build_spk_map(v)
        return v

    @property
    def lang_map(self) -> Dict[str, int]:
        lang_map = {lang: i for i, lang in enumerate(sorted(self.dictionaries.keys()), start=1)}
        return lang_map

    @property
    def spk_map(self) -> Dict[str, int]:
        spk_map = DataConfig._check_and_build_spk_map(self.sources)
        return spk_map

    @staticmethod
    def _check_and_build_spk_map(sources: List[DataSourceConfig]):
        spk_map: Dict[str, int] = {}
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
    model_path: str = Field(None)
    f0_min: float = Field(65, ge=0)
    f0_max: float = Field(1100, ge=0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() >= ref("binarizer.extractors.pitch_extraction.f0_min"),
            message="f0_max must be greater than or equal to f0_min."
        )
    })


class HarmonicNoiseSeparationConfig(ConfigBaseModel):
    method: Literal["world", "vr"] = Field("vr")
    model_path: str = Field(None)


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
    vmin: float = Field(-12)
    vmax: float = Field(0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > ref("binarizer.features.spectrogram.vmin"),
            message="vmax must be greater than vmin."
        )
    })


class CurveParameterConfig(ConfigBaseModel):
    used: bool = Field(...)
    smooth_width: float = Field(0.12, gt=0)


class BinarizerFeaturesConfig(ConfigBaseModel):
    audio_sample_rate: int = Field(44100, gt=0)
    hop_size: int = Field(512, gt=0)
    fft_size: int = Field(2048, gt=0)
    win_size: int = Field(2048, gt=0)
    spectrogram: SpectrogramConfig = Field(..., json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    energy: CurveParameterConfig = Field(...)
    breathiness: CurveParameterConfig = Field(...)
    voicing: CurveParameterConfig = Field(...)
    tension: CurveParameterConfig = Field(...)


# noinspection PyMethodParameters
class RandomPitchShiftingConfig(ConfigBaseModel):
    enabled: bool = Field(True)
    range: List[float] = Field([-5.0, 5.0])
    scale: float = Field(0.75, ge=0)

    @field_validator("range")
    def check_range(cls, v):
        if len(v) != 2 or v[0] >= v[1] or v[0] >= 0 or v[1] <= 0:
            raise ValueError("Pitch shifting range must be in the form of (min, max) where min < 0 < max.")


# noinspection PyMethodParameters
class RandomTimeStretchingConfig(ConfigBaseModel):
    enabled: bool = Field(True)
    range: List[float] = Field([0.5, 2.0])
    scale: float = Field(0.75, ge=0)

    @field_validator("range")
    def check_range(cls, v):
        if len(v) != 2 or v[0] >= v[1] or v[0] <= 0 or v[1] <= 0:
            raise ValueError("Time stretching range must be in the form of (min, max) where 0 < min < max.")


class BinarizerAugmentationConfig(ConfigBaseModel):
    random_pitch_shifting: RandomPitchShiftingConfig = Field(...)
    random_time_stretching: RandomTimeStretchingConfig = Field(...)


class BinarizerConfig(ConfigBaseModel):
    binary_data_dir: str = Field(...)
    num_workers: int = Field(0, ge=0)
    batch_size: int = Field(8, gt=0)
    extractors: BinarizerExtractorsConfig = Field(...)
    features: BinarizerFeaturesConfig = Field(...)
    augmentation: BinarizerAugmentationConfig = Field(..., json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })


class LinguisticEncoderConfig(ConfigBaseModel):
    embedding_dim: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("model.encoder.hidden_size")
    })
    arch: Literal["fs2"] = Field("fs2")
    kwargs: Dict[str, Any] = Field({})


class EmbeddingsConfig(ConfigBaseModel):
    embedding_dim: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("model.encoder.hidden_size")
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


class EncoderConfig(ConfigBaseModel):
    use_lang_id: bool = Field(False)
    num_lang: int = Field(..., json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() >= len_(ref("data.dictionaries")),
            message="Number of language embeddings must be greater than or equal to the number of languages."
        )
    })
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
    hidden_size: int = Field(256, gt=0)
    linguistic_encoder: LinguisticEncoderConfig = Field(...)


class AuxDecoderConfig(ConfigBaseModel):
    input_dim: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("model.encoder.hidden_size")
    })
    output_dim: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("binarizer.features.spectrogram.num_bins")
    })
    arch: Literal["convnext"] = Field("convnext")
    kwargs: Dict[str, Any] = Field({})


class DecoderBackboneConfig(ConfigBaseModel):
    input_dim: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("model.encoder.hidden_size")
    })
    output_dim: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("binarizer.features.spectrogram.num_bins")
    })
    arch: Literal["wavenet", "lynxnet"] = Field("wavenet")
    kwargs: Dict[str, Any] = Field(...)


class DecoderConfig(ConfigBaseModel):
    diffusion_type: Literal["reflow"] = Field("reflow")
    time_scale_factor: int = Field(1000, gt=0)
    sampling_algorithm: Literal["euler", "rk2", "rk4", "rk5"] = Field("euler")
    sampling_steps: int = Field(20, gt=0)
    backbone: DecoderBackboneConfig = Field(...)


class ModelConfig(ConfigBaseModel):
    encoder: EncoderConfig = Field(...)
    embeddings: EmbeddingsConfig = Field(..., json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    use_shallow_diffusion: bool = Field(True, json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    aux_decoder: AuxDecoderConfig = Field(..., json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })
    decoder: DecoderConfig = Field(..., json_schema_extra={
        "scope": ConfigurationScope.ACOUSTIC
    })


class BatchSamplerConfig(ConfigBaseModel):
    max_batch_frames: int = Field(50000, gt=0)
    max_batch_size: int = Field(64, gt=0)
    frame_count_grid: int = Field(6, ge=1)


class DataLoaderConfig(ConfigBaseModel):
    num_workers: int = Field(4, ge=0)
    prefetch_factor: int = Field(2, ge=0)


class OptimizerConfig(ConfigBaseModel):
    cls: str = Field("torch.optim.AdamW")
    kwargs: Dict[str, Any] = Field(...)


class LRSchedulerConfig(ConfigBaseModel):
    cls: str = Field("torch.optim.lr_scheduler.StepLR")
    unit: Literal["step", "epoch"] = Field("step")
    kwargs: Dict[str, Any] = Field(...)


class UnitCheckpointConfig(ConfigBaseModel):
    prefix: str = Field("model_ckpt")
    monitor: Literal["unit"] = Field("unit")
    save_top_k: int = Field(2)
    every_n_units: int = Field(...)
    since_m_units: int = Field(0, ge=0)


class MetricCheckpointConfig(ConfigBaseModel):
    prefix: str = Field("model_ckpt")
    monitor: Literal["metric"] = Field("metric")
    expr: str = Field(...)
    save_top_k: int = Field(5)
    mode: Literal["max", "min"] = Field(...)


ModelCheckpointConfig = Annotated[
    Union[UnitCheckpointConfig, MetricCheckpointConfig],
    Field(discriminator="monitor")
]


class TrainerStrategyConfig(ConfigBaseModel):
    name: str = Field("auto")
    kwargs: Dict[str, Any] = Field(...)


class TrainerConfig(ConfigBaseModel):
    unit: Literal["step", "epoch"] = Field("step")
    min_steps: int = Field(0)
    max_steps: int = Field(160000)
    min_epochs: int = Field(0)
    max_epochs: int = Field(100)
    checkpoints: List[ModelCheckpointConfig] = Field(..., min_length=1)
    accelerator: str = Field("auto")
    devices: Union[Literal["auto"], int, List[int]] = Field("auto")
    num_nodes: int = Field(1, ge=1)
    strategy: TrainerStrategyConfig = Field(...)
    precision: str = Field("16-mixed")
    val_every_n_units: int = Field(2000, ge=1)
    log_every_n_steps: int = Field(100)
    num_sanity_val_steps: int = Field(1)
    accumulate_grad_batches: int = Field(1, ge=1)


class TrainingConfig(ConfigBaseModel):
    batch_sampler: BatchSamplerConfig = Field(...)
    data_loader: DataLoaderConfig = Field(...)
    optimizer: OptimizerConfig = Field(...)
    lr_scheduler: LRSchedulerConfig = Field(...)
    trainer: TrainerConfig = Field(...)


class BaseConfig(ConfigBaseModel):
    data: DataConfig = Field(...)
    binarizer: BinarizerConfig = Field(...)
    model: ModelConfig = Field(...)
    training: TrainingConfig = Field(...)

    def _resolve_recursive(self, current: ConfigBaseModel, context: ConfigOperationContext):
        """
        Recursively resolve all dynamic expressions in the config.
        :param current: The current model instance.
        :param context: The context for resolving dynamic expressions.
        """
        for field_name, field_info in type(current).model_fields.items():
            field_scope = current.__field_scopes__.get(field_name)
            if field_scope is not None and not field_scope & context.scope_mask:
                continue
            context.current_path.append(field_name)
            value = getattr(current, field_name)
            if isinstance(value, ConfigBaseModel):
                self._resolve_recursive(value, context)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    context.current_path.append(i)
                    if isinstance(item, ConfigBaseModel):
                        self._resolve_recursive(item, context)
                    context.current_path.pop()
            if field_info.json_schema_extra is not None:
                expr = field_info.json_schema_extra.get('dynamic_expr')
                if expr:
                    if isinstance(expr, ConfigOperationBase):
                        context.current_value = value
                        expr = expr.resolve(context)
                    setattr(current, field_name, expr)
            context.current_path.pop()

    def _check_recursive(self, current: ConfigBaseModel, context: ConfigOperationContext):
        """
        Recursively check all dynamic expressions in the config.
        :param current: The current model instance.
        :param context: The context for checking dynamic expressions.
        """
        for field_name, field_info in type(current).model_fields.items():
            field_scope = current.__field_scopes__.get(field_name)
            if field_scope is not None and not field_scope & context.scope_mask:
                continue
            context.current_path.append(field_name)
            value = getattr(current, field_name)
            if isinstance(value, ConfigBaseModel):
                self._check_recursive(value, context)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    context.current_path.append(i)
                    if isinstance(item, ConfigBaseModel):
                        self._check_recursive(item, context)
                    context.current_path.pop()
            if field_info.json_schema_extra is not None:
                check = field_info.json_schema_extra.get('dynamic_check')
                if check:
                    context.current_value = value
                    check.run(context)
            context.current_path.pop()

    def _process_nested(self, f, scope: int = 0, path: str = None):
        """
        Process the nested config object under a given path.
        :param f: The function to apply to the object.
        :param scope: The scope mask.
        :param path: Path of the object. None points to the root.
        """
        current = self
        if path:
            parts = split_path(path)
            for p in parts:
                if isinstance(current, (tuple, list)):
                    current = current[int(p)]
                elif isinstance(current, dict):
                    current = current[p]
                else:
                    current = getattr(current, p)
        else:
            parts = []
        if not isinstance(current, ConfigBaseModel):
            return
        context = ConfigOperationContext(
            root=self,
            current_path=parts,
            current_value=current,
            scope_mask=scope
        )
        f(current, context)

    def resolve(self, scope_mask: int = 0, from_path: str = None):
        """
        Resolve all dynamic expressions from a given path in the config.
        :param scope_mask: The scope mask to use for dynamic resolving.
        :param from_path: The path to resolve from. If None, resolve from the root.
        """
        self._process_nested(self._resolve_recursive, scope_mask, from_path)

    def check(self, scope_mask: int = 0, from_path: str = None):
        """
        Check all dynamic expressions from a given path in the config.
        :param scope_mask: The scope mask to use for dynamic checking.
        :param from_path: The path to check from. If None, check from the root.
        """
        self._process_nested(self._check_recursive, scope_mask, from_path)
