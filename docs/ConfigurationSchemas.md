# Configuration Schemas

## The configuration system

DiffSinger uses a cascading configuration system based on YAML files. Inheritance is completely explicit: a configuration file inherits from other files by listing them in its `base_config` attribute. Sources are applied in the following order, with later sources overriding earlier ones:

1. **The `base_config` chain** (from `--config`): base files are loaded depth-first, and configurations are merged recursively: when the overriding value is a mapping and the key already exists in the inherited configuration, it is merged key by key into the existing mapping instead of replacing the whole mapping; non-mapping values (scalars, lists, etc.) simply replace whatever was there before. Keys that exist in only one configuration are kept. All configurations in the inheritance chain are squashed as the final configuration of this source.
2. **The saved experiment configuration**: when `--exp_name` is given, the final configuration is saved to `checkpoints/<exp_name>/config.yaml` (with `base_config` emptied), which is detached from the chain and independent of other configuration files. When the same `--exp_name` is used again (e.g., when resuming training), every key present in the saved file replaces the chain's value wholesale, including nested mappings, while keys that exist only in the chain are kept. Pass `--reset` to discard the saved configuration and rebuild it from `--config` (the rebuilt configuration is then saved again).
3. **Command-line overrides** (from `--hparams key=value,key=value`): applied last, taking precedence over both sources above. The argument string is split on `,` and then on `=`, so values must not contain either character. The override syntax addresses top-level keys only (it does not interpret dotted paths). For an existing key, conversion is only reliable for scalar values whose current type is `bool`, `int`, `float` or `str`; list/mapping values and `None` are not reliably convertible. Boolean overrides currently require Python's `True`/`False` spellings, and the parser uses `eval()` for overrides, so do not use it with untrusted input.

The final configuration is saved to the experiment directory at startup only when `checkpoints/<exp_name>/config.yaml` does not exist yet or `--reset` is given; resuming an existing experiment does *not* re-save it. The saving step is skipped when `--infer` is given, which also marks the run as inference (`hparams['infer']` set to `true`). Only the main process performs the saving.

## Configurable parameters

The following are the meanings and usages of all editable keys in a configuration file.

Each configuration key (including nested keys) is described with a brief explanation and several attributes listed as follows:

| Attribute | Explanation |
| :-: | :-- |
| visibility | Represents which kinds of models and tasks this configuration applies to. Possible values are:<br>**acoustic** - This configuration applies to the acoustic model and task.<br>**variance** - This configuration applies to the variance model and task. |
| scope | The scope of the configuration's effects, indicating what it can influence within the whole pipeline. Possible values are:<br>**nn** - This configuration determines the presence or shapes of parameters and persistent buffers of the neural networks. Modifying it will result in failure when loading or resuming from checkpoints. Configurations that are read at model construction but do **not** change any saved key or shape are not **nn**.<br>**preprocessing** - This configuration controls how raw data pieces or inference inputs are converted to inputs of neural networks. Binarizers should be re-run if this configuration is modified.<br>**training** - This configuration describes the training procedures. Most training configurations can affect training performance, memory consumption, device utilization and loss calculation. Modifying training-only configurations will not cause severe inconsistency or errors in most situations.<br>**inference** - This configuration describes the calculation logic through the model graph. Changing it can lead to inconsistent or wrong outputs of inference or validation. |
| customizability | The level of customizability of the configuration. Possible values are:<br>**required** - This configuration **must** be set or modified according to the actual situation or condition, otherwise errors can be raised.<br>**recommended** - It is recommended to adjust this configuration according to the dataset, requirements, environment and hardware. Most functionality-related and feature-related configurations are at this level, and all configurations at this level are widely tested with different values. However, leaving it unchanged will not cause problems.<br>**normal** - There is no need to modify it as the default value is carefully tuned and widely validated. However, one can still use another value if there are some special requirements or situations.<br>**not recommended** - No values other than the default one are tested for this configuration. Modifying it will not cause errors, but may cause unpredictable or significant impacts on the pipelines.<br>**reserved** - This configuration **must not** be modified. It appears in the configuration file only for future scalability, and currently changing it will result in errors. |
| type | Value type of the configuration. Follows the syntax of Python type hints. Optional omission and fallback behavior are stated in the field description, while explicit `null` is included in the type only when it is accepted. |
| default | Default value of the configuration. Uses YAML value syntax. |
| constraints | Value constraints of the configuration. |

### accumulate_grad_batches

Indicates how many training steps' gradients are accumulated before each `optimizer.step()` call. 1 means no gradient accumulation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### audio_num_mel_bins

Number of mel channels for the mel-spectrogram.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>128</td>
</tbody></table>

### audio_sample_rate

Sampling rate of waveforms.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>44100</td>
</tbody></table>

### augmentation_args

Arguments for data augmentation.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### augmentation_args.fixed_pitch_shifting

Arguments for fixed pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### augmentation_args.fixed_pitch_shifting.enabled

Whether to apply fixed pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
<tr><td align="center"><b>constraints</b></td><td>Must be false if <a href="#augmentation_argsrandom_pitch_shiftingenabled">augmentation_args.random_pitch_shifting.enabled</a> is set to true. Enabling it requires <a href="#use_spk_id">use_spk_id</a> to be true, and <a href="#num_spk">num_spk</a> &ge; (1 + number of targets) &times; (max <a href="#datasetsspk_id">spk_id</a> + 1).</td>
</tbody></table>

### augmentation_args.fixed_pitch_shifting.scale

Scale ratio of each target in fixed pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.5</td>
<tr><td align="center"><b>constraints</b></td><td>Must be smaller than 1.</td>
</tbody></table>

### augmentation_args.fixed_pitch_shifting.targets

Targets (in semitones) of fixed pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>list[float]</td>
<tr><td align="center"><b>default</b></td><td>[-5.0, 5.0]</td>
<tr><td align="center"><b>constraints</b></td><td>Must not contain duplicate values.</td>
</tbody></table>

### augmentation_args.random_pitch_shifting

Arguments for random pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### augmentation_args.random_pitch_shifting.enabled

Whether to apply random pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
<tr><td align="center"><b>constraints</b></td><td>Must be false if <a href="#augmentation_argsfixed_pitch_shiftingenabled">augmentation_args.fixed_pitch_shifting.enabled</a> is set to true. Enabling it requires <a href="#use_key_shift_embed">use_key_shift_embed</a> to be true.</td>
</tbody></table>

### augmentation_args.random_pitch_shifting.range

Range of the random pitch shifting (in semitones). Besides being the augmentation sampling range, this value also calibrates the `gender` parameter at inference and ONNX export time: positive gender values are scaled by `max`, negative ones by the absolute value of `min`, and the resulting key shift of a *dynamic* (curve) gender value is clipped to this range. At Python inference time, a *static* scalar gender value is scaled the same way but **not** clipped, so values with absolute magnitude larger than 1 can produce key shifts outside this range; at ONNX export time, however, a static (frozen) gender value **is** clipped to this range, and exported graphs also clip the gender input to [-1, 1] before scaling, so the key shift always stays within this range. Do not modify it after preprocessing or training, otherwise inference behavior becomes inconsistent with the training data. An error is raised at inference or export time if [use_key_shift_embed](#use_key_shift_embed) is `true` while this key is missing from the configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>list[float]</td>
<tr><td align="center"><b>default</b></td><td>[-5.0, 5.0]</td>
<tr><td align="center"><b>constraints</b></td><td>Must satisfy min &lt; 0 &lt; max.</td>
</tbody></table>

### augmentation_args.random_pitch_shifting.scale

Scale ratio of the random pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.75</td>
</tbody></table>

### augmentation_args.random_time_stretching

Arguments for random time stretching augmentation.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### augmentation_args.random_time_stretching.enabled

Whether to apply random time stretching augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
<tr><td align="center"><b>constraints</b></td><td>Enabling it requires <a href="#use_speed_embed">use_speed_embed</a> to be true.</td>
</tbody></table>

### augmentation_args.random_time_stretching.range

Range of random time stretching factors. Besides being the augmentation sampling range, this value is also read at inference and ONNX export time as the clipping bounds of the `velocity` parameter curve before it is embedded. Do not modify it after preprocessing or training, otherwise inference behavior becomes inconsistent with the training data. At ONNX export time an error is raised if [use_speed_embed](#use_speed_embed) is `true` while this key is missing from the configuration; at inference time the key is only read when the input data actually provides a `velocity` parameter curve — if no velocity curve is given, speed silently defaults to 1.0 and the key is not accessed at all (unlike the pitch shifting range, which is read unconditionally at inference).

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>list[float]</td>
<tr><td align="center"><b>default</b></td><td>[0.5, 2]</td>
<tr><td align="center"><b>constraints</b></td><td>Must satisfy 0 &lt; min &lt; 1 &lt; max.</td>
</tbody></table>

### augmentation_args.random_time_stretching.scale

Scale ratio of random time stretching augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.75</td>
</tbody></table>

### backbone_args

Keyword arguments for the backbone of the main decoder module.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

Available arguments for each backbone type are listed below.

**WaveNet** (`backbone_type: wavenet`)

| argument name | type | default | description |
| :-- | :-: | :-: | :-- |
| num_layers | int | 20 | Number of residual block layers, or depth of the network |
| num_channels | int | 512 | Number of channels, or width of the network |
| dilation_cycle_length | int | 4 | Length k of the cycle $2^0, 2^1, \ldots, 2^{k-1}$ of convolution dilation factors through WaveNet residual blocks |

**LYNXNet** (`backbone_type: lynxnet`)

| argument name | type | default | description |
| :-- | :-: | :-: | :-- |
| num_layers | int | 6 | Number of LYNXNet blocks, or depth of the network |
| num_channels | int | 1024 | Number of channels, or width of the network |
| expansion_factor | int | 2 | Channel expansion factor within each conv module |
| kernel_size | int | 31 | Kernel size of the depthwise convolution layers |
| activation | str | `PReLU` | Type of activation function. Choose from `PReLU`, `SiLU`, `ReLU`. |
| dropout_rate | float | 0.0 | Dropout rate applied in each LYNXNet block |
| strong_cond | bool | true | Whether to use strong conditioning, which injects condition before the residual split of each block |

**LYNXNet2** (`backbone_type: lynxnet2`)

| argument name | type | default | description |
| :-- | :-: | :-: | :-- |
| num_layers | int | 6 | Number of LYNXNet2 blocks, or depth of the network |
| num_channels | int | 1024 | Number of channels, or width of the network |
| kernel_size | int | 31 | Kernel size of the depthwise convolution layers |
| dropout_rate | float | 0.0 | Dropout rate applied in each LYNXNet2 block |
| use_conditioner_cache | bool | true | Whether to use Conv1d-based conditioner projection (compatible with conditioner caching) |
| glu_type | str | `atanglu` | Type of gated linear unit activation. Choose from `swiglu` for SwiGLU, `atanglu` for ATanGLU, `softsign_glu` for SoftSignGLU |
| expansion_factor | int | 1 | Channel expansion factor within each gated block (not commonly overridden) |

### backbone_type

Backbone type of the main decoder/predictor module.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>lynxnet2</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'wavenet', 'lynxnet', 'lynxnet2'.</td>
</tbody></table>

### base_config

Path(s) to other configuration files on which the current configuration is based; values in the current configuration override them.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>str | list[str]</td>
</tbody></table>

### binarization_args

Arguments for binarizers.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### binarization_args.num_workers

Number of worker subprocesses when running binarizers. More workers can speed up the preprocessing but will consume more memory. 0 means the main process does everything.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>0</td>
</tbody></table>

### binarization_args.prefer_ds

Whether to prefer loading attributes and parameters from DS files.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### binarizer_cls

Binarizer class name.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str | None</td>
<tr><td align="center"><b>default</b></td><td>null</td>
<tr><td align="center"><b>constraints</b></td><td>The base configuration may leave this as `null`; the preprocessing entry point requires a non-null importable class name.</td>
</tbody></table>

### binary_data_dir

Path to the binarized dataset.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, training</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>str | None</td>
<tr><td align="center"><b>default</b></td><td>null</td>
<tr><td align="center"><b>constraints</b></td><td>The base configuration may leave this as `null`; a non-null path must be supplied before preprocessing or training.</td>
</tbody></table>

### breathiness_db_max

Maximum breathiness value in dB used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-20.0</td>
</tbody></table>

### breathiness_db_min

Minimum breathiness value in dB used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-96.0</td>
</tbody></table>

### breathiness_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on the extracted breathiness curve.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.06</td>
</tbody></table>

### clip_grad_norm

The value at which to clip gradients. Equivalent to `gradient_clip_val` in `lightning.pytorch.Trainer`.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float | None</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### dataloader_prefetch_factor

Number of batches loaded in advance by each `torch.utils.data.DataLoader` worker.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>2</td>
</tbody></table>

### dataset_size_key

The key that indexes the binarized metadata to be used as `sizes` when batching by size.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>lengths</td>
</tbody></table>

### datasets

List of dataset configs for preprocessing.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>list[dict[str, Any]]</td>
</tbody></table>

### datasets[].language

Language context of this dataset.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>constraints</b></td><td>Must be a key of <a href="#dictionaries">dictionaries</a>.</td>
</tbody></table>

### datasets[].raw_data_dir

Path to this dataset including audio files, transcriptions, etc.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>str</td>
</tbody></table>

### datasets[].speaker

The name of the speaker of this dataset. Speaker names are mapped to speaker indexes and stored in spk_map.json when preprocessing.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>str</td>
</tbody></table>

### datasets[].spk_id

The speaker ID assigned to this dataset. Will be automatically assigned if not given. IDs can be duplicated or discontinuous to merge multiple datasets into one speaker.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int | None</td>
<tr><td align="center"><b>constraints</b></td><td>Must be smaller than <a href="#num_spk">num_spk</a>. The same speaker name must always map to the same ID.</td>
</tbody></table>

### datasets[].test_prefixes

List of data item names or name prefixes in this dataset for the validation set. For each string `s` in the list:

- If `s` equals an actual item name, add that item to the validation set.
- If `s` does not equal any item name, add all items whose names start with `s` to the validation set.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>list[str]</td>
</tbody></table>

### dictionaries

Map of language names and their corresponding dictionary file paths. The phonemes in these dictionaries will be combined into the final phoneme set and assigned phoneme IDs. Note that the phoneme set built from these dictionaries directly determines the vocabulary size of the token embedding when models are constructed or loaded (in training, inference and ONNX export), and defines how inference inputs are converted to phoneme IDs. The standard format is a mapping; `null` is accepted only for legacy single-dictionary configurations, which must provide the legacy `dictionary` path.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>dict[str, str] | None</td>
<tr><td align="center"><b>constraints</b></td><td>Every phoneme ID in the final phoneme set must occur in at least one data item, including validation items.</td>
<tr><td align="center"><b>default</b></td><td>{}</td>
</tbody></table>

### diff_accelerator

DDPM sampling acceleration method. The following methods are currently available:

- DDIM: the DDIM method from [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502).
- PNDM: the PLMS method from [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
- DPM-Solver++ adapted from [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://github.com/LuChengTHU/dpm-solver).
- UniPC adapted from [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://github.com/wl-zhao/UniPC).

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>ddim</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'ddim', 'pndm', 'dpm-solver', 'unipc'.</td>
</tbody></table>

### diff_speedup

DDPM sampling speed-up ratio. 1 means no speeding up.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>10</td>
<tr><td align="center"><b>constraints</b></td><td>Must be a factor of <a href="#k_step_infer">K_step_infer</a>.</td>
</tbody></table>

### diffusion_type

The generative modeling algorithm used by the main decoder/predictor module. The following algorithms are currently available:

- Denoising Diffusion Probabilistic Models (DDPM) from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Rectified Flow from [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)

Modifying it switches the algorithm family used by training loss computation and by inference sampling, and results in failure when loading or resuming from checkpoints, because DDPM and Rectified Flow modules keep different saved states.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>reflow</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'ddpm', 'reflow'.</td>
</tbody></table>

### dropout

Dropout rate in some FastSpeech2 modules. Modifying it does not change any parameter or saved state, so it does not prevent checkpoint loading; dropout is inactive in evaluation, so modifications only silently affect training behavior.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.1</td>
</tbody></table>

### ds_workers

Number of workers for `torch.utils.data.DataLoader`.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>4</td>
<tr><td align="center"><b>constraints</b></td><td>Must be at least 1. The data loaders are always constructed with a non-null prefetch factor and <code>persistent_workers=True</code>; setting this to 0 makes <code>torch.utils.data.DataLoader</code> raise a <code>ValueError</code> at the very beginning of training or validation.</td>
</tbody></table>

### dur_prediction_args

Arguments for phoneme duration prediction.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### dur_prediction_args.arch

Architecture of duration predictor. `'fs2'` uses the original FastSpeech2 duration predictor with standard convolution layers. `'resnet'` uses a residual-style variant with additional layer normalization and residual connections, which may improve training stability.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>resnet</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'fs2', 'resnet'.</td>
</tbody></table>

### dur_prediction_args.dropout

Dropout rate in duration predictor. Like [dropout](#dropout), modifying it does not change any parameter or saved state, so it does not prevent checkpoint loading and only silently affects training behavior.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.1</td>
</tbody></table>

### dur_prediction_args.hidden_size

Dimensions of hidden layers in duration predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>256</td>
</tbody></table>

### dur_prediction_args.kernel_size

Kernel size of convolution layers of duration predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>3</td>
</tbody></table>

### dur_prediction_args.lambda_pdur_loss

Coefficient of single-phoneme duration loss when calculating joint duration loss.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.3</td>
</tbody></table>

### dur_prediction_args.lambda_sdur_loss

Coefficient of sentence duration loss when calculating joint duration loss.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>3.0</td>
</tbody></table>

### dur_prediction_args.lambda_wdur_loss

Coefficient of word duration loss when calculating joint duration loss.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1.0</td>
</tbody></table>

### dur_prediction_args.log_offset

Offset for log domain duration loss calculation, where the following transformation is applied:
$$
D' = \ln{(D+d)}
$$
with the offset value $d$.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1.0</td>
</tbody></table>

### dur_prediction_args.loss_type

Underlying loss type of duration loss.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>mse</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'mse', 'huber'.</td>
</tbody></table>

### dur_prediction_args.num_layers

Number of duration predictor layers.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>5</td>
</tbody></table>

### enc_ffn_kernel_size

Size of TransformerFFNLayer convolution kernel in FastSpeech2 encoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>3</td>
</tbody></table>

### enc_layers

Number of FastSpeech2 encoder layers.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>4</td>
</tbody></table>

### energy_db_max

Maximum energy value in dB used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-12.0</td>
</tbody></table>

### energy_db_min

Minimum energy value in dB used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-96.0</td>
</tbody></table>

### energy_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on the extracted energy curve.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.06</td>
</tbody></table>

### extra_phonemes

Extra phonemes to be added to the phoneme set. This list can be used to define custom global phoneme tags besides `AP` and `SP`, or to contain phonemes that are not present in any of the dictionaries. Like [dictionaries](#dictionaries), this list directly determines the vocabulary size of the token embedding when models are constructed or loaded.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>list[str] | None</td>
<tr><td align="center"><b>default</b></td><td>[]</td>
</tbody></table>

### f0_max

Maximum fundamental frequency (F0) in Hz for pitch extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1100</td>
</tbody></table>

### f0_min

Minimum fundamental frequency (F0) in Hz for pitch extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>65</td>
</tbody></table>

### ffn_act

Activation function of TransformerFFNLayer in FastSpeech2 encoder:

- `torch.nn.ReLU` if 'relu'
- `torch.nn.GELU` if 'gelu'
- `torch.nn.SiLU` if 'swish'
- `SwiGLU` if 'swiglu'
- `ATanGLU` if 'atanglu'

The last two are gated linear unit activations (the filter size of the first convolution is internally doubled to compensate for the halved output of the GLU). Switching between a GLU-family activation and a non-GLU one changes parameter shapes and prevents checkpoint loading; switching within the non-GLU family (`relu`, `gelu`, `swish`) keeps shapes unchanged and does not prevent checkpoint loading, but silently changes the behavior of an already trained model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>gelu</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'relu', 'gelu', 'swish', 'swiglu', 'atanglu'.</td>
</tbody></table>

### fft_size

Fast Fourier Transform parameter for mel extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>2048</td>
</tbody></table>

### finetune_enabled

Whether to finetune from a pretrained model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### finetune_ckpt_path

Path to the pretrained model for finetuning.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str | None</td>
<tr><td align="center"><b>default</b></td><td>null</td>
</tbody></table>

### finetune_ignored_params

Prefixes of parameter key names in the state dict of the pretrained model that need to be dropped before finetuning.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>list[str] | None</td>
<tr><td align="center"><b>default</b></td><td>[]</td>
</tbody></table>

### finetune_strict_shapes

Whether to raise an error if the tensor shapes of any parameter of the pretrained model and the target model mismatch. If set to `false`, parameters with mismatching shapes will be skipped.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### fmax

Maximum frequency of mel extraction. `null` uses the Nyquist frequency (`audio_sample_rate / 2`).

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>float | None</td>
<tr><td align="center"><b>default</b></td><td>16000</td>
</tbody></table>

### fmin

Minimum frequency of mel extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>40</td>
</tbody></table>

### freezing_enabled

Whether to enable parameter freezing during training.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### frozen_params

Parameter name prefixes to freeze during training.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>list[str]</td>
<tr><td align="center"><b>default</b></td><td>[]</td>
</tbody></table>

### glide_embed_scale

The scale factor by which the glide embedding values are multiplied for melody encoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>11.313708498984760</td>
</tbody></table>

### glide_types

Type names of glide notes.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>list[str]</td>
<tr><td align="center"><b>default</b></td><td>['up', 'down']</td>
<tr><td align="center"><b>constraints</b></td><td>Type name <code>none</code> is reserved (index 0 in the glide embedding, whose size is <code>len(glide_types) + 1</code>) and must not appear in this list.</td>
</tbody></table>

### hidden_size

Dimension of hidden layers of FastSpeech2, token and parameter embeddings, and diffusion condition.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>384</td>
</tbody></table>

### hnsep

Harmonic-noise separation algorithm type.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>vr</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'world', 'vr'.</td>
</tbody></table>

### hnsep_ckpt

Checkpoint or model path of NN-based harmonic-noise separator.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>checkpoints/vr/model.pt</td>
</tbody></table>

### hop_size

Hop size or step length (in number of waveform samples) of mel and feature extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>512</td>
</tbody></table>

### lambda_aux_mel_loss

Coefficient of aux mel loss when calculating total loss of acoustic model with shallow diffusion.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.2</td>
</tbody></table>

### lambda_dur_loss

Coefficient of duration loss when calculating total loss of variance model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1.0</td>
</tbody></table>

### lambda_pitch_loss

Coefficient of pitch loss when calculating total loss of variance model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1.0</td>
</tbody></table>

### lambda_var_loss

Coefficient of variance loss (all variance parameters other than pitch, like energy, breathiness, etc.) when calculating total loss of variance model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1.0</td>
</tbody></table>

### K_step

Maximum number of DDPM steps used by shallow diffusion. Only takes effect when [diffusion_type](#diffusion_type) is `'ddpm'` and [use_shallow_diffusion](#use_shallow_diffusion) is set to `true`; with Rectified Flow the shallow starting point is controlled by [T_start](#t_start) instead, and this key is ignored.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>400</td>
<tr><td align="center"><b>constraints</b></td><td>Must not be larger than <a href="#timesteps">timesteps</a>.</td>
</tbody></table>

### K_step_infer

Number of DDPM steps used during shallow diffusion inference. Normally set to the same value as [K_step](#k_step). Only takes effect when [diffusion_type](#diffusion_type) is `'ddpm'` and [use_shallow_diffusion](#use_shallow_diffusion) is set to `true`; with Rectified Flow the shallow starting point is controlled by [T_start_infer](#t_start_infer) instead, and this key is ignored.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>400</td>
<tr><td align="center"><b>constraints</b></td><td>Should be no larger than <a href="#k_step">K_step</a>. Values larger than <a href="#k_step">K_step</a> are silently clamped to <a href="#k_step">K_step</a> instead of raising errors.</td>
</tbody></table>

### log_interval

Controls how often training metrics are logged to TensorBoard, measured in global training steps.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>100</td>
</tbody></table>

### lr_scheduler_args

Arguments of learning rate scheduler. Keys will be used as keyword arguments of the `__init__()` method of [lr_scheduler_args.scheduler_cls](#lr_scheduler_argsscheduler_cls).

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### lr_scheduler_args.scheduler_cls

Learning rate scheduler class name.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>torch.optim.lr_scheduler.StepLR</td>
</tbody></table>

### main_loss_log_norm

Whether to use log-normalized weight for the main loss. This is similar to the method in the Stable Diffusion 3 paper [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206). Only takes effect when [diffusion_type](#diffusion_type) is `'reflow'`; ignored with DDPM.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### main_loss_type

Loss type of the main decoder/predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>l2</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'l1', 'l2'.</td>
</tbody></table>

### max_batch_frames

Maximum number of data frames in each training batch. Used to dynamically control the batch size.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>50000</td>
</tbody></table>

### max_batch_size

The maximum training batch size.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>64</td>
</tbody></table>

### max_beta

Max beta of the DDPM noise schedule. Only takes effect when [diffusion_type](#diffusion_type) is `'ddpm'` and [schedule_type](#schedule_type) is `'linear'`; ignored with Rectified Flow and with the cosine schedule. The noise schedule derived from this value is saved as persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but the value is silently overridden by the buffers stored in the checkpoint, so it only takes effect when training from scratch.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.02</td>
</tbody></table>

### max_updates

Stop training after this number of steps. Equivalent to `max_steps` in `lightning.pytorch.Trainer`.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>100000</td>
</tbody></table>

### max_val_batch_frames

Maximum number of data frames in each validation batch.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>60000</td>
</tbody></table>

### max_val_batch_size

The maximum validation batch size.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### mel_base

The logarithmic base of the mel-spectrogram calculation. The legacy value `10` (integer or string `'10'`) and the natural-log value `'e'` are accepted by vocoder compatibility paths. New dataset preprocessing and NSF-HiFiGAN export require `'e'`.

**WARNING: Since the v2.4.0 release, this value is no longer configurable for preprocessing new datasets.**

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str | int</td>
<tr><td align="center"><b>default</b></td><td>e</td>
<tr><td align="center"><b>constraints</b></td><td>Use `'e'` for current preprocessing and export; legacy vocoder paths may also accept `'10'` or `10`.</td>
</tbody></table>

### mel_vmax

Maximum mel-spectrogram heatmap value for TensorBoard plotting.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>4.</td>
</tbody></table>

### mel_vmin

Minimum mel-spectrogram heatmap value for TensorBoard plotting.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-14.</td>
</tbody></table>

### melody_encoder_args

Arguments for melody encoder. Available sub-keys: `hidden_size`, `enc_layers`, `enc_ffn_kernel_size`, `ffn_act`, `dropout`, `num_heads`, `use_pos_embed`, `rel_pos`, `use_rope`. If any parameter does not exist in this configuration key, it inherits from the linguistic encoder. The scope implications of each sub-key follow the root-level keys of the same names.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### merged_phoneme_groups

Phoneme groups to merge. Each group is a phoneme name list. The merged phonemes share the same ID and thus the same phoneme embedding. Like [dictionaries](#dictionaries), these groups directly determine the vocabulary size of the token embedding when models are constructed or loaded.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>list[list[str]] | None</td>
<tr><td align="center"><b>default</b></td><td>[]</td>
</tbody></table>

### midi_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on the step function representing MIDI sequence for base pitch calculation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.06</td>
</tbody></table>

### mix_ln_layer

List of 0-based encoder layer indices where Mixed LayerNorm is applied. Only takes effect when [use_mix_ln](#use_mix_ln) is enabled. For each selected layer, both self-attention layer norm and FFN layer norm are replaced with `Mixed_LayerNorm` which conditions the normalization on speaker embedding.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>list[int]</td>
<tr><td align="center"><b>default</b></td><td>[0, 2]</td>
<tr><td align="center"><b>constraints</b></td><td>Every element should be in the range [0, <a href="#enc_layers">enc_layers</a>).</td>
</tbody></table>

### nccl_p2p

Whether to enable P2P when using NCCL as the backend. Set it to `false` if the training process is stuck upon beginning.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### num_ckpt_keep

Number of newest checkpoints kept during training.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>8</td>
</tbody></table>

### num_heads

The number of attention heads of the in-house `MultiheadSelfAttentionWithRoPE` (formerly `torch.nn.MultiheadAttention`, which has been deprecated due to ONNX export issues) in FastSpeech2 encoder. This does not change parameter shapes (the Q/K/V and output projections have the same shapes regardless of the number of heads); modifying it does not prevent checkpoint loading, but silently changes the behavior of an already trained model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>2</td>
<tr><td align="center"><b>constraints</b></td><td><a href="#hidden_size">hidden_size</a> must be divisible by <a href="#num_heads">num_heads</a>. When both <a href="#use_pos_embed">use_pos_embed</a> and <a href="#use_rope">use_rope</a> are true, <a href="#hidden_size">hidden_size</a> must be divisible by 2 &times; <a href="#num_heads">num_heads</a>.</td>
</tbody></table>

### num_lang

Number of languages. This value is used to allocate language embeddings in the linguistic encoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1</td>
<tr><td align="center"><b>constraints</b></td><td>Must be at least the number of entries in <a href="#dictionaries">dictionaries</a>.</td>
</tbody></table>

### num_sanity_val_steps

Number of sanity validation steps at the beginning.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int | None</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### num_spk

Maximum number of speakers in multi-speaker models.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### num_valid_plots

Number of validation plots for each validation run. Plots will be chosen from the start of the validation set.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>10</td>
</tbody></table>

### optimizer_args

Arguments of optimizer. Keys will be used as keyword arguments of the `__init__()` method of [optimizer_args.optimizer_cls](#optimizer_argsoptimizer_cls).

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### optimizer_args.optimizer_cls

Optimizer class name. The following optimizers are currently recommended:

- `torch.optim.AdamW` — Standard AdamW optimizer. Set `weight_decay` and other arguments (`lr`, `betas`, `eps`, ...) as top-level keys of [optimizer_args](#optimizer_args).
- `modules.optimizer.muon.Muon_AdamW` — Chained optimizer that applies Muon (MomentUm Orthogonalized by Newton-Schulz) to internal weight matrices (e.g. linear layers) and AdamW to other parameters (e.g. biases, embeddings). Per-optimizer arguments are configured via the `muon_args` and `adamw_args` sub-keys under [optimizer_args](#optimizer_args), while the top-level `lr` and `weight_decay` serve as the shared defaults of both sub-optimizers. Note that an `lr` set in either sub-key takes no effect in practice: at every `optimizer.step()` the top-level `lr` is copied into all parameter groups of the sub-optimizers, so that the learning rate scheduler keeps applying.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>modules.optimizer.muon.Muon_AdamW</td>
</tbody></table>

### pe

Pitch extraction algorithm type.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>parselmouth</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'parselmouth', 'rmvpe', 'harvest'.</td>
</tbody></table>

### pe_ckpt

Checkpoint or model path of NN-based pitch extractor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>checkpoints/rmvpe/model.pt</td>
</tbody></table>

### permanent_ckpt_interval

The interval (in number of training steps) of permanent checkpoints. Permanent checkpoints will not be removed even if they are not the newest ones. Permanent checkpoints are enabled only when this value is larger than 9 and [permanent_ckpt_start](#permanent_ckpt_start) is larger than 0; `null` or `false` is normalized to 0 and silently disables them.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int | bool | None</td>
<tr><td align="center"><b>default</b></td><td>10000</td>
</tbody></table>

### permanent_ckpt_start

Checkpoints are only saved at validation checks, i.e. every [val_check_interval](#val_check_interval) global steps (the interval passed to the trainer is multiplied by [accumulate_grad_batches](#accumulate_grad_batches), so proportionally more micro-batches run between validation checks when gradient accumulation is enabled). A saved checkpoint is kept as permanent if its step count is no less than this value and the difference is divisible by [permanent_ckpt_interval](#permanent_ckpt_interval). Milestone steps that do not coincide with a saved checkpoint are skipped, so the effective cadence of permanent checkpoints is the least common multiple of the two intervals. Permanent checkpoints are enabled only when this value is larger than 0 and [permanent_ckpt_interval](#permanent_ckpt_interval) is larger than 9; `null` or `false` is normalized to 0 and silently disables them.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int | bool | None</td>
<tr><td align="center"><b>default</b></td><td>60000</td>
</tbody></table>

### pitch_prediction_args

Arguments for pitch prediction.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### pitch_prediction_args.backbone_args

Equivalent to [backbone_args](#backbone_args) but only for the pitch predictor model.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### pitch_prediction_args.backbone_type

Equivalent to [backbone_type](#backbone_type) but only for the pitch predictor model. If not set, use the root backbone type.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>lynxnet2</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'wavenet', 'lynxnet', 'lynxnet2'.</td>
</tbody></table>

### pitch_prediction_args.pitd_clip_max

Maximum clipping value (in semitones) of pitch delta between actual pitch and base pitch.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>12.0</td>
</tbody></table>

### pitch_prediction_args.pitd_clip_min

Minimum clipping value (in semitones) of pitch delta between actual pitch and base pitch.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-12.0</td>
</tbody></table>

### pitch_prediction_args.pitd_norm_max

Maximum pitch delta value in semitones used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>8.0</td>
</tbody></table>

### pitch_prediction_args.pitd_norm_min

Minimum pitch delta value in semitones used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-8.0</td>
</tbody></table>

### pitch_prediction_args.repeat_bins

Number of repeating bins in the pitch predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>64</td>
</tbody></table>

### pl_trainer_accelerator

Type of Lightning trainer hardware accelerator.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>auto</td>
<tr><td align="center"><b>constraints</b></td><td>See <a href="https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html?highlight=accelerator">Accelerator — PyTorch Lightning 2.X.X documentation</a> for available values.</td>
</tbody></table>

### pl_trainer_devices

Determines which device(s) the model should be trained on.

`'auto'` will utilize all visible devices defined with the `CUDA_VISIBLE_DEVICES` environment variable, or utilize all available devices if that variable is not set. Otherwise, it behaves like `CUDA_VISIBLE_DEVICES` which can filter out visible devices. Lightning also accepts a positive device count as an integer or a list of device indices.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str | int | list[int]</td>
<tr><td align="center"><b>default</b></td><td>auto</td>
</tbody></table>

### pl_trainer_precision

The computation precision of training.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str | int | None</td>
<tr><td align="center"><b>default</b></td><td>16-mixed</td>
<tr><td align="center"><b>constraints</b></td><td>Lightning accepts integer precisions `16`, `32`, `64` and string forms such as `'32-true'`, `'bf16-mixed'` and `'16-mixed'`; `null` is passed through to Lightning and falls back to `'32-true'`. See the <a href="https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api">Trainer — PyTorch Lightning 2.X.X documentation</a> for the version-specific list.</td>
</tbody></table>

### pl_trainer_num_nodes

Number of nodes in the training cluster of Lightning trainer.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### pl_trainer_strategy

Arguments of Lightning Strategy. Values will be used as keyword arguments when constructing the Strategy object.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### pl_trainer_strategy.name

Strategy name for the Lightning trainer.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>auto</td>
</tbody></table>

### predict_breathiness

Whether to enable breathiness prediction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### predict_dur

Whether to enable phoneme duration prediction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### predict_energy

Whether to enable energy prediction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### predict_pitch

Whether to enable pitch prediction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### predict_tension

Whether to enable tension prediction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### predict_voicing

Whether to enable voicing prediction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### rel_pos

Whether to use relative positional encoding in FastSpeech2 module. Only consulted when [use_rope](#use_rope) is `false`: with `rel_pos: false` the encoder uses `SinusoidalPositionalEmbedding`, which owns a persistent buffer saved in checkpoints, so toggling this option changes the set of saved keys and results in failure when loading or resuming from checkpoints.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### rope_interleaved

Whether to use the interleaved (alternating) layout for RoPE (Rotary Positional Encoding) in the encoder self-attention. When set to `false`, the non-interleaved (contiguous half-real-half-imaginary) layout is used instead. This option only changes the layout of the frequency buffers, which are recomputed at initialization; modifying it does not change parameter shapes or prevent checkpoint loading, but silently changes the behavior of an already trained model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### sampler_frame_count_grid

The batch sampler applies an algorithm called _sorting by similar length_ when collecting batches. Data samples are first shuffled, and then stably sorted by their approximate lengths, so that samples of similar lengths are grouped together while the order within each group stays random. Assuming this value is set to $L_{grid}$, the approximate length of a data sample with length $L_{real}$ can be calculated through the following expression:

$$
L_{approx} = \max\left(\mathrm{round}\left(\frac{L_{real}}{L_{grid}}\right)\cdot L_{grid},\; L_{grid}\right)
$$

where $\mathrm{round}$ is the nearest-integer rounding (round half to even), and the result is clamped to a minimum of $L_{grid}$.

Training performance on some datasets may be very sensitive to this value. Change it to 1 (approximate length becomes the exact length, so batches are perfectly sorted by length) to get the best performance in theory.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>6</td>
</tbody></table>

### sampling_algorithm

The algorithm to solve the ODE of Rectified Flow. The following methods are currently available:

- Euler: the Euler method.
- Runge-Kutta (order 2): the 2nd-order Runge-Kutta method.
- Runge-Kutta (order 4): the 4th-order Runge-Kutta method.
- Runge-Kutta (order 5): the 5th-order Runge-Kutta method.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>euler</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'euler', 'rk2', 'rk4', 'rk5'.</td>
</tbody></table>

### sampling_steps

The total number of sampling steps to solve the Rectified Flow ODE. Note that this value may not be equal to NFE (Number of Function Evaluations) because some methods may require more than one function evaluation per step.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>20</td>
</tbody></table>

### schedule_type

The DDPM schedule type. Only takes effect when [diffusion_type](#diffusion_type) is `'ddpm'`; ignored with Rectified Flow. Like [max_beta](#max_beta), the derived noise schedule is saved as persistent buffers in checkpoints, so modifying this value for an existing experiment is silently overridden on checkpoint loading and only takes effect when training from scratch.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>linear</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'linear', 'cosine'.</td>
</tbody></table>

### shallow_diffusion_args

Arguments for shallow diffusion.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### shallow_diffusion_args.aux_decoder_arch

Architecture type of the auxiliary decoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>convnext</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'convnext'.</td>
</tbody></table>

### shallow_diffusion_args.aux_decoder_args

Keyword arguments for dynamically constructing the auxiliary decoder.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### shallow_diffusion_args.aux_decoder_grad

Scale factor of the gradients from the auxiliary decoder to the encoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.1</td>
</tbody></table>

### shallow_diffusion_args.train_aux_decoder

Whether to run the auxiliary decoder in both the forward and backward passes during training. If set to `false`, the auxiliary decoder remains in memory and does not get any updates.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### shallow_diffusion_args.train_diffusion

Whether to run the diffusion (main) decoder in both the forward and backward passes during training. If set to `false`, the diffusion decoder remains in memory and does not get any updates.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### shallow_diffusion_args.val_gt_start

Whether to use the ground truth as `x_start` in the shallow diffusion validation process. If set to `true`, Gaussian noise is added to the ground truth before shallow diffusion is performed; otherwise the noise is added to the output of the auxiliary decoder. This option is useful when the auxiliary decoder has not been trained yet. It only takes effect in validation runs during training, where a ground truth mel-spectrogram is available; pure inference (where none is given) is unaffected.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### sort_by_len

Whether to apply the _sorting by similar length_ algorithm described in [sampler_frame_count_grid](#sampler_frame_count_grid). Turning off this option may slow down training because sorting by length can better utilize the computing resources.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### spec_min

Minimum mel-spectrogram value used for normalization to [-1, 1]. Different mel bins can have different minimum values. Note that with `diffusion_type: ddpm` these values are stored as persistent buffers in checkpoints: changing the list length causes checkpoint loading to fail, while changed values are silently overridden by the checkpoint on loading; with Rectified Flow they are always read from the current configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>list[float]</td>
<tr><td align="center"><b>default</b></td><td>[-12]</td>
<tr><td align="center"><b>constraints</b></td><td>Must contain either one value or <a href="#audio_num_mel_bins">audio_num_mel_bins</a> values.</td>
</tbody></table>

### spec_max

Maximum mel-spectrogram value used for normalization to [-1, 1]. Different mel bins can have different maximum values. For buffer persistence behavior in checkpoints, see the note in [spec_min](#spec_min).

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>list[float]</td>
<tr><td align="center"><b>default</b></td><td>[0.0]</td>
<tr><td align="center"><b>constraints</b></td><td>Must contain either one value or <a href="#audio_num_mel_bins">audio_num_mel_bins</a> values.</td>
</tbody></table>

### T_start

The starting value of time $t$ in the Rectified Flow ODE which applies for $t \in (T_{start}, 1)$. Only takes effect when [use_shallow_diffusion](#use_shallow_diffusion) is set to `true`; otherwise it is forced to 0. The [0, 1] range constraint is asserted only when shallow diffusion is enabled.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.4</td>
<tr><td align="center"><b>constraints</b></td><td>Must be in the range [0, 1].</td>
</tbody></table>

### T_start_infer

The starting value of time $t$ in the ODE during shallow Rectified Flow inference. Normally set to the same value as [T_start](#t_start); when this key is not set, [T_start](#t_start) is used as the fallback. Only takes effect when [use_shallow_diffusion](#use_shallow_diffusion) is set to `true`; ignored otherwise.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.4</td>
<tr><td align="center"><b>constraints</b></td><td>Should be no less than <a href="#t_start">T_start</a>. This is not asserted: smaller values silently sample from time steps outside the trained range. Values greater than or equal to 1 are silently treated as 1, i.e., the shallow diffusion source is returned without any actual sampling; values no greater than 0 are silently treated as 0, i.e., full sampling from pure noise.</td>
</tbody></table>

### task_cls

Task trainer class name.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str | None</td>
<tr><td align="center"><b>default</b></td><td>null</td>
<tr><td align="center"><b>constraints</b></td><td>The base configuration may leave this as `null`; the training entry point requires a non-null importable class name.</td>
</tbody></table>

### tension_logit_max

Maximum tension logit value used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration. Logits are calculated using the inverse of Sigmoid function:

$$
f(x) = \ln\frac{x}{1-x}
$$

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>10.0</td>
</tbody></table>

### tension_logit_min

Minimum tension logit value used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration. Logits are calculated using the inverse of Sigmoid function:

$$
f(x) = \ln\frac{x}{1-x}
$$

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-10.0</td>
</tbody></table>

### tension_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on the extracted tension curve.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.06</td>
</tbody></table>

### time_scale_factor

The scale factor that applied to time $t$ of Rectified Flow before embedding into the model. It is read in both the training loss computation and the inference ODE solver, and is baked into exported ONNX graphs; modifying it does not change parameter shapes or prevent checkpoint loading, but silently changes the behavior of an already trained model. Only takes effect when [diffusion_type](#diffusion_type) is `'reflow'`; with DDPM the time scaling is internally fixed to [timesteps](#timesteps) and this key is ignored.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1000</td>
</tbody></table>

### timesteps

Total number of DDPM steps. Only takes effect when [diffusion_type](#diffusion_type) is `'ddpm'`; ignored with Rectified Flow, whose sampling grid is controlled by [sampling_steps](#sampling_steps) and [T_start_infer](#t_start_infer) instead.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1000</td>
</tbody></table>

### use_breathiness_embed

Whether to accept and embed breathiness values into the model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_energy_embed

Whether to accept and embed energy values into the model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_glide_embed

Whether to accept and embed glide types in the melody encoder. This option only takes effect when [use_melody_encoder](#use_melody_encoder) is enabled.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_key_shift_embed

Whether to embed key shifting values introduced by random pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
<tr><td align="center"><b>constraints</b></td><td>Must be true if <a href="#augmentation_argsrandom_pitch_shiftingenabled">random pitch shifting</a> is enabled.</td>
</tbody></table>

### use_lang_id

Whether to embed the language ID from a multilingual dataset. This option only takes effect for those cross-lingual phonemes in the merged groups. Language IDs are always extracted and stored by binarizers regardless of this value, so enabling it after preprocessing does not require re-running binarizers.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_melody_encoder

Whether to enable the melody encoder for the pitch predictor. This option only takes effect when [predict_pitch](#predict_pitch) is true; otherwise the melody encoder is not built regardless of this value.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### use_mix_ln

Whether to use Mixed LayerNorm with speaker-conditioned mixup in the acoustic encoder. When enabled, encoder layers specified in [mix_ln_layer](#mix_ln_layer) use `Mixed_LayerNorm`, which mixes the standard layer normalization with a speaker-conditioned scale factor, allowing speaker identity to influence the normalization behavior.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_pos_embed

Whether to enable positional encoding in FastSpeech2 encoder. When [use_rope](#use_rope) is `false`, this key controls the additive input embedding (`SinusoidalPositionalEmbedding` when `rel_pos` is `false`, or `RelPositionalEncoding` when `rel_pos` is `true`). When `use_rope` is `true`, no additive embedding is created, but RoPE is only created if this key is also `true` — disabling it removes RoPE as well and leaves the encoder with no positional encoding at all. The additive embedding module itself is created based on [use_rope](#use_rope) and [rel_pos](#rel_pos) alone, regardless of this key, so toggling it never changes parameter shapes or the set of saved keys and never prevents checkpoint loading; it only selects whether the positional encoding is actually applied at run time (and, when `use_rope` is `true`, whether RoPE is created and applied in attention), which changes the behavior of both training and inference. Since an already trained model expects its trained positional encoding scheme, modifying it silently produces inconsistent or wrong outputs.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### use_rope

Whether to use RoPE (Rotary Positional Encoding) in FastSpeech2 encoder. RoPE is only created when [use_pos_embed](#use_pos_embed) is also `true`; otherwise the encoder gets no positional encoding. When enabled, no positional embedding is added to the encoder input, so [rel_pos](#rel_pos) has no effect. RoPE itself keeps no parameters, and its frequency buffers are recomputed at initialization and never saved in checkpoints; however, enabling RoPE removes and disabling RoPE creates the input positional embedding module. When [rel_pos](#rel_pos) is `true` that module (`RelPositionalEncoding`) owns no parameters or persistent buffers, so toggling this option does not prevent checkpoint loading but silently changes the behavior of an already trained model. When `rel_pos` is `false` that module is a `SinusoidalPositionalEmbedding`, which owns a persistent buffer saved in checkpoints, so toggling this option then changes the set of saved keys and results in failure when loading or resuming from checkpoints.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### use_shallow_diffusion

Whether to use shallow diffusion.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### use_speed_embed

Whether to embed speed values introduced by random time stretching augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
<tr><td align="center"><b>constraints</b></td><td>Must be true if <a href="#augmentation_argsrandom_time_stretchingenabled">random time stretching</a> is enabled.</td>
</tbody></table>

### use_spk_id

Whether to embed the speaker ID from a multi-speaker dataset. Speaker IDs are always extracted and stored by binarizers regardless of this value, so enabling it after preprocessing does not require re-running binarizers.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_stretch_embed

Whether to embed the per-frame relative position within phonemes into the encoder. The value is computed by the `StretchRegulator` module: for each mel frame, its zero-based position within its phoneme is divided by that phoneme's duration, forming a normalized ramp from 0 to 1.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### use_tension_embed

Whether to accept and embed tension values into the model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_variance_scaling

Whether to normalize variance-related inputs to compress their dynamic range before embedding. When enabled: phoneme durations are embedded in log space via `log(1 + dur)` in the acoustic task, and in the variance task only when [predict_dur](#predict_dur) is `false` — in the word mode of the variance task (`predict_dur: true`), word durations are embedded linearly without log scaling; note durations in the melody encoder are embedded via `log(1 + dur)`; MIDI note numbers are divided by 128; pitch is divided by 12; in the pitch prediction branch, the division differs by mode: when the melody encoder is disabled, base pitch is divided by 128 before embedding, but when the melody encoder is enabled (see [use_melody_encoder](#use_melody_encoder), which defaults to `true`), base pitch is not embedded at all and delta pitch (pitch minus base pitch) divided by 12 is embedded instead; energy, breathiness and voicing are divided by 96; tension is multiplied by 0.1; key shift is divided by 12. This scaling helps the model handle the wide range of these values more stably during training and inference. It only selects the scaling factors applied inside the model graph and does not change parameter shapes, so modifying it does not prevent checkpoint loading, but silently changes the behavior of an already trained model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### use_voicing_embed

Whether to accept and embed voicing values into the model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### val_check_interval

Interval (in number of optimizer updates, i.e. global steps) between validation checks. The value actually passed to the trainer is multiplied by [accumulate_grad_batches](#accumulate_grad_batches), so when gradient accumulation is larger than 1, proportionally more micro-batches run between validation checks.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>4000</td>
</tbody></table>

### val_with_vocoder

Whether to load and use the vocoder to generate audio during validation. Validation audio will not be available if this option is disabled.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### variances_prediction_args

Arguments for predicting variance parameters other than pitch, such as energy, breathiness, etc.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### variances_prediction_args.backbone_args

Equivalent to [backbone_args](#backbone_args) but only for the multi-variance predictor.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict[str, Any]</td>
</tbody></table>

### variances_prediction_args.backbone_type

Equivalent to [backbone_type](#backbone_type) but only for the multi-variance predictor model. If not set, use the root backbone type.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>lynxnet2</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'wavenet', 'lynxnet', 'lynxnet2'.</td>
</tbody></table>

### variances_prediction_args.total_repeat_bins

Total number of repeating bins in the multi-variance predictor. Repeating bins are distributed evenly among the variance parameters.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>72</td>
<tr><td align="center"><b>constraints</b></td><td>Must be divisible by the number of predicted variance parameters.</td>
</tbody></table>

### vocoder

Vocoder class name.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>NsfHifiGAN</td>
</tbody></table>

### vocoder_ckpt

Checkpoint or model path of NN-based vocoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>checkpoints/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt</td>
</tbody></table>

### voicing_db_max

Maximum voicing value in dB used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-12.0</td>
</tbody></table>

### voicing_db_min

Minimum voicing value in dB used for normalization to [-1, 1]. Note that with [diffusion_type](#diffusion_type) `'ddpm'`, this value is latched into persistent buffers in checkpoints: modifying it for an existing experiment does not raise errors, but is silently overridden by the checkpoint on loading, so it only takes effect when training from scratch; with Rectified Flow it is always read from the current configuration.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-96.0</td>
</tbody></table>

### voicing_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on the extracted voicing curve.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.06</td>
</tbody></table>

### win_size

Window size for mel or feature extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>2048</td>
</tbody></table>
