# Configuration Schemas

## The configuration system

DiffSinger uses a cascading configuration system based on YAML files. All configuration files originally inherit and override [configs/base.yaml](../configs/base.yaml), and each file directly override another file by setting the `base_config` attribute. The overriding rules are:

- Configuration keys with the same path and the same name will be replaced. Other paths and names will be merged.
- All configurations in the inheritance chain will be squashed (via the rule above) as the final configuration.
- The trainer will save the final configuration in the experiment directory, which is detached from the chain and made independent from other configuration files.

## Configurable parameters

This following are the meaning and usages of all editable keys in a configuration file.

Each configuration key (including nested keys) are described with a brief explanation and several attributes listed as follows:

|    Attribute    | Explanation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|:---------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   visibility    | Represents what kind(s) of models and tasks this configuration belongs to.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|      scope      | The scope of effects of the configuration, indicating what it can influence within the whole pipeline. Possible values are:<br>**nn** - This configuration is related to how the neural networks are formed and initialized. Modifying it will result in failure when loading or resuming from checkpoints.<br>**preprocessing** - This configuration controls how raw data pieces or inference inputs are converted to inputs of neural networks. Binarizers should be re-run if this configuration is modified.<br>**training** - This configuration describes the training procedures. Most training configurations can affect training performance, memory consumption, device utilization and loss calculation. Modifying training-only configurations will not cause severe inconsistency or errors in most situations.<br>**inference** - This configuration describes the calculation logic through the model graph. Changing it can lead to inconsistent or wrong outputs of inference or validation.<br>**others** - Other configurations not discussed above. Will have different effects according to the descriptions.                                                          |
| customizability | The level of customizability of the configuration. Possible values are:<br>**required** - This configuration **must** be set or modified according to the actual situation or condition, otherwise errors can be raised.<br>**recommended** - It is recommended to adjust this configuration according to the dataset, requirements, environment and hardware. Most functionality-related and feature-related configurations are at this level, and all configurations in this level are widely tested with different values. However, leaving it unchanged will not cause problems.<br>**normal** - There is no need to modify it as the default value is carefully tuned and widely validated. However, one can still use another value if there are some special requirements or situations.<br>**not recommended** - No other values except the default one of this configuration are tested. Modifying it will not cause errors, but may cause unpredictable or significant impacts to the pipelines.<br>**reserved** - This configuration **must not** be modified. It appears in the configuration file only for future scalability, and currently changing it will result in errors. |
|      type       | Value type of the configuration. Follows the syntax of Python type hints.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|   constraints   | Value constraints of the configuration.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|     default     | Default value of the configuration. Uses YAML value syntax.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

### accumulate_grad_batches

Indicates that gradients of how many training steps are accumulated before each `optimizer.step()` call. 1 means no gradient accumulation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
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
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>44100</td>
</tbody></table>

### augmentation_args

Arguments for data augmentation.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### augmentation_args.fixed_pitch_shifting

Arguments for fixed pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### augmentation_args.fixed_pitch_shifting.enabled

Whether to apply fixed pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
<tr><td align="center"><b>constraints</b></td><td>Must be false if <a href="#augmentation_argsrandom_pitch_shiftingenabled">augmentation_args.random_pitch_shifting.enabled</a> is set to true.</td>
</tbody></table>

### augmentation_args.fixed_pitch_shifting.scale

Scale ratio of each target in fixed pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>tuple</td>
<tr><td align="center"><b>default</b></td><td>0.5</td>
</tbody></table>

### augmentation_args.fixed_pitch_shifting.targets

Targets (in semitones) of fixed pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>tuple</td>
<tr><td align="center"><b>default</b></td><td>[-5.0, 5.0]</td>
</tbody></table>

### augmentation_args.random_pitch_shifting

Arguments for random pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### augmentation_args.random_pitch_shifting.enabled

Whether to apply random pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
<tr><td align="center"><b>constraints</b></td><td>Must be false if <a href="#augmentation_argsfixed_pitch_shiftingenabled">augmentation_args.fixed_pitch_shifting.enabled</a> is set to true.</td>
</tbody></table>

### augmentation_args.random_pitch_shifting.range

Range of the random pitch shifting ( in semitones).

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>tuple</td>
<tr><td align="center"><b>default</b></td><td>[-5.0, 5.0]</td>
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
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### augmentation_args.random_time_stretching.enabled

Whether to apply random time stretching augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### augmentation_args.random_time_stretching.range

Range of random time stretching factors.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>tuple</td>
<tr><td align="center"><b>default</b></td><td>[0.5, 2]</td>
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

### backbone_type

Backbone type of the main decoder/predictor module.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>wavenet</td>
</tbody></table>

### base_config

Path(s) of other config files that the current config is based on and will override.

<table><tbody>
<tr><td align="center"><b>scope</b></td><td>others</td>
<tr><td align="center"><b>type</b></td><td>Union[str, list]</td>
</tbody></table>

### binarization_args

Arguments for binarizers.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### binarization_args.num_workers

Number of worker subprocesses when running binarizers. More workers can speed up the preprocessing but will consume more memory. 0 means the main processing doing everything.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### binarization_args.prefer_ds

Whether to prefer loading attributes and parameters from DS files.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>False</td>
</tbody></table>

### binarization_args.shuffle

Whether binarized dataset will be shuffled or not.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### binarizer_cls

Binarizer class name.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str</td>
</tbody></table>

### binary_data_dir

Path to the binarized dataset.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, training</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>str</td>
</tbody></table>

### breathiness_db_max

Maximum breathiness value in dB used for normalization to [-1, 1].

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-20.0</td>
</tbody></table>

### breathiness_db_min

Minimum breathiness value in dB used for normalization to [-1, 1].

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-96.0</td>
</tbody></table>

### breathiness_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on extracted breathiness curve.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.12</td>
</tbody></table>

### clip_grad_norm

The value at which to clip gradients. Equivalent to `gradient_clip_val` in `lightning.pytorch.Trainer`.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### dataloader_prefetch_factor

Number of batches loaded in advance by each `torch.utils.data.DataLoader` worker.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### dataset_size_key

The key that indexes the binarized metadata to be used as the `sizes` when batching by size

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>lengths</td>
</tbody></table>

### dictionary

Path to the word-phoneme mapping dictionary file. Training data must fully cover phonemes in the dictionary.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
</tbody></table>

### diff_accelerator

DDPM sampling acceleration method. The following methods are currently available:

- DDIM: the DDIM method from [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- PNDM: the PLMS method from [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778)
- DPM-Solver++ adapted from [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://github.com/LuChengTHU/dpm-solver)
- UniPC adapted from [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://github.com/wl-zhao/UniPC)

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>dpm-solver</td>
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
<tr><td align="center"><b>constraints</b></td><td>Must be a factor of <a href="#K_step">K_step</a>.</td>
</tbody></table>

### diffusion_type

The type of ODE-based generative model algorithm. The following models are currently available:

- Denoising Diffusion Probabilistic Models (DDPM) from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Rectified Flow from [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>reflow</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'ddpm', 'reflow'.</td>
</tbody></table>

### dilation_cycle_length

Length k of the cycle $2^0, 2^1 ...., 2^k$ of convolution dilation factors through WaveNet residual blocks.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>4</td>
</tbody></table>

### dropout

Dropout rate in some FastSpeech2 modules.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.1</td>
</tbody></table>

### ds_workers

Number of workers of `torch.utils.data.DataLoader`.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>4</td>
</tbody></table>

### dur_prediction_args

Arguments for phoneme duration prediction.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### dur_prediction_args.arch

Architecture of duration predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>fs2</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'fs2'.</td>
</tbody></table>

### dur_prediction_args.dropout

Dropout rate in duration predictor of FastSpeech2.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.1</td>
</tbody></table>

### dur_prediction_args.hidden_size

Dimensions of hidden layers in duration predictor of FastSpeech2.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>512</td>
</tbody></table>

### dur_prediction_args.kernel_size

Kernel size of convolution layers of duration predictor of FastSpeech2.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>3</td>
</tbody></table>

### dur_prediction_args.lambda_pdur_loss

Coefficient of single phone duration loss when calculating joint duration loss.

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
<tr><td align="center"><b>scope</b></td><td>training</td>
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

Size of TransformerFFNLayer convolution kernel size in FastSpeech2 encoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>9</td>
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

Maximum energy value in dB used for normalization to [-1, 1].

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-12.0</td>
</tbody></table>

### energy_db_min

Minimum energy value in dB used for normalization to [-1, 1].

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-96.0</td>
</tbody></table>

### energy_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on extracted energy curve.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.12</td>
</tbody></table>

### f0_max

Maximum base frequency (F0) in Hz for pitch extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1100</td>
</tbody></table>

### f0_min

Minimum base frequency (F0) in Hz for pitch extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>65</td>
</tbody></table>

### ffn_act

Activation function of TransformerFFNLayer in FastSpeech2 encoder:

- `torch.nn.ReLU` if 'relu'
- `torch.nn.GELU` if 'gelu'
- `torch.nn.SiLU` if 'swish'

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>gelu</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'relu', 'gelu', 'swish'.</td>
</tbody></table>

### fft_size

Fast Fourier Transforms parameter for mel extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>2048</td>
</tbody></table>

### finetune_enabled

Whether to finetune from a pretrained model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>False</td>
</tbody></table>

### finetune_ckpt_path

Path to the pretrained model for finetuning.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>null</td>
</tbody></table>

### finetune_ignored_params

Prefixes of parameter key names in the state dict of the pretrained model that need to be dropped before finetuning.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>list</td>
</tbody></table>

### finetune_strict_shapes

Whether to raise error if the tensor shapes of any parameter of the pretrained model and the target model mismatch. If set to `False`, parameters with mismatching shapes will be skipped.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>True</td>
</tbody></table>

### fmax

Maximum frequency of mel extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>16000</td>
</tbody></table>

### fmin

Minimum frequency of mel extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>40</td>
</tbody></table>

### freezing_enabled

Whether enabling parameter freezing during training.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>False</td>
</tbody></table>

### frozen_params

Parameter name prefixes to freeze during training.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>list</td>
<tr><td align="center"><b>default</b></td><td>[]</td>
</tbody></table>

### glide_embed_scale

The scale factor to be multiplied on the glide embedding values for melody encoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>11.313708498984760</td>
</tbody></table>

### glide_types

Type names of glide notes.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>list</td>
<tr><td align="center"><b>default</b></td><td>[up, down]</td>
</tbody></table>

### hidden_size

Dimension of hidden layers of FastSpeech2, token and parameter embeddings, and diffusion condition.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>256</td>
</tbody></table>

### hnsep

Harmonic-noise separation algorithm type.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>world</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'world', 'vr'.</td>
</tbody></table>

### hnsep_ckpt

Checkpoint or model path of NN-based harmonic-noise separator.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
</tbody></table>

### hop_size

Hop size or step length (in number of waveform samples) of mel and feature extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
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

Maximum number of DDPM steps used by shallow diffusion.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>400</td>
</tbody></table>

### K_step_infer

Number of DDPM steps used during shallow diffusion inference. Normally set as same as [K_step](#K_step).

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>400</td>
<tr><td align="center"><b>constraints</b></td><td>Should be no larger than K_step.</td>
</tbody></table>

### log_interval

Controls how often to log within training steps. Equivalent to `log_every_n_steps` in `lightning.pytorch.Trainer`.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>100</td>
</tbody></table>

### lr_scheduler_args

Arguments of learning rate scheduler. Keys will be used as keyword arguments of the `__init__()` method of [lr_scheduler_args.scheduler_cls](#lr_scheduler_argsscheduler_cls).

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### lr_scheduler_args.scheduler_cls

Learning rate scheduler class name.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>torch.optim.lr_scheduler.StepLR</td>
</tbody></table>

### main_loss_log_norm

Whether to use log-normalized weight for the main loss. This is similar to the method in the Stable Diffusion 3 paper [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
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
<tr><td align="center"><b>default</b></td><td>80000</td>
</tbody></table>

### max_batch_size

The maximum training batch size.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>48</td>
</tbody></table>

### max_beta

Max beta of the DDPM noise schedule.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.02</td>
</tbody></table>

### max_updates

Stop training after this number of steps. Equivalent to `max_steps` in `lightning.pytorch.Trainer`.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>320000</td>
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
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### mel_base

The logarithmic base of mel spectrogram calculation.

**WARNING: Since v2.4.0 release, this value is no longer configurable for preprocessing new datasets.**

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>e</td>
</tbody></table>

### mel_vmax

Maximum mel spectrogram heatmap value for TensorBoard plotting.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1.5</td>
</tbody></table>

### mel_vmin

Minimum mel spectrogram heatmap value for TensorBoard plotting.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-6.0</td>
</tbody></table>

### melody_encoder_args

Arguments for melody encoder. Available sub-keys: `hidden_size`, `enc_layers`, `enc_ffn_kernel_size`, `ffn_act`, `dropout`, `num_heads`, `use_pos_embed`, `rel_pos`. If either of the parameter does not exist in this configuration key, it inherits from the linguistic encoder.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### midi_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on the step function representing MIDI sequence for base pitch calculation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.06</td>
</tbody></table>

### nccl_p2p

Whether to enable P2P when using NCCL as the backend. Turn it to `false` if the training process is stuck upon beginning.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### num_ckpt_keep

Number of newest checkpoints kept during training.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>5</td>
</tbody></table>

### num_heads

The number of attention heads of `torch.nn.MultiheadAttention` in FastSpeech2 encoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>2</td>
</tbody></table>

### num_sanity_val_steps

Number of sanity validation steps at the beginning.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
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

Number of validation plots in each validation. Plots will be chosen from the start of the validation set.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>10</td>
</tbody></table>

### optimizer_args

Arguments of optimizer. Keys will be used as keyword arguments  of the `__init__()` method of [optimizer_args.optimizer_cls](#optimizer_argsoptimizer_cls).

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### optimizer_args.optimizer_cls

Optimizer class name

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>torch.optim.AdamW</td>
</tbody></table>

### pe

Pitch extraction algorithm type.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>parselmouth</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'parselmouth', 'rmvpe', 'harvest'.</td>
</tbody></table>

### pe_ckpt

Checkpoint or model path of NN-based pitch extractor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
</tbody></table>

### permanent_ckpt_interval

The interval (in number of training steps) of permanent checkpoints. Permanent checkpoints will not be removed even if they are not the newest ones.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>40000</td>
</tbody></table>

### permanent_ckpt_start

Checkpoints will be marked as permanent every [permanent_ckpt_interval](#permanent_ckpt_interval) training steps after this number of training steps.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>120000</td>
</tbody></table>

### pitch_prediction_args

Arguments for pitch prediction.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### pitch_prediction_args.dilation_cycle_length

Equivalent to [dilation_cycle_length](#dilation_cycle_length) but only for the pitch predictor model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>default</b></td><td>5</td>
</tbody></table>

### pitch_prediction_args.pitd_clip_max

Maximum clipping value (in semitones) of pitch delta between actual pitch and base pitch.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>12.0</td>
</tbody></table>

### pitch_prediction_args.pitd_clip_min

Minimum clipping value (in semitones) of pitch delta between actual pitch and base pitch.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-12.0</td>
</tbody></table>

### pitch_prediction_args.pitd_norm_max

Maximum pitch delta value in semitones used for normalization to [-1, 1].

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>8.0</td>
</tbody></table>

### pitch_prediction_args.pitd_norm_min

Minimum pitch delta value in semitones used for normalization to [-1, 1].

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
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

### pitch_prediction_args.residual_channels

Equivalent to [residual_channels](#residual_channels) but only for the pitch predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>default</b></td><td>256</td>
</tbody></table>

### pitch_prediction_args.residual_layers

Equivalent to [residual_layers](#residual_layers) but only for the pitch predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>default</b></td><td>20</td>
</tbody></table>

### pl_trainer_accelerator

Type of Lightning trainer hardware accelerator.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>auto</td>
<tr><td align="center"><b>constraints</b></td><td>See <a href="https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html?highlight=accelerator">Accelerator — PyTorch Lightning 2.X.X documentation</a> for available values.</td>
</tbody></table>

### pl_trainer_devices

To determine on which device(s) model should be trained.

'auto' will utilize all visible devices defined with the `CUDA_VISIBLE_DEVICES` environment variable, or utilize all available devices if that variable is not set. Otherwise, it behaves like `CUDA_VISIBLE_DEVICES` which can filter out visible devices.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>auto</td>
</tbody></table>

### pl_trainer_precision

The computation precision of training.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>16-mixed</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from '32-true', 'bf16-mixed', '16-mixed'. See more possible values at <a href="https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api">Trainer — PyTorch Lightning 2.X.X documentation</a>.</td>
</tbody></table>

### pl_trainer_num_nodes

Number of nodes in the training cluster of Lightning trainer.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>1</td>
</tbody></table>

### pl_trainer_strategy

Arguments of Lightning Strategy. Values will be used as keyword arguments when constructing the Strategy object.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### pl_trainer_strategy.name

Strategy name for the Lightning trainer.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
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
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### predict_voicing

Whether to enable voicing prediction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### raw_data_dir

Path(s) to the raw dataset including wave files, transcriptions, etc.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>str, List[str]</td>
</tbody></table>

### rel_pos

Whether to use relative positional encoding in FastSpeech2 module.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### residual_channels

Number of dilated convolution channels in residual blocks in WaveNet.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>512</td>
</tbody></table>

### residual_layers

Number of residual blocks in WaveNet.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>20</td>
</tbody></table>

### sampler_frame_count_grid

The batch sampler applies an algorithm called _sorting by similar length_ when collecting batches. Data samples are first grouped by their approximate lengths before they get shuffled within each group. Assume this value is set to $L_{grid}$, the approximate length of a data sample with length $L_{real}$ can be calculated through the following expression:

$$
L_{approx} = \lfloor\frac{L_{real}}{L_{grid}}\rfloor\cdot L_{grid}
$$

Training performance on some datasets may be very sensitive to this value. Change it to 1 (completely sorted by length without shuffling) to get the best performance in theory.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>6</td>
</tbody></table>

### sampling_algorithm

The algorithm to solve the ODE of Rectified Flow. The following methods are currently available:

- Euler: The Euler method.
- Runge-Kutta (order 2): The 2nd-order Runge-Kutta method.
- Runge-Kutta (order 4): The 4th-order Runge-Kutta method.
- Runge-Kutta (order 5): The 5th-order Runge-Kutta method.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>euler</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'euler', 'rk2', 'rk4', 'rk5'.</td>
</tbody></table>

### sampling_steps

The total sampling steps to solve the ODE of Rectified Flow. Note that this value may not equal to NFE (Number of Function Evaluations) because some methods may require more than one function evaluation per step.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>20</td>
</tbody></table>

### schedule_type

The DDPM schedule type.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>linear</td>
<tr><td align="center"><b>constraints</b></td><td>Choose from 'linear', 'cosine'.</td>
</tbody></table>

### shallow_diffusion_args

Arguments for shallow diffusion.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
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
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>type</b></td><td>dict</td>
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

Whether to forward and backward the auxiliary decoder during training. If set to `false`, the auxiliary decoder hangs in the memory and does not get any updates.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### shallow_diffusion_args.train_diffusion

Whether to forward and backward the diffusion (main) decoder during training. If set to `false`, the diffusion decoder hangs in the memory and does not get any updates.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### shallow_diffusion_args.val_gt_start

Whether to use the ground truth as `x_start` in the shallow diffusion validation process. If set to `true`, gaussian noise is added to the ground truth before shallow diffusion is performed; otherwise the noise is added to the output of the auxiliary decoder. This option is useful when the auxiliary decoder has not been trained yet.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
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

### speakers

The names of speakers in a multi-speaker model. Speaker names are mapped to speaker indexes and stored into spk_map.json when preprocessing.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>list</td>
</tbody></table>

### spk_ids

The IDs of speakers in a multi-speaker model. If an empty list is given, speaker IDs will be automatically generated as $0,1,2,...,N_{spk}-1$. IDs can be duplicate or discontinuous.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>List[int]</td>
<tr><td align="center"><b>default</b></td><td>[]</td>
</tbody></table>

### spec_min

Minimum mel spectrogram value used for normalization to [-1, 1]. Different mel bins can have different minimum values.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>List[float]</td>
<tr><td align="center"><b>default</b></td><td>[-5.0]</td>
</tbody></table>

### spec_max

Maximum mel spectrogram value used for normalization to [-1, 1]. Different mel bins can have different maximum values.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>List[float]</td>
<tr><td align="center"><b>default</b></td><td>[0.0]</td>
</tbody></table>

### T_start

The starting value of time $t$ in the Rectified Flow ODE which applies on $t \in (T_{start}, 1)$.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.4</td>
</tbody></table>

### T_start_infer

The starting value of time $t$ in the ODE during shallow Rectified Flow inference. Normally set as same as [T_start](#T_start).

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.4</td>
<tr><td align="center"><b>constraints</b></td><td>Should be no less than T_start.</td>
</tbody></table>

### task_cls

Task trainer class name.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>str</td>
</tbody></table>

### tension_logit_max

Maximum tension logit value used for normalization to [-1, 1]. Logit is the reverse function of Sigmoid:

$$
f(x) = \ln\frac{x}{1-x}
$$

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>10.0</td>
</tbody></table>

### tension_logit_min

Minimum tension logit value used for normalization to [-1, 1]. Logit is the reverse function of Sigmoid:

$$
f(x) = \ln\frac{x}{1-x}
$$

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-10.0</td>
</tbody></table>

### tension_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on extracted tension curve.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.12</td>
</tbody></table>

### test_prefixes

List of data item names or name prefixes for the validation set. For each string `s` in the list:

- If `s` equals to an actual item name, add that item to validation set.
- If `s` does not equal to any item names, add all items whose names start with `s` to validation set.

For multi-speaker combined datasets, "ds_id:name_prefix" can be used to apply the rules above within data from a specific sub-dataset, where ds_id represents the dataset index.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>required</td>
<tr><td align="center"><b>type</b></td><td>list</td>
</tbody></table>

### time_scale_factor

The scale factor that will be multiplied on the time $t$ of Rectified Flow before embedding into the model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>1000</td>
</tbody></table>

### timesteps

Total number of DDPM steps.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
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
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_energy_embed

Whether to accept and embed energy values into the model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_glide_embed

Whether to accept and embed glide types in melody encoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>false</td>
<tr><td align="center"><b>constraints</b></td><td>Only take affects when melody encoder is enabled.</td>
</tbody></table>

### use_key_shift_embed

Whether to embed key shifting values introduced by random pitch shifting augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>false</td>
<tr><td align="center"><b>constraints</b></td><td>Must be true if random pitch shifting is enabled.</td>
</tbody></table>

### use_melody_encoder

Whether to enable melody encoder for the pitch predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_pos_embed

Whether to use SinusoidalPositionalEmbedding in FastSpeech2 encoder.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn</td>
<tr><td align="center"><b>customizability</b></td><td>not recommended</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>true</td>
</tbody></table>

### use_shallow_diffusion

Whether to use shallow diffusion.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_speed_embed

Whether to embed speed values introduced by random time stretching augmentation.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>false</td>
<tr><td align="center"><b>constraints</b></td><td>Must be true if random time stretching is enabled.</td>
</tbody></table>

### use_spk_id

Whether embed the speaker id from a multi-speaker dataset.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>bool</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_tension_embed

Whether to accept and embed tension values into the model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### use_voicing_embed

Whether to accept and embed voicing values into the model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>nn, preprocessing, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>boolean</td>
<tr><td align="center"><b>default</b></td><td>false</td>
</tbody></table>

### val_check_interval

Interval (in number of training steps) between validation checks.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>all</td>
<tr><td align="center"><b>scope</b></td><td>training</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>2000</td>
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

Arguments for prediction of variance parameters other than pitch, like energy, breathiness, etc.

<table><tbody>
<tr><td align="center"><b>type</b></td><td>dict</td>
</tbody></table>

### variances_prediction_args.dilation_cycle_length

Equivalent to [dilation_cycle_length](#dilation_cycle_length) but only for the multi-variance predictor model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>default</b></td><td>4</td>
</tbody></table>

### variances_prediction_args.total_repeat_bins

Total number of repeating bins in the multi-variance predictor. Repeating bins are distributed evenly to each variance parameter.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>nn, inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>48</td>
</tbody></table>

### variances_prediction_args.residual_channels

Equivalent to [residual_channels](#residual_channels) but only for the multi-variance predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>default</b></td><td>192</td>
</tbody></table>

### variances_prediction_args.residual_layers

Equivalent to [residual_layers](#residual_layers) but only for the multi-variance predictor.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>default</b></td><td>10</td>
</tbody></table>

### vocoder

The vocoder class name.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>NsfHifiGAN</td>
</tbody></table>

### vocoder_ckpt

Path of the vocoder model.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing, training, inference</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>str</td>
<tr><td align="center"><b>default</b></td><td>checkpoints/nsf_hifigan/model</td>
</tbody></table>

### voicing_db_max

Maximum voicing value in dB used for normalization to [-1, 1].

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-20.0</td>
</tbody></table>

### voicing_db_min

Minimum voicing value in dB used for normalization to [-1, 1].

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>inference</td>
<tr><td align="center"><b>customizability</b></td><td>recommended</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>-96.0</td>
</tbody></table>

### voicing_smooth_width

Length of sinusoidal smoothing convolution kernel (in seconds) on extracted voicing curve.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>normal</td>
<tr><td align="center"><b>type</b></td><td>float</td>
<tr><td align="center"><b>default</b></td><td>0.12</td>
</tbody></table>

### win_size

Window size for mel or feature extraction.

<table><tbody>
<tr><td align="center"><b>visibility</b></td><td>acoustic, variance</td>
<tr><td align="center"><b>scope</b></td><td>preprocessing</td>
<tr><td align="center"><b>customizability</b></td><td>reserved</td>
<tr><td align="center"><b>type</b></td><td>int</td>
<tr><td align="center"><b>default</b></td><td>2048</td>
</tbody></table>
