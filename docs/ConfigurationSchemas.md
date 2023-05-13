# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
| [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)

## Configuration Schemas

This document explains the meaning and usages of all editable keys in a configuration file.

Each configuration key (including nested keys) are described with a brief explanation and several attributes listed as
follows:

|    Attribute    | Explanation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:---------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     used by     | Represents what kind(s) of models and tasks this configuration belongs to.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|      scope      | The scope of effects of the configuration, indicating what it can influence within the whole pipeline. Possible values are:<br>**nn** - This configuration is related to how the neural networks are formed and initialized. Modifying it will result in failure when loading or resuming from checkpoints.<br>**preprocessing** - This configuration controls how raw data pieces or inference inputs are converted to inputs of neural networks. Binarizers should be re-run if this configuration is modified.<br>**training** - This configuration describes the training procedures. Most training configurations can affect training performance, memory consumption, device utilization and loss calculation. Modifying training-only configurations will not cause severe inconsistency or errors in most situations.<br>**inference** - This configuration describes the calculation logic through the model graph. Changing it can lead to inconsistent or wrong outputs of inference or validation.<br>**others** - Other configurations not discussed above. Will have different effects according to  the descriptions.                                                          |
| customizability | The level of customizability of the configuration. Possible values are:<br>**required** - This configuration **must** be set or modified according to the actual situation or condition, otherwise errors can be raised.<br>**recommended** - It is recommended to adjust this configuration according to the dataset, requirements, environment and hardware. Most functionality-related and feature-related configurations are at this level, and all configurations in this level are widely tested with different values. However, leaving it unchanged will not cause problems.<br>**normal** - There is no need to modify it as the default value is carefully tuned and widely validated. However, one can still use another value if there are some special requirements or situations.<br>**not recommended** - No other values except the default one of this configuration are tested. Modifying it will not cause errors, but may cause unpredictable or significant impacts to the pipelines.<br>**preserved** - This configuration **must not** be modified. It appears in the configuration file only for future scalability, and currently changing it will result in errors. |
|      type       | Value type of the configuration. Follows the syntax of Python type hints.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|   constraints   | Value constraints of the configuration.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|     default     | Default value of the configuration. Uses YAML value syntax.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

### accumulate_grad_batches

Indicates that gradients of how many training steps are accumulated before each `optimizer.step()` call. 1 means no gradient accumulation.

#### used by

all

#### scope

training

#### customizability

recommended

#### type

int

#### default

1

### audio_num_mel_bins

Number of mel channels for feature extraction, diffusion sampling and waveform reconstruction.

#### used by

acoustic

#### scope

nn, preprocessing, inference

#### customizability

preserved

#### type

int

#### default

128

### audio_sample_rate

Sampling rate of waveforms.

#### used by

all

#### scope

preprocessing

#### customizability

preserved

#### type

int

#### default

44100

### augmentation_args

Arguments for data augmentation.

#### type

dict

### augmentation_args.fixed_pitch_shifting

Arguments for fixed pitch shifting augmentation.

#### type

dict

### augmentation_args.fixed_pitch_shifting.scale

Scale ratio of each target in fixed pitch shifting augmentation.

#### used by

acoustic

#### scope

preprocessing

#### customizability

recommended

#### type

tuple

#### default

0.75

### augmentation_args.fixed_pitch_shifting.targets

Targets (in semitones) of fixed pitch shifting augmentation.

#### used by

acoustic

#### scope

preprocessing

#### customizability

not recommended

#### type

tuple

#### default

[-5.0, 5.0]

### augmentation_args.random_pitch_shifting

all

#### scope

preprocessing

#### customizability

recommended

#### type

float

#### default

0.75

### augmentation_args.fixed_pitch_shifting.targets

Targets of fixed pitch shifting augmentation, each in semitones.

#### used by

all

#### scope

preprocessing

#### customizability

not recommended

#### type

list

#### default

[-5, 5]

### augmentation_args.random_pitch_shifting

Arguments for random pitch shifting augmentation.

#### type

dict

### augmentation_args.random_pitch_shifting.range

Range of the random pitch shifting ( in semitones).

#### used by

acoustic

#### scope

preprocessing

#### customizability

not recommended

#### type

tuple

#### default

[-5.0, 5.0]

### augmentation_args.random_pitch_shifting.scale

Scale ratio of the random pitch shifting augmentation.

#### used by

acoustic

#### scope

preprocessing

#### customizability

recommended

#### type

float

#### default

1.0

### augmentation_args.random_time_stretching.domain

The domain where random time stretching factors are uniformly distributed in.

- If 'linear', stretching ratio $x$ will be uniformly distributed in $[V_{min}, V_{max}]$.
- If 'log', $\ln{x}$ will be uniformly distributed in $[\ln{V_{min}}, \ln{V_{max}}]$.

#### used by

acoustic

#### scope

preprocessing

#### customizability

not recommended

#### type

str

#### default

log

#### constraint

Choose from 'log', 'linear'.

### augmentation_args.random_time_stretching.range

Range of random time stretching factors.

#### used by

acoustic

#### scope

preprocessing

#### customizability

not recommended

#### type

tuple

#### default

[0.5, 2]

### augmentation_args.random_time_stretching.scale

Scale ratio of random time stretching augmentation.

#### used by

acoustic

#### scope

preprocessing

#### customizability

recommended

#### type

float

#### default

0.75

### base_config

Path(s) of other config files that the current config is based on and will override.

#### scope

others

#### type

Union[str, list]

### binarization_args

Arguments for binarizers.

#### type

dict

### binarization_args.num_workers

Number of worker subprocesses when running binarizers. More workers can speed up the preprocessing but will consume more memory.

#### used by

all

#### scope

preprocessing

#### customizability

recommended

#### type

int

#### default

1

### binarization_args.shuffle

Whether binarized dataset will be shuffled or not.

#### used by

all

#### scope

preprocessing

#### customizability

normal

#### type

bool

#### default

true

### binarizer_cls

Binarizer class name.

#### used by

all

#### scope

preprocessing

#### customizability

preserved

#### type

str

### binary_data_dir

Path to the binarized dataset.

#### used by

all

#### scope

preprocessing, training

#### customizability

required

#### type

str

### clip_grad_norm

The value at which to clip gradients. Equivalent to `gradient_clip_val` in `lightning.pytorch.Trainer`.

#### used by

all

#### scope

training

#### customizability

not recommended

#### type

float

#### default

1

### dataloader_prefetch_factor

Number of batches loaded in advance by each `torch.utils.data.DataLoader` worker.

#### used by

all

#### scope

training

#### customizability

normal

#### type

bool

#### default

true

### ddp_backend

The distributed training backend.

#### used by

all

#### scope

training

#### customizability

normal

#### type

str

#### default

nccl

#### constraints

Choose from 'gloo', 'nccl', 'nccl_no_p2p'. Windows platforms may use 'gloo'; Linux platforms may use 'nccl'; if Linux ddp gets stuck, use 'nccl_no_p2p'.

### dictionary

path to the word-phoneme mapping dictionary file. Training data must fully cover phonemes in the dictionary.

#### used by

acoustic

#### scope

preprocessing

#### customizability

normal

#### type

str

### diff_decoder_type

Denoiser type of the DDPM.

#### used by

acoustic

#### scope

nn

#### customizability

preserved

#### type

str

#### default

wavenet

### diff_loss_type

Loss type of the DDPM.

#### used by

acoustic

#### scope

training

#### customizability

not recommended

#### type

str

#### default

l2

#### constraints

Choose from 'l1', 'l2'.

### dilation_cycle_length

Length k of the cycle $2^0, 2^1 ...., 2^k$ of convolution dilation factors through WaveNet residual blocks.

#### used by

acoustic

#### scope

nn

#### customizability

not recommended

#### type

int

#### default

4

### dropout

Dropout rate in some FastSpeech2 modules.

#### used by

all

#### scope

nn

#### customizability

not recommended

#### type

float

#### default

0.1

### ds_workers

Number of workers of `torch.utils.data.DataLoader`.

#### used by

all

#### scope

training

#### customizability

normal

#### type

int

#### default

4

### enc_ffn_kernel_size

Size of TransformerFFNLayer convolution kernel size in FastSpeech2 encoder.

#### used by

all

#### scope

nn

#### customizability

not recommended

#### type

int

#### default

9

### enc_layers

Number of FastSpeech2 encoder layers.

#### used by

all

#### scope

nn

#### customizability

normal

#### type

int

#### default

4

### f0_embed_type

Map f0 to embedding using:

- `torch.nn.Linear` if 'continuous'
- `torch.nn.Embedding` if 'discrete'

#### used by

acoustic

#### scope

nn

#### customizability

normal

#### type

str

#### default

continuous

#### constraints

Choose from 'continuous', 'discrete'.

### ffn_act

Activation function of TransformerFFNLayer in FastSpeech2 encoder:

- `torch.nn.ReLU` if 'relu'
- `torch.nn.GELU` if 'gelu'
- `torch.nn.SiLU` if 'swish'

#### used by

all

#### scope

nn

#### customizability

not recommended

#### type

str

#### default

gelu

#### constraints

Choose from 'relu', 'gelu', 'swish'.

### ffn_padding

Padding mode of TransformerFFNLayer convolution in FastSpeech2 encoder.

#### used by

all

#### scope

nn

#### customizability

not recommended

#### type

str

#### default

SAME

### fft_size

Fast Fourier Transforms parameter for mel extraction.

#### used by

all

#### scope

preprocessing

#### customizability

preserved

#### type

int

#### default

2048

### fmax

Maximum frequency of mel extraction.

#### used by

acoustic

#### scope

preprocessing

#### customizability

preserved

#### type

int

#### default

16000

### fmin

Minimum frequency of mel extraction.

#### used by

acoustic

#### scope

preprocessing

#### customizability

preserved

#### type

int

#### default

40

### hidden_size

Dimension of hidden layers of FastSpeech2, token and variance embeddings, and diffusion condition.

#### used by

acoustic

#### scope

nn

#### customizability

normal

#### type

int

#### default

256

### hop_size

Hop size or step length (in number of waveform samples) of mel and feature extraction.

#### used by

acoustic

#### scope

preprocessing

#### customizability

preserved

#### type

int

#### default

512

### interp_uv

Whether to apply linear interpolation to unvoiced parts in f0.

#### used by

acoustic

#### scope

preprocessing

#### customizability

preserved

#### type

boolean

#### default

true

### K_step

Total number of diffusion steps.

#### used by

all

#### scope

nn

#### customizability

not recommended

#### type

int

#### default

1000

### log_interval

Controls how often to log within training steps. Equivalent to `log_every_n_steps` in `lightning.pytorch.Trainer`.

#### used by

all

#### scope

training

#### customizability

normal

#### type

int

#### default

100

### lr_scheduler_args.gamma

Learning rate decay ratio of `torch.optim.lr_scheduler.StepLR`.

#### used by

all

#### scope

training

#### customizability

recommended

#### type

float

#### default

0.5

### lr_scheduler_args

Arguments of learning rate scheduler. Keys will be used as keyword arguments when initializing the scheduler object.

#### type

dict

### lr_scheduler_args.scheduler_cls

Learning rate scheduler class name.

#### used by

all

#### scope

training

#### customizability

not recommended

#### type

str

#### default

torch.optim.lr_scheduler.StepLR

### lr_scheduler_args.step_size

Learning rate decays every this number of training steps.

#### used by

all

#### scope

training

#### customizability

recommended

#### type

int

#### default

50000

### lr_scheduler_args.warmup_steps

Number of warmup steps of the learning rate scheduler.

#### used by

all

#### scope

training

#### customizability

normal

#### type

int

#### default

2000

### max_batch_frames

Maximum number of data frames in each training batch. Used to dynamically control the batch size.

#### used by

all

#### scope

training

#### customizability

recommended

#### type

int

#### default

80000

### max_batch_size

The maximum training batch size.

#### used by

all

#### scope

training

#### customizability

recommended

#### type

int

#### default

48

### max_beta

Max beta of the DDPM noise schedule.

#### used by

all

#### scope

nn, inference

#### customizability

normal

#### type

float

#### default

0.02

### max_updates

Stop training after this number of steps. Equivalent to `max_steps` in `lightning.pytorch.Trainer`.

#### used by

all

#### scope

training

#### customizability

recommended

#### type

int

#### default

320000

### max_val_batch_frames

Maximum number of data frames in each validation batch.

#### used by

all

#### scope

training

#### customizability

preserved

#### type

int

#### default

60000

### max_val_batch_size

The maximum validation batch size.

#### used by

all

#### scope

training

#### customizability

preserved

#### type

int

#### default

1

### mel_vmax

Maximum mel spectrogram heatmap value for TensorBoard plotting.

#### used by

all

#### scope

training

#### customizability

not recommended

#### type

float

#### default

1.5

### mel_vmin

Minimum mel spectrogram heatmap value for TensorBoard plotting.

#### used by

all

#### scope

training

#### customizability

not recommended

#### type

float

#### default

-6.0

### num_ckpt_keep

Number of newest checkpoints kept during training.

#### used by

all

#### scope

training

#### customizability

normal

#### type

int

#### default

5

### num_heads

The number of attention heads of `torch.nn.MultiheadAttention` in FastSpeech2 encoder.

#### used by

all

#### scope

nn

#### customizability

not recommended

#### type

int

#### default

2

### num_sanity_val_steps

Number of sanity validation steps at the beginning.

#### used by

all

#### scope

training

#### customizability

preserved

#### type

int

#### default

5

### num_spk

Maximum number of speakers in multi-speaker models.

#### used by

acoustic

#### scope

nn

#### customizability

required

#### type

int

#### default

1

### num_valid_plots

Number of validation plots in each validation. Plots will be chosen from the start of the validation set.

#### used by

acoustic

#### scope

training

#### customizability

recommended

#### type

int

#### default

10

### optimizer_args

Arguments of optimizer. Keys will be used as keyword arguments when initializing the optimizer object.

#### type

dict

### optimizer_args.beta1

Parameter of the `torch.optim.AdamW` optimizer.

#### used by

all

#### scope

training

#### customizability

not recommended

#### type

float

#### default

0.9

### optimizer_args.beta2

Parameter of the `torch.optim.AdamW` optimizer.

#### used by

all

#### scope

training

#### customizability

not recommended

#### type

float

#### default

0.98

### optimizer_args.lr

Initial learning rate of the optimizer.

#### used by

all

#### scope

training

#### customizability

recommended

#### type

float

#### default

0.0004

### optimizer_args.optimizer_cls

Optimizer class name

#### used by

all

#### scope

training

#### customizability

preserved

#### type

str

#### default

torch.optim.AdamW

### optimizer_args.weight_decay

Weight decay ratio of optimizer.

#### used by

all

#### scope

training

#### customizability

not recommended

#### type

float

#### default

0

### permanent_ckpt_interval

The interval (in number of training steps) of permanent checkpoints. Permanent checkpoints will not be removed even if they are not the newest ones.

#### used by

all

#### scope

training

#### type

int

#### default

40000

### permanent_ckpt_start

Checkpoints will be marked as permanent every [permanent_ckpt_interval](#permanent_ckpt_interval) training steps after this number of training steps.

#### used by

all

#### scope

training

#### type

int

#### default

120000

### pl_trainer_accelerator

Type of Lightning trainer hardware accelerator.

#### used by

all

#### scope

training

#### customization

not recommended

#### type

str

#### default

auto

#### constraints

See [Accelerator — PyTorch Lightning 2.X.X documentation](https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html?highlight=accelerator) for available values.

### pl_trainer_devices

To determine on which device(s) model should be trained.

'auto' will utilize all visible devices defined with the `CUDA_VISIBLE_DEVICES` environment variable, or utilize all available devices if that variable is not set. Otherwise, it behaves like `CUDA_VISIBLE_DEVICES` which can filter out visible devices.

#### used by

all

#### scope

training

#### customization

not recommended

#### type

str

#### default

auto

### pl_trainer_precision

The computation precision of training.

#### used by

all

#### scope

training

#### customization

normal

#### type

str

#### default

32-true

#### constraints

Choose from '32-true', 'bf16-mixed', '16-mixed', 'bf16', '16'. See more possible values at [Trainer — PyTorch Lightning 2.X.X documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api).

### pl_trainer_num_nodes

Number of nodes in the training cluster of Lightning trainer.

#### used by

all

#### scope

training

#### customization

preserved

#### type

int

#### default

1

### pl_trainer_strategy

Strategies of the Lightning trainer behavior.

#### used by

all

#### scope

training

#### customization

preserved

#### type

str

#### default

auto

### pndm_speedup

Diffusion sampling speed-up ratio. 1 means no speeding up.

#### used by

all

#### type

int

#### default

10

#### constraints

Must be a factor of [K_step](#K_step).

### raw_data_dir

Path(s) to the raw dataset including wave files, transcriptions, etc.

#### used by

all

#### scope

preprocessing

#### customizability

required

#### type

str, List[str]

### rel_pos

Whether to use relative positional encoding in FastSpeech2 module.

#### used by

all

#### scope

nn

#### customizability

not recommended

#### type

boolean

#### default

true

### residual_channels

Number of dilated convolution channels in residual blocks in WaveNet.

#### used by

acoustic

#### scope

nn

#### customizability

normal

#### type

int

#### default

512

### residual_layers

Number of residual blocks in WaveNet.

#### used by

acoustic

#### scope

nn

#### customizability

normal

#### type

int

#### default

20

### sampler_frame_count_grid

The batch sampler applies an algorithm called _sorting by similar length_ when collecting batches. Data samples are first grouped by their approximate lengths before they get shuffled within each group. Assume this value is set to $L_{grid}$, the approximate length of a data sample with length $L_{real}$ can be calculated through the following expression:

$$
L_{approx} = \lfloor\frac{L_{real}}{L_{grid}}\rfloor\cdot L_{grid}
$$

Training performance on some datasets may be very sensitive to this value. Change it to 1 (completely sorted by length without shuffling) to get the best performance in theory.

#### used by

all

#### scope

training

#### customizability

normal

#### type

int

#### default

6

### save_codes

Files in these folders will be backed up every time a training starts.

#### used by

all

#### scope

training

#### customizability

normal

#### type

list

#### default

[configs, modules, training, utils]

### schedule_type

The diffusion schedule type.

#### used by

acoustic

#### scope

nn

#### customizability

not recommended

#### type

str

#### default

linear

#### constraints

Choose from 'linear', 'cosine'.

### seed

The global random seed used to shuffle data, initializing model weights, etc.

#### used by

all

#### scope

preprocessing, training

#### customizability

normal

#### type

int

#### default

1234

### sort_by_len

Whether to apply the _sorting by similar length_ algorithm described in [sampler_frame_count_grid](#sampler_frame_count_grid). Turning off this option may slow down training because sorting by length can better utilize the computing resources.

#### used by

all

#### scope

training

#### customizability

not recommended

#### type

bool

#### default

true

### speakers

The names of speakers in a multi-speaker model. Speaker names are mapped to speaker indexes and stored into spk_map.json when preprocessing.

#### used by

acoustic

#### scope

preprocessing

#### customizability

required

#### type

list

### spec_min

Minimum mel spectrogram value used for normalization to [-1, 1]. Different mel bins can have different minimum values.

#### used by

all

#### scope

inference

#### customizability

not recommended

#### type

List[float]

#### default

[-5.0]

### spec_max

Maximum mel spectrogram value used for normalization to [-1, 1]. Different mel bins can have different maximum values.

#### used by

all

#### scope

inference

#### customizability

not recommended

#### type

List[float]

#### default

[0.0]

### task_cls

Task trainer class name.

#### used by

all

#### scope

training

#### customizability

preserved

#### type

str

### test_prefixes

List of data item names or name prefixes for the validation set. For each string `s` in the list:

- If `s` equals to an actual item name, add that item to validation set.
- If `s` does not equal to any item names, add all items whose names start with `s` to validation set.

For multi-speaker datasets, "spk_id:name_prefix" can be used to apply the rules above within data from a specific speaker, where spk_id represents the speaker index.

#### used by

all

#### scope

preprocessing

#### customizability

required

#### type

list

### timesteps

Equivalent to [K_step](#K_step).

### train_set_name

Name of the training set used in binary filenames, TensorBoard keys, etc.

#### used by

all

#### scope

preprocessing, training

#### customizability

preserved

#### type

str

#### default

train

### use_key_shift_embed

Whether to embed key shifting values introduced by random pitch shifting augmentation.

#### used by

acoustic

#### scope

nn, preprocessing, inference

#### customizability

recommended

#### type

boolean

#### default

false

#### constraints

Must be true if random pitch shifting is enabled.

### use_pos_embed

Whether to use SinusoidalPositionalEmbedding in FastSpeech2 encoder.

#### used by

acoustic

#### scope

nn

#### customizability

not recommended

#### type

boolean

#### default

true

### use_speed_embed

Whether to embed speed values introduced by random time stretching augmentation.

#### used by

acoustic

#### type

boolean

#### default

false

#### constraints

Must be true if random time stretching is enabled.

### use_spk_id

Whether embed the speaker id from a multi-speaker dataset.

#### used by

acoustic

#### scope

nn, preprocessing, inference

#### customizability

recommended

#### type

bool

#### default

false

### val_check_interval

Interval (in number of training steps) between validation checks.

#### used by

all

#### scope

training

#### customizability

recommended

#### type

int

#### default

2000

### val_with_vocoder

Whether to load and use the vocoder to generate audio during validation. Validation audio will not be available if this option is disabled.

#### used by

acoustic

#### scope

training

#### customizability

normal

#### type

bool

#### default

true

### valid_set_name

Name of the validation set used in binary filenames, TensorBoard keys, etc.

#### used by

all

#### scope

preprocessing, training

#### customizability

preserved

#### type

str

#### default

valid

### vocoder

The vocoder class name.

#### used by

acoustic

#### scope

preprocessing, training, inference

#### customizability

normal

#### type

str

#### default

NsfHifiGAN

### vocoder_ckpt

Path of the vocoder model.

#### used by

acoustic

#### scope

preprocessing, training, inference

#### customizability

normal

#### type

str

#### default

checkpoints/nsf_hifigan/model

### win_size

Window size for mel or feature extraction.

#### used by

all

#### scope

preprocessing

#### customizability

preserved

#### type

int

#### default

2048

