# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)

## Configuration Schemas

This document explains the meaning and usages of all editable keys in a configuration file.

### Common configurations

#### base_config

Path(s) of other config files that the current config is based on and will override.

##### used by

all

##### type

str, List[str]

### Vocoder Related

#### vocoder:

Model type for reconstruction .wav from mel information.

##### used by

all

##### type

str

#### vocoder_ckpt:

Pat of vocoder model for reconstruction .wav from mel information.

##### used by

all

##### type

str

#### audio_sample_rate

Sample rate of audio files

##### used by

all

##### type

int

##### default

44100

#### audio_num_mel_bins

Mel spectrum resolution for wavfile reconstruction.

##### used by

all

##### type

int

##### default

_128_

#### hop_size

Defines time frame width (in number of sample points) during mel extraction/waveform reconstruction.

##### used by

all

##### type

int

##### default

_512_

#### fft_size

Defines n_fft (as fft parameter) during mel extraction from ground-truth wavefile.

##### used by

all

##### type

int

##### default

_2048_

#### win_size

Defines window size (as fft parameter) during mel extraction from ground-truth wavefile.

##### used by

all

##### type

int

##### default

_2048_

#### fmin

Lowest f0 in mel bins.

##### used by

all

##### type

int

##### default

_40_

#### fmax

Highest f0 in mel bins.

##### used by

all

##### type

int

##### default

_16000_

### Neural networks: Diffusion

#### K_step

TBD

#### timesteps

TBD

#### max_beta

TBD

#### rel_pos

TBD

#### pndm_speedup

TBD


### Neural networks: Backbone

#### hidden_size

Dimension of hidden layers of FastSpeech2, token and variance embeddings, and diffusion condition.

##### used by

acoustic model

##### type

int

##### default

_256_

#### residual_channels

Dimension of residual block output of Conv1D layers in WaveNet.

##### used by

acoustic model

##### type

int

##### default

_512_

#### residual_layers

Number of residual blocks in WaveNet.

##### used by

acoustic model

##### type

int

##### default

_20_

#### diff_decoder_type

Denoiser type of the DDPM.

##### used by

acoustic model

##### type

str

##### default

_wavenet_

##### Constraints

choose from [ _wavenet_ ]

#### diff_loss_type

Loss type of the DDPM.

##### used by

acoustic model

##### type

str

##### default

_l2_

##### Constraints

choose from [ _l1_, _l2_ ]


### Dataset information and preprocessing

#### binarization_args shuffle

Whether binarized dataset is shuffled or not.

##### used by

all

##### type

boolean

##### default

true

#### augmentation_args random_pitch_shifting range

Data will be augmented by random pitch shifting, shifted data will be seen as the original speaker.

When range is set to [-M,N], data's f0 will be shifted in th range of f0 - M semitones to f0 + N semitones.

##### used by

all

##### type

list[float]

##### default

[_-5_,_5_]

#### augmentation_args random_pitch_shifting scale

Data will be augmented by random pitch shifting, shifted data will be seen as the original speaker.

When scale is set to s, extra s times of orignal dataset total length will be added to the total dataset after augmentation. 

##### used by

all

##### type

float

##### default

_1.0_

#### augmentation_args fixed_pitch_shifting targets

Data will be augmented by fixed pitch shifting, shifted data will NOT be seen as the original speaker.

For each target t in the list, data's f0 of certain speaker will be shifted to f0 + t semitones as a  different speaker.

##### used by

all

##### type

list[float]

##### default

[_-5_,_5_]

#### augmentation_args fixed_pitch_shifting scale

Data will be augmented by fixed pitch shifting, shifted data will NOT be seen as the original speaker.

When scale is set to s, extra s times of orignal dataset total length will be added to the total dataset after augmentation. 

##### used by

all

##### type

float

##### default

_0.75_

#### augmentation_args random_time_stretching range

Data will be augmented by random time stretching, shifted data will be seen as the original speaker.

Random values will be sampled from the range [M, N]. Lower values means speeding up, higher values means slowing down.

##### used by

all

##### type

list[float]

##### default

[_0.5_,_2_]

#### augmentation_args random_time_stretching domain

Data will be augmented by random time stretching, shifted data will be seen as the original speaker.

Random values will be sampled from the range:

- If 'linear', stretching ratio will be uniformly drawn from [M,N].
- If 'log', x will be uniformly drawn from [log(M),log(N)] then stretching ratio will be set as $\text{e}^{\text{x}}$
##### used by

all

##### type

str

##### default

log

##### constraint

Choose from [_'log'_,_'linear'_]


#### augmentation_args random_time_stretching scale

Data will be augmented by fixed pitch shifting, shifted data will NOT be seen as the original speaker.

When scale is set to s, extra s times of orignal dataset total length will be added to the total dataset after augmentation. 

##### used by

all

##### type

float

##### default

0.75

#### raw_data_dir

Path(s) to the raw data including wave files, transcriptions, etc.

##### used by

all

##### type

str, List[str]

#### binary_data_dir

TBD

#### binarizer_cls

TBD

#### dictionary

TBD

#### spec_min

TBD

#### spec_max

TBD

#### keep_bins

TBD

#### mel_vmin

TBD

#### mel_vmax

TBD

#### interp_uv

TBD

#### save_f0

TBD

#### use_spk_id

TBD

#### f0_embed_type

TBD

#### use_key_shift_embed

TBD

#### use_speed_embed

TBD



### Training, validation and inference

#### task_cls

TBD

#### lr

Initial learning rate of the scheduler.

##### used by

all

##### type

float

##### default

_0.0004_

#### lr_decay_steps

TBD

#### lr_decay_gamma

TBD

#### max_batch_frames

$\text{1 time frame} =\frac{\text{hop size}}{\text{Wav file sample rate}} \text{ second(s)}$.

Number of time frames in per card batch is not allowed to exceed max_batch_frames to get rid of OOM. max_batch_size will be automatically reduced to meet this requirement.

##### used by

all

##### type

int

##### default

_80000_

#### max_batch_size

Number of .wav slices in each data batch sent from dataloader to each single GPU.

This number will be kept the same unless max_batch_frames requirement is not met. In this case, per card batch size will be automatically reduced to a feasible size to get rid of OOM.

##### used by

all

##### type

int

##### default

_48_

#### val_with_vocoder

TBD

#### val_check_interval

TBD

#### num_valid_plots

TBD

#### max_updates

TBD

#### permanent_ckpt_start

TBD

#### permanent_ckpt_interval

TBD

### Distributed Training Setup

#### pl_trainer_accelerator

TBD

#### pl_trainer_devices

TBD

#### pl_trainer_precision

TBD

#### pl_trainer_num_nodes

TBD

#### pl_trainer_strategy

TBD

#### ddp_backend

TBD

