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

### Neural networks

#### hidden_size

Dimension of hidden layers of FastSpeech2, token and variance embeddings, and diffusion condition.

##### used by

acoustic model

##### type

int

##### default

_256_

#### residual_channels

Dimension of residual block Conv1D layers in WaveNet.

##### used by

acoustic model

##### type

int

##### default

_512_

(_384_ is recommended if you GPU has <=6GB memory) 

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

#### raw_data_dir

Path(s) to the raw data including wave files, transcriptions, etc.

##### used by

all

##### type

str, List[str]

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

#### max_batch_size

Number of .wav slices in each data batch sent from dataloader to each single GPU.

This number will be kept the same unless max_batch_frames requirement is not met. In this case, per card batch size will be automatically reduced to a feasible size to get rid of OOM.

##### used by

all

##### type

int

##### default

_48_

#### max_batch_frames

$\text{1 time frame} =\frac{\text{hop size}}{\text{Wav file sample rate}} \text{ second(s)}$.

Number of time frames in per card batch is not allowed to exceed max_batch_frames to get rid of OOM. max_batch_size will be automatically reduced to meet this requirement.

##### used by

all

##### type

int

##### default

_80000_



