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

Diffusion steps for generating mel spectrogram.

##### used by

all

##### type

int

##### default

_1000_

#### timesteps

Same as K_step.

#### max_beta

Max beta for the discrete-time DPM. (See the original DDPM paper for details)

##### used by

all

##### type

float

##### default

_0.02_


#### rel_pos

Relative positional encoding in FastSpeech2 module.

##### used by

all

##### type

boolean

##### default

_true_

#### pndm_speedup

PNDM speeding up ratio. 1 means no speeding up.

Read https://openreview.net/forum?id=PlKWVd2yBkY: Pseudo Numerical Methods for Diffusion Models on Manifolds.

##### used by

all

##### type

int

##### default

_10_

##### constraints

choose from [_1 , 10 , 20, 50, 100_]


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

#### dilation_cycle_length

Number k of different dilation width parameters $2^0, 2^1 ...., 2^k$

##### used by

all

##### type

int

##### default

4

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

Path(s) to the binarized data in .npy format .

##### used by

all

##### type

str, List[str]

#### binarizer_cls

Binarizer class to specify binarized data structure.

##### used by

all

##### type

class

##### default

_preprocessing.acoustic_binarizer.AcousticBinarizer_

#### dictionary

Dictionary for training acoustic model. Training data must fully coverage phonemes in the dictionary.

##### used by

all

##### type

str

##### default

_dictionaries/opencpop-extension.txt_

#### spec_min

Mel Spectrogram value mapped to -1 in Gaussian Diffusion normalization/denormalization step.

##### used by

all

##### type

list[int]

##### default

_[-5]_


#### spec_max

Mel Spectrogram value mapped to 1 in Gaussian Diffusion normalization/denormalization step.

##### used by

all

##### type

list[int]

##### default

_[0]_

#### keep_bins

Number of mel bins.

##### used by

all

##### type

int

##### default

128

#### mel_vmin

Minimum mel spectrogram value for visualizing heatmap.

##### used by

all

##### type

float

##### default

_-6_


#### mel_vmax

Maximum mel spectrogram value for visualizing heatmap.

##### used by

all

##### type

float

##### default

_1.5_

#### interp_uv

Unvoiced F0 interpolation.

##### used by

all

##### type

boolean

##### default

true

#### use_spk_id

True if training a multi-speaker model, or single speaker data augmented as different speakers during training.

##### used by

acoustic 

##### type

boolean

##### default

true

#### f0_embed_type

Map f0 to embedding using :

- torch.nn.Linear if continuous
- torch.nn.Embedding if discrete

##### used by

acoustic 

##### type

str

##### default

'continuous'

#### use_key_shift_embed

Whether to use embedding information when data is augmented by key shifting.

##### used by

acoustic 

##### type

boolean

##### default

true

#### use_speed_embed

Whether to use embedding information when data is augmented by time stretching.

##### used by

acoustic 

##### type

boolean

##### default

true


### Training, validation and inference

#### task_cls

Class of model training.

##### used by

acoustic

##### type

class

##### default

_training.acoustic_task.AcousticTask_

#### lr

Initial learning rate of the scheduler.

##### used by

acoustic

##### type

float

##### default

_0.0004_

#### lr_decay_steps

Learning rate will be reduced by certain ratio repeatedly after each lr_decay_steps. 

##### used by

acoustic

##### type

int

##### default

_50000_

#### lr_decay_gamma

Learning rate will be reduced by lr_decay_gamma * previous_lr repeatedly after certain steps. 

##### used by

acoustic

##### type

float

##### default

_0.5_

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

Whether using vocoder to generate .wav during each validation step.

##### used by

acoustic

##### type

boolean

##### default

_true_


#### val_check_interval

Validating data repeatedly after each val_check_interval steps.

##### used by

acoustic

##### type

int

##### default

_2000_

#### num_valid_plots

Number of validation plots in each validation.

##### used by

acoustic

##### type

int

##### default

_10_

#### max_updates

Total training steps ( in each step, gradient will be accumulated and updated to weights ). 

##### used by

acoustic

##### type

int

##### default

_320000_

#### permanent_ckpt_start

Checkpoints will be saved and kept not deleted from permanent_ckpt_start training steps.

##### used by

acoustic

##### type

int

##### default

_120000_

#### permanent_ckpt_interval

After permanent_ckpt_start training steps, checkpoints after each permanent_ckpt_interval steps will be repeatedly saved and kept not deleted.

##### used by

acoustic

##### type

int

##### default

_40000_

### Distributed Training Setup

Pytorch lightning is utilized for distributed training.

Read https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api for possible values

#### pl_trainer_accelerator

To determine on which kind of device(s) (CPU, GPU, customized XPU, etc. ) model should be trained.

Read https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html?highlight=accelerator for accelerator customization.

##### used by

all

##### type

str

##### default

'auto'

#### pl_trainer_devices

To determine on which device(s) model should be trained.

'auto' will utilize all possible devices, while '0,1,...' specifies model training on certain device(s).

( This is exactly the same as specifying CUDA_VISIBLE_DEVICES os variable before model training )

##### used by

all

##### type

str

##### default

'auto'

#### pl_trainer_precision

The data precision of model and during model training should be trained.

##### used by

all

##### type

str

##### default

_'32-true'_

##### Constraints

choose from [_'32-true','bf16-mixed','16-mixed','bf16','16'_]

#### pl_trainer_num_nodes

Number of nodes in the training cluster running pytorch lightning training

##### used by

all

##### type

int

##### default

_1_

#### pl_trainer_strategy

Strategies that change the behaviour of the training, validation and test- loop.

##### used by

all

##### type

str

##### default

_'auto'_


#### ddp_backend

The distributed training backend for model weights/gradients communication.

Choose available ddp_backend for certain operating system and computing hardwares.

##### used by

all

##### type

str

##### default

_'nccl'_

##### Constraints

choose from [_'gloo', 'nccl', 'nccl_no_p2p'_]

