# Getting Started

## Installation

### Environments and dependencies

DiffSinger requires Python 3.8 or later. We strongly recommend you create a virtual environment via Conda or venv before installing dependencies.

Install PyTorch 1.13 or later following the [official instructions](https://pytorch.org/get-started/locally/) according to your OS and hardware.

Install other dependencies via the following command:

```bash
pip install -r requirements.txt
```

### Pretrained models

- **(Required)** Get the pretrained vocoder from the [DiffSinger Community Vocoders Project](https://openvpi.github.io/vocoders) and unzip it into `checkpoints/` folder, or train a ultra-lightweight [DDSP](https://github.com/yxlllc/pc-ddsp) vocoder first by yourself, then configure it according to the relevant [instructions](https://github.com/yxlllc/pc-ddsp/blob/master/DiffSinger.md).
- Get acoustic or variance models from [Releases](https://github.com/openvpi/DiffSinger/releases) or elsewhere and unzip them into the `checkpoints/` folder.

## Preprocessing

Raw data pieces and transcriptions should be binarized into dataset files before training.

Assume that you have a configuration file called `my_config.yaml`. Run:

```bash
python scripts/binarize.py --config my_config.yaml
```

Preprocessing can be accelerated through multiprocessing. See [binarization_args.num_workers](ConfigurationSchemas.md#binarization_args.num_workers) for more explanations.

## Training

Assume that you have a configuration file called `my_config.yaml` and the name of your model is `my_experiment`. Run:

```bash
python scripts/train.py --config my_config.yaml --exp_name my_experiment --reset
```

Checkpoints will be saved at the `checkpoints/my_experiment/` directory. When interrupting the program and running the above command again, the training resumes automatically from the latest checkpoint.

For more suggestions related to training performance, see [performance tuning](BestPractices.md#performance-tuning).

## Inference

Inference of DiffSinger is based on DS files. Assume that you have a DS file named `my_song.ds` and your model is named `my_experiment`.

If your model is a variance model, run:

```bash
python scripts/infer.py variance my_song.ds --exp my_experiment
```

or run

```bash
python scripts/infer.py variance --help
```

for more configurable options.

If your model is an acoustic model, run:

```bash
python scripts/infer.py acoustic my_song.ds --exp my_experiment
```

or run

```bash
python scripts/infer.py acoustic --help
```

for more configurable options.

## Deployment

DiffSinger uses [ONNX](https://onnx.ai/) as the deployment format. Due to TorchScript issues, exporting to ONNX now requires PyTorch **1.13**. Assume that you have a model named `my_experiment`.

If your model is a variance model, run:

```bash
python scripts/export.py --exp my_experiment
```

or run

```bash
python scripts/export.py variance --help
```

for more configurable options.

If your model is an acoustic model, run:

```bash
python scripts/export.py acoustic --exp my_experiment
```

or run

```bash
python scripts/export.py acoustic --help
```

for more configurable options.

## Other utilities

There are other useful CLI tools in the [scripts/](../scripts/) directory not mentioned above:

- drop_spk.py - delete speaker embeddings from checkpoints (for data security reasons when distributing models)
- migrate.py - migrate old transcription files or checkpoints from previous versions of DiffSinger
- vocoder.py - bypass the acoustic model and only run the vocoder on given mel-spectrograms.
