# Best Practices

## Using custom dictionaries

This section is about using a custom grapheme-to-phoneme dictionary for any language(s).

### Dictionary format

A dictionary is a .txt file, in which each line represents a mapping rule from one syllable to its phoneme sequence. The syllable and the phonemes are split by `tab`, and the phonemes are split by `space`:

```
<syllable>	<phoneme1> <phoneme2> ...
```

Syllable names and phoneme names can be customized, but with the following limitations/suggestions:

- `SP` (rest), `AP` (breath) and `<PAD>` (padding) cannot be phoneme names because they are reserved.
- `-` and `+` cannot be phoneme names because they are defined as slur tags in most singing voice synthesis editors.
- Special characters including but not limited to `@`, `#`, `&`, `|`, `/`, `<`, `>`, etc. should be avoided because they may be used as special tags in the future format changes. Using them now is okay, and all modifications will be notified in advance.
- ASCII characters are preferred for the most encoding compatibility, but all UTF-8 characters are acceptable.

### Add a dictionary

Assume you have made a dictionary file named `my_dict.txt`. Edit your  configuration file:

```yaml
dictionary: my_dict.txt
```

Then you can binarize your data as normal. The phonemes in your dataset must cover, and must only cover the phonemes appeared in your dictionary. Otherwise, the binarizer will raise an error:

```
AssertionError: transcriptions and dictionary mismatch.
 (+) ['E', 'En', 'i0', 'ir']
 (-) ['AP', 'SP']
```

This means there are 4 unexpected symbols in the data labels (`ir`, `i0`, `E`, `En`) and 2 missing phonemes that are not covered by the data labels (`AP`, `SP`).

Once the coverage checks passed, a phoneme distribution summary will be saved into your binary data directory. Below is an example.

![phoneme-distribution](resources/phoneme-distribution.jpg)

During the binarization process, each phoneme will be assigned with a unique phoneme ID according the order of their names. By default, there are one padding index before all real phonemes IDs. You may edit the number of padding indices, but it is not recommended to do so:

```yaml
num_pad_tokens: 1
```

The dictionary used to binarize the dataset will be copied to the binary data directory by the binarizer, and will be copied again to the experiment directory by the trainer. When exported to ONNX, the dictionary and the phoneme sequence ordered by IDs will be saved to the artifact directory. You do not need to carry the original dictionary file for training and inference.

### Preset dictionaries

There are currently some preset dictionaries for users to use directly:

|     dictionary     |        filename        | description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|:------------------:|:----------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      Opencpop      |      opencpop.txt      | The original dictionary used by the Opencpop mandarin singing dataset that is fully aligned with the pinyin writing system. We copied the dictionary from [here](http://wenet.org.cn/opencpop/resources/annotationformat/), removed 5 syllables that has no occurrence in the data labels (`hm`, `hng`, `m`, `n` and `ng`) and added some aliases for some syllables (e.g. `jv` for `ju`). Due to pronunciation issues, this dictionary is deprecated and remained only for backward compatibility. |
| Opencpop extension | opencpop-extension.txt | The modified version of the opencpop dictionary, with stricter phoneme division rules for some pinyin syllables. For example, `ci` is mapped to `c i0` and `chi` is mapped to `ch ir` to distinguish with `bi` (`b i`). This dictionary is now used as the default dictionary for mandarin Chinese. There are also many new syllables for more phoneme combinations.                                                                                                                                |

### Submit or propose a new dictionary

You can submit or propose a new dictionary by raising a topic in [Discussions](https://github.com/openvpi/DiffSinger/discussions). Any dictionary to be formally supported in the main branch must match the following principles:

- Only monolingual dictionaries are accepted for now. Support for multilingual dictionaries will be designed in the future.
- All syllables and phonemes in the dictionary should have linguistic meanings. Style tags (vocal fry, falsetto, etc.) should not appear in the dictionary.
- Its syllables should be standard spelling or phonetic transcriptions (like pinyin in mandarin Chinese and romaji in Japanese) for easy integration with G2P modules.
- Its phonemes should cover all (or almost all) possible pronunciations in that language.
- Every syllable and every phoneme should have one, and only one certain pronunciation, in all or almost all situations in that language. Some slightly context-based pronunciation differences are allowed as the networks can learn.
- Most native speakers/singers of that language should be able to easily cover all phonemes in the dictionary. This means the dictionary should not contain extremely rare or highly customized phonemes of some dialects or accents.
- It should not bring too much difficulty and complexity to the data labeling workflow, and it should be easy to use for end users of voicebanks.

## Performance tuning

This section is about accelerating training and utilizing hardware.

### Data loader and batch sampler

The data loader loads data pieces from the binarized dataset, and the batch sampler forms batches according to data lengths.

To configure the data loader, edit your configuration file:

```yaml
ds_workers: 4  # number of DataLoader workers
dataloader_prefetch_factor: 2  # load data in advance
```

To configure the batch sampler, edit your configuration file:

```yaml
sampler_frame_count_grid: 6  # lower value means higher speed but less randomness
```

For more details of the batch sampler algorithm and this configuration key, see [sampler_frame_count_grid](ConfigurationSchemas.md#sampler_frame_count_grid).

### Automatic mixed precision

Enabling automatic mixed precision (AMP) can accelerate training and save GPU memory. DiffSinger have adapted the latest version of PyTorch Lightning for AMP functionalities.

By default, the training runs in FP32 precision. To enable AMP, edit your configuration file:

```yaml
pl_trainer_precision: 16  # FP16 precision
```

or

```yaml
pl_trainer_precision: bf16  # BF16 precision
```

For more precision options, please checkout the official [documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html#precision).

### Training on multiple GPUs

Using data distributed parallel (DDP) can divide training tasks to multiple GPUs and synchronize gradients and weights between them. DiffSinger have adapted the latest version of PyTorch Lightning for DDP functionalities.

By default, the trainer will utilize all CUDA devices defined in the `CUDA_VISIBLE_DEVICES` environment variable (empty means using all available devices). If you want to specify which GPUs to use, edit your configuration file:

```yaml
pl_trainer_devices: [0, 1, 2, 3]  # using the first 4 GPUs defined in CUDA_VISIBLE_DEVICES
```

Please note that `max_batch_size` and `max_batch_frames` are values for **each** GPU.

By default, the trainer uses NCCL as the DDP backend. If this gets stuck on your machine, try disabling P2P via

```yaml
ddp_backend: nccl_no_p2p  # disable P2P in NCCL
```

Or if your machine does not support NCCL, you can switch to Gloo instead:

```yaml
ddp_backend: gloo  # however, it has a lower performance than NCCL
```

### Gradient accumulation

Gradient accumulation means accumulating losses for several batches before each time the weights are updated. This can simulate a larger batch size with a lower GPU memory cost.

By default, the trainer calls `backward()` each time the losses are calculated through one batch of data. To enable gradient accumulation, edit your configuration file:

```yaml
accumulate_grad_batches: 4  # the actual batch size will be 4x.
```

Please note that enabling gradient accumulation will slow down training because the losses must be calculated for several times before the weights are updated (1 update to the weights = 1 actual training step).

### Optimizer and learning rate

The optimizer and the learning rate scheduler can take an important role in accelerating the training process. DiffSinger uses a flexible configuration logic for these two modules.

You can modify options of the optimizer and learning rate scheduler, or even use other classes of them by editing the configuration file:

```yaml
optimizer_args:
  optimizer_cls: torch.optim.AdamW  # class name of optimizer
  lr: 0.0004
  beta1: 0.9
  beta2: 0.98
  weight_decay: 0
lr_scheduler_args:
  scheduler_cls: torch.optim.lr_scheduler.StepLR  # class name of learning rate schedule
  warmup_steps: 2000
  step_size: 50000
  gamma: 0.5
```

Note that `optimizer_args` and `lr_scheduler_args` will be filtered by needed parameters and passed to `__init__` as keyword arguments (`kwargs`) when constructing the optimizer and scheduler. Therefore, you could specify all arguments according to your need in the configuration file to directly control the behavior of optimization and LR scheduling. It will also tolerate parameters existing in the configuration but not needed in `__init__`.

Also, note that the LR scheduler performs scheduling on the granularity of steps, not epochs.

The special case applies when a tuple is needed in `__init__`: `beta1` and `beta2` are treated separately and form a tuple in the code. You could try to pass in an array instead. (And as an experiment, AdamW does accept `[beta1, beta2]`). If there is another special treatment required, please submit an issue.

If you found other optimizer and learning rate scheduler useful, you can raise a topic in [Discussions](https://github.com/openvpi/DiffSinger/discussions), raise [Issues](https://github.com/openvpi/DiffSinger/issues) or submit [PRs](https://github.com/openvpi/DiffSinger/pulls) if it introduces new codes or dependencies.
