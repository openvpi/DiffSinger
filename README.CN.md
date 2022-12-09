# 为何选用本重构分支？
这是更加洁净高效的DiffSinger版本。它的优化体现在：
- 拆除垃圾山：许多在DiffSinger中使用不到的脚本被标记为**\*隔离的\***
- 提升可读性：一些重要的函数已经被注释了（**但是，这也是建立在认为读者熟悉神经网络原理的基础上的**）
- 分离混杂类：相当重要的类们已经被分到了 "basics/" 文件夹并且有详细的注解，其他类则大多从这些类中继承与发展出来。
- 改善文件树：与TTS（文字转语音）相关的那些DiffSinger用不到的文件被提取出来，然后丢进 "tts/" 文件夹。
- **（新增！）预处理、训练和推理变得更加浓缩**。 预处理Pipeline在 'preprocessing/opencpop.py', 训练Pipeline在 'training/diffsinger.py',推理Pipeline在 'inference/ds_cascade.py' 或者 'inference/ds_e2e.py'.

# 快速开始

0. 安装

```bash
# 手动安装Pytorch (推荐1.8.2 LTS版本)
# 这里有更多步骤指示： https://pytorch.org/get-started/locally/
# 下面的这些是给CUDA 11.1的范例
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# 安装其他依赖
pip install -r requirements.txt
```
1. 预处理

```sh
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python data_gen/binarize.py --config configs/midi/cascade/opencs/aux_rel.yaml
```

2. 训练

```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name $MY_DS_EXP_NAME --reset  
```

3. 推理

```sh
CUDA_VISIBLE_DEVICES=0 python run.py --config configs/midi/e2e/opencpop/ds100_adj_rel.yaml --exp_name $MY_DS_EXP_NAME --reset --infer
```
Easy inference with Google Colab:

Version 1: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kfmZ6Y018c5trSwQAbhdQtZ7Il8W_4BU)

Version 2: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V7yjNlh8_8o3IPK9buFb5MHVFrYmhELi)

# DiffSinger: 基于Shallow Diffusion的人声歌声合成引擎
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 TTS](https://huggingface.co/spaces/NATSpeech/DiffSpeech)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)

 这个项目是官方的AAAI-2022 [paper](https://arxiv.org/abs/2105.02446) 实现（译者注：指被Fork的仓库），这个实现中包括着DiffSinger (歌声合成)和 DiffSpeech (文字朗读)

*请注意：后者在此Fork中被移除*

<table style="width:100%">
  <tr>
    <th>训练中的DiffSinger/DiffSpeech</th>
    <th>推理中的DiffSinger/DiffSpeech</th>
  </tr>
  <tr>
    <td><img src="docs/resources/model_a.png" alt="Training" height="300"></td>
    <td><img src="docs/resources/model_b.png" alt="Inference" height="300"></td>
  </tr>
</table>

:tada: :tada: :tada: **更新**:
 - Sep.11, 2022: :electric_plug: [DiffSinger-PN](docs/README-SVS-opencpop-pndm.md). 添加了插件 [PNDM](https://arxiv.org/abs/2202.09778), ICLR 2022 在我们的实验中，以便方便的加速DiffSinger.
 - Jul.27, 2022: Update documents for [SVS](docs/README-SVS.md). Add easy inference [A](docs/README-SVS-opencpop-cascade.md#4-inference-from-raw-inputs) & [B](docs/README-SVS-opencpop-e2e.md#4-inference-from-raw-inputs); 添加了交互式SVS在 [HuggingFace🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger).
 - Mar.2, 2022: MIDI-B-version.
 - Mar.1, 2022: [NeuralSVB](https://github.com/MoonInTheRiver/NeuralSVB), 歌声美化解决方案已经发布.
 - Feb.13, 2022: [NATSpeech](https://github.com/NATSpeech/NATSpeech), 这一被改善的代码框架，伴随着针对DiffSinger 和 NeurIPS-2021相关工作的实现 [PortaSpeech](https://openreview.net/forum?id=xmJsuh8xlq) 已经被发布了. 
 - Jan.29, 2022: 支持 MIDI-A-version SVS.
 - Jan.13, 2022: 支持 SVS, 发布 PopCS 数据集.
 - Dec.19, 2021: 支持 TTS. [HuggingFace🤗 TTS](https://huggingface.co/spaces/NATSpeech/DiffSpeech)

:rocket: **新闻**: 
 - Feb.24, 2022: 我们的新成果， NeuralSVB已经被ACL-2022接受 [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2202.13277). [Demo Page](https://neuralsvb.github.io).
 - Dec.01, 2021: DiffSinger 被 AAAI-2022 接受.
 - Sep.29, 2021: 我们最近的工作: `PortaSpeech: 便携与高质量的文字朗读` 已经被 NeurIPS-2021 接受 [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2109.15166) .
 - May.06, 2021: 我们已将DiffSinger提交给 Arxiv [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446).

## 环境配置
```sh
conda create -n your_env_name python=3.8
source activate your_env_name 
pip install -r requirements_2080.txt   (GPU 2080Ti, CUDA 10.2)
# 或者使用： pip install -r requirements_3090.txt   (GPU 3090, CUDA 11.4)
```

## 文档

- [运行 DiffSpeech (文字朗读)](docs/README-TTS.md).
- [运行 DiffSinger (歌声合成)](docs/README-SVS.md).

## Tensorboard
```sh
tensorboard --logdir_spec exp_name
```
<table style="width:100%">
  <tr>
    <td><img src="docs/resources/tfb.png" alt="Tensorboard" height="250"></td>
  </tr>
</table>

## 音频范例
早期的一些音频范例可以在[demo page](https://diffsinger.github.io/) 找到。用这个仓库（原仓库）生成的范例在下面列出：

### 文字朗读范例

朗读范例 (LJSpeech的测试集) 可以在 [demos_1213](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_1213) 找到

### 歌声合成范例

歌声范例（PopCS的测试集）可以在 [demos_0112](https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/demos_0112) 找到

## 引用
    @article{liu2021diffsinger,
      title={Diffsinger: Singing voice synthesis via shallow diffusion mechanism},
      author={Liu, Jinglin and Li, Chengxi and Ren, Yi and Chen, Feiyang and Liu, Peng and Zhao, Zhou},
      journal={arXiv preprint arXiv:2105.02446},
      volume={2},
      year={2021}}

## 特别鸣谢

我们的代码基于这些项目:
* [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
* [HifiGAN](https://github.com/jik876/hifi-gan)
* [espnet](https://github.com/espnet/espnet)
* [DiffWave](https://github.com/lmnt-com/diffwave)

同时也感谢由[Keon Lee](https://github.com/keonlee9420/DiffSinger)提供的又快又好的实现。
