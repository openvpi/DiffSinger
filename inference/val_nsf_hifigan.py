import os
import sys

import librosa
import numpy as np
import resampy
import torch
import torchcrepe
import tqdm

from lib.feature.binarizer_utils import get_pitch_parselmouth, get_mel_torch
from modules.vocoders.nsf_hifigan import NsfHifiGAN
from utils.infer_utils import save_wav
from utils.hparams import set_hparams, hparams

sys.argv = [
    'inference/svs/ds_acoustic.py',
    '--config',
    'configs/acoustic.yaml',
]


def get_pitch(wav_data, mel, hparams, threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # crepe只支持16khz采样率，需要重采样
    wav16k = resampy.resample(wav_data, hparams['audio_sample_rate'], 16000)
    wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(device)

    # 频率范围
    f0_min = 40
    f0_max = 800

    # 重采样后按照hopsize=80,也就是5ms一帧分析f0
    f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, f0_min, f0_max, pad=True, model='full', batch_size=1024,
                                device=device, return_periodicity=True)

    # 滤波，去掉静音，设置uv阈值，参考原仓库readme
    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.)(pd, wav16k_torch, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)

    # 将nan频率（uv部分）转换为0频率
    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)

    # 去掉0频率，并线性插值
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = np.arange(len(mel)) * hparams['hop_size'] / hparams['audio_sample_rate']
    f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    return f0


set_hparams()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocoder = NsfHifiGAN()
in_path = 'path/to/input/wavs'
out_path = 'path/to/output/wavs'
os.makedirs(out_path, exist_ok=True)
for filename in tqdm.tqdm(os.listdir(in_path)):
    if not filename.endswith('.wav'):
        continue
    wav, _ = librosa.load(os.path.join(in_path, filename), sr=hparams['audio_sample_rate'], mono=True)
    mel = get_mel_torch(
        wav, hparams['audio_sample_rate'], num_mel_bins=hparams['audio_num_mel_bins'],
        hop_size=hparams['hop_size'], win_size=hparams['win_size'], fft_size=hparams['fft_size'],
        fmin=hparams['fmin'], fmax=hparams['fmax'],
        device=device
    )

    f0, _ = get_pitch_parselmouth(
        wav, samplerate=hparams['audio_sample_rate'], length=len(mel),
        hop_size=hparams['hop_size']
    )

    wav_out = vocoder.spec2wav(mel, f0=f0)
    save_wav(wav_out, os.path.join(out_path, filename), hparams['audio_sample_rate'])
