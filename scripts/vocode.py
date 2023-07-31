# coding=utf8
import argparse
import os
import pathlib
import sys

root_dir = pathlib.Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

import numpy as np
import torch
import tqdm

from inference.ds_acoustic import DiffSingerAcousticInfer
from utils.infer_utils import cross_fade, save_wav
from utils.hparams import set_hparams, hparams

parser = argparse.ArgumentParser(description='Run DiffSinger vocoder')
parser.add_argument('mel', type=str, help='Path to the input file')
parser.add_argument('--exp', type=str, required=False, help='Read vocoder class and path from chosen experiment')
parser.add_argument('--config', type=str, required=False, help='Read vocoder class and path from config file')
parser.add_argument('--class', type=str, required=False, help='Specify vocoder class')
parser.add_argument('--ckpt', type=str, required=False, help='Specify vocoder checkpoint path')
parser.add_argument('--out', type=str, required=False, help='Path of the output folder')
parser.add_argument('--title', type=str, required=False, help='Title of output file')
args = parser.parse_args()

mel = pathlib.Path(args.mel)
name = mel.stem if not args.title else args.title
config = None
if args.exp:
    config = root_dir / 'checkpoints' / args.exp / 'config.yaml'
elif args.config:
    config = pathlib.Path(args.config)
else:
    assert False, 'Either argument \'--exp\' or \'--config\' should be specified.'

sys.argv = [
    sys.argv[0],
    '--config',
    str(config)
]
set_hparams(print_hparams=False)

cls = getattr(args, 'class')
if cls:
    hparams['vocoder'] = cls
if args.ckpt:
    hparams['vocoder_ckpt'] = args.ckpt


out = args.out
if args.out:
    out = pathlib.Path(args.out)
else:
    out = mel.parent

mel_seq = torch.load(mel)
assert isinstance(mel_seq, list), 'Not a valid mel sequence.'
assert len(mel_seq) > 0, 'Mel sequence is empty.'

sample_rate = hparams['audio_sample_rate']
infer_ins = DiffSingerAcousticInfer(load_model=False)


def run_vocoder(path: pathlib.Path):
    result = np.zeros(0)
    current_length = 0

    for seg_mel in tqdm.tqdm(mel_seq, desc='mel segment', total=len(mel_seq)):
        seg_audio = infer_ins.run_vocoder(seg_mel['mel'].to(infer_ins.device), f0=seg_mel['f0'].to(infer_ins.device))
        seg_audio = seg_audio.squeeze(0).cpu().numpy()
        silent_length = round(seg_mel['offset'] * sample_rate) - current_length
        if silent_length >= 0:
            result = np.append(result, np.zeros(silent_length))
            result = np.append(result, seg_audio)
        else:
            result = cross_fade(result, seg_audio, current_length + silent_length)
        current_length = current_length + silent_length + seg_audio.shape[0]

    print(f'| save audio: {path}')
    save_wav(result, path, sample_rate)


os.makedirs(out, exist_ok=True)
try:
    run_vocoder(out / (name + '.wav'))
except KeyboardInterrupt:
    exit(-1)
