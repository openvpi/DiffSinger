# coding=utf8
import argparse
import json
import os
import pathlib
import sys

root_dir = str(pathlib.Path(__file__).parent.parent.resolve())
os.environ["PYTHONPATH"] = root_dir
sys.path.insert(0, root_dir)

import numpy as np
import torch

from inference.ds_acoustic import DiffSingerAcousticInfer
from utils.hparams import set_hparams, hparams
from utils.infer_utils import (
    merge_slurs,
    cross_fade,
    trans_key,
    parse_commandline_spk_mix,
    save_wav,
)

parser = argparse.ArgumentParser(description="Run DiffSinger inference")
parser.add_argument("proj", type=str, help="Path to the input file")
parser.add_argument("--exp", type=str, required=True, help="Selection of model")
parser.add_argument(
    "--ckpt", type=int, required=False, help="Selection of checkpoint training steps"
)
parser.add_argument("--spk", type=str, required=False, help="Speaker name or mix of speakers")
parser.add_argument("--out", type=str, required=False, help="Path of the output folder")
parser.add_argument("--title", type=str, required=False, help="Title of output file")
parser.add_argument("--num", type=int, required=False, default=1, help="Number of runs")
parser.add_argument(
    "--key", type=int, required=False, default=0, help="Key transition of pitch"
)
parser.add_argument(
    "--gender", type=float, required=False, default=0, help="Formant shifting (gender control)"
)
parser.add_argument("--seed", type=int, required=False, help="Random seed of the inference")
parser.add_argument(
    "--speedup", type=int, required=False, default=0, help="PNDM speed-up ratio"
)
parser.add_argument(
    "--mel",
    action="store_true",
    required=False,
    default=False,
    help="Save intermediate mel format instead of waveform",
)
args = parser.parse_args()

name = args.title or os.path.basename(args.proj).split(".")[0]
exp = args.exp
if not os.path.exists(f"{root_dir}/checkpoints/{exp}"):
    for ckpt in os.listdir(os.path.join(root_dir, "checkpoints")):
        if ckpt.startswith(exp):
            print(f"| match ckpt by prefix: {ckpt}")
            exp = ckpt
            break
    assert os.path.exists(f"{root_dir}/checkpoints/{exp}"), (
        "There are no matching exp in 'checkpoints' folder. "
        "Please specify '--exp' as the folder name or prefix."
    )
else:
    print(f"| found ckpt by name: {exp}")

out = args.out
if not out:
    out = os.path.dirname(os.path.abspath(args.proj))

sys.argv = [f"{root_dir}/inference/ds_cascade.py", "--exp_name", exp, "--infer"]

with open(args.proj, "r", encoding="utf-8") as f:
    params = json.load(f)
if not isinstance(params, list):
    params = [params]

if args.key != 0:
    params = trans_key(params, args.key)
    key_suffix = "%+dkey" % args.key
    if not args.title:
        name += key_suffix
    print(f"| key transition: {args.key:+d}")

if args.gender is not None:
    assert -1 <= args.gender <= 1, "Gender must be in [-1, 1]."

set_hparams(print_hparams=False)
if args.speedup > 0:
    hparams["pndm_speedup"] = args.speedup

sample_rate = hparams["audio_sample_rate"]

# Check for vocoder path
assert os.path.exists(os.path.join(root_dir, hparams["vocoder_ckpt"])), (
    f'Vocoder ckpt \'{hparams["vocoder_ckpt"]}\' not found. '
    f"Please put it to the checkpoints directory to run inference."
)

infer_ins = None
if len(params) > 0:
    infer_ins = DiffSingerAcousticInfer(load_vocoder=not args.mel, ckpt_steps=args.ckpt)

spk_mix = (
    parse_commandline_spk_mix(args.spk)
    if hparams["use_spk_id"] and args.spk is not None
    else None
)

for param in params:
    if args.gender is not None and hparams.get("use_key_shift_embed"):
        param["gender"] = args.gender

    if spk_mix is not None:
        param["spk_mix"] = spk_mix

    merge_slurs(param)


def infer_once(path: str, save_mel=False):
    result = [] if save_mel else np.zeros(0)
    current_length = 0

    for i, param in enumerate(params):
        if "seed" in param:
            print(f'| set seed: {param["seed"] & 0xffff_ffff}')
            torch.manual_seed(param["seed"] & 0xFFFF_FFFF)
            torch.cuda.manual_seed_all(param["seed"] & 0xFFFF_FFFF)
        elif args.seed:
            print(f"| set seed: {args.seed & 0xffff_ffff}")
            torch.manual_seed(args.seed & 0xFFFF_FFFF)
            torch.cuda.manual_seed_all(args.seed & 0xFFFF_FFFF)
        else:
            torch.manual_seed(torch.seed() & 0xFFFF_FFFF)
            torch.cuda.manual_seed_all(torch.seed() & 0xFFFF_FFFF)

        if save_mel:
            mel, f0 = infer_ins.infer_once(param, return_mel=True)
            result.append({"offset": param.get("offset", 0.0), "mel": mel, "f0": f0})
        else:
            seg_audio = infer_ins.infer_once(param)
            silent_length = round(param.get("offset", 0) * sample_rate) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_audio)
            else:
                result = cross_fade(result, seg_audio, current_length + silent_length)
            current_length = current_length + silent_length + seg_audio.shape[0]
        sys.stdout.flush()
        print(
            "| finish segment: %d/%d (%.2f%%)"
            % (i + 1, len(params), (i + 1) / len(params) * 100)
        )

    if save_mel:
        print(f"| save mel: {path}")
        torch.save(result, path)
    else:
        print(f"| save audio: {path}")
        save_wav(result, path, sample_rate)


os.makedirs(out, exist_ok=True)
suffix = ".mel.pt" if args.mel else ".wav"
if args.num == 1:
    infer_once(os.path.join(out, f"{name}{suffix}"), save_mel=args.mel)
else:
    for i in range(1, args.num + 1):
        infer_once(os.path.join(out, f"{name}-{str(i).zfill(3)}{suffix}"), save_mel=args.mel)
