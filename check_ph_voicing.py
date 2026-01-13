import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn.functional as F

from modules.fastspeech.tts_modules import mel2ph_to_dur
from utils.indexed_datasets import IndexedDataset

data_file = pathlib.Path(r"data/qixuan_v4/binary/acoustic/train.data")
meta_file = pathlib.Path(r"data/qixuan_v4/binary/acoustic/train.meta")
inspect_phonemes = ["SP", "AP"]
value_threshold = -40
frame_threshold = 5
save_dir = pathlib.Path(r"data/_check")

save_dir.mkdir(parents=True, exist_ok=True)

with open(meta_file, 'rb') as f:
    metadata = pickle.load(f)
data = IndexedDataset(data_file.parent, data_file.stem)
for i, name in enumerate(metadata["names"]):
    spk = metadata["spk_names"][i]
    sample = data[i]
    ph_text = metadata["ph_texts"][i].split()
    inspect_mask_ph = torch.tensor([
        ph in inspect_phonemes
        for ph in ph_text
    ])  # [T_ph,]
    if not inspect_mask_ph.any():
        continue
    mel2ph = sample["mel2ph"]  # [T_mel,]
    inspect_mask_mel = torch.gather(F.pad(inspect_mask_ph, [1, 0]), 0, mel2ph)  # [T_mel,]
    if not inspect_mask_mel.any():
        continue
    voicing = sample["voicing"]  # [T_mel,]
    inspect_voicing_max = voicing[inspect_mask_mel].max()
    inspect_over_threshold = (voicing >= value_threshold) & inspect_mask_mel  # [T_mel,]
    inspect_over_threshold_frames = inspect_over_threshold.sum()
    if inspect_over_threshold_frames < frame_threshold:
        continue
    durations = mel2ph_to_dur(mel2ph[None], len(ph_text))[0]  # [T_ph,]
    print(f"{i:06d}_{spk}_{name}: frameCount = {inspect_over_threshold_frames}, maxVoicing = {inspect_voicing_max:.2f}")
    mel = sample["mel"].cpu().numpy().T
    emphasis_mask = inspect_over_threshold[None].repeat(mel.shape[0], 1).cpu().numpy()
    alpha = numpy.where(emphasis_mask, 0.4, 0)
    curve = (voicing.cpu().numpy() / 96 + 1) * mel.shape[0]
    fig = plt.figure(figsize=(24, 6))
    plt.pcolor(mel, vmin=-14, vmax=4)
    plt.pcolor(emphasis_mask, alpha=alpha, color='red')
    plt.plot((voicing / 96 + 1) * 128, label="voicing", color="white")
    plt.plot(numpy.full_like(voicing, (value_threshold / 96 + 1) * 128),
             label=f"threshold={value_threshold:.2f}", color="yellow", linestyle="--")
    ph_edges = torch.cumsum(durations, dim=0)[:-1]
    plt.vlines(ph_edges.cpu().numpy(), 0, mel.shape[0], colors='cyan', linestyles='--', label='boundaries')
    x_pos = 0
    for ph_name, duration in zip(ph_text, durations.cpu().tolist()):
        color = "yellow" if ph_name in inspect_phonemes else "white"
        plt.text(
            x_pos + duration / 2, 10, ph_name,
            size=12, horizontalalignment='center', color=color
        )
        x_pos += duration
    plt.ylim(0, mel.shape[0])
    plt.title(f"{spk} - {name}: frameCount = {inspect_over_threshold_frames}, maxVoicing = {inspect_voicing_max:.2f}")
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_dir / f"{i:06d}_{spk}_{name}.jpg")
    plt.close(fig)

del data
