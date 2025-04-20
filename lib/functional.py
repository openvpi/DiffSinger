import torch


@torch.no_grad()
def dur_to_mel2ph(lr, durs, length, timestep):
    non_batched = durs.ndim == 1
    if non_batched:
        b = 1
        durs = durs.unsqueeze(0)
    else:
        b = durs.shape[0]
    ph_acc = torch.round(torch.cumsum(durs, dim=1) / timestep + 0.5).long()
    ph_dur = torch.diff(ph_acc, dim=1, prepend=torch.zeros(b, 1).to(durs.device))
    mel2ph = lr(ph_dur)
    num_frames = mel2ph.shape[1]
    if num_frames < length:
        mel2ph = torch.nn.functional.pad(mel2ph, (0, length - num_frames), mode="replicate")
    elif num_frames > length:
        mel2ph = mel2ph[:, :length]
    if non_batched:
        mel2ph = mel2ph.squeeze(0)
    return mel2ph


def mel2ph_to_dur(mel2ph, n_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, n_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur
