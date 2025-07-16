import torch
import torch.nn.functional as F


def fast_bbc_mask(mel2ph, mask_length=3, min_segment_length=5, mask_prob=1.):

    batch_size, seq_len = mel2ph.shape
    device = mel2ph.device
    masked_mel2ph = torch.where(mel2ph > 0, mel2ph + 1, mel2ph)

    padded = F.pad(masked_mel2ph, [1, 1], value=-1)  # [B, L+2]

    diff_mask = padded[:, 1:] != padded[:, :-1]  # [B, L+1]
    result = masked_mel2ph.clone()
    for batch_idx in range(batch_size):
        seq = masked_mel2ph[batch_idx]
        boundaries = torch.where(diff_mask[batch_idx])[0]

        if len(boundaries) < 2:
            continue

        starts = boundaries[:-1]
        ends = boundaries[1:]
        lengths = ends - starts
        values = seq[starts]

        valid_mask = (values != 0) & (lengths >= min_segment_length)

        if not valid_mask.any():
            continue

        valid_indices = torch.where(valid_mask)[0]
        random_mask = torch.rand(len(valid_indices), device=device) < mask_prob
        selected_segments = valid_indices[random_mask]

        for seg_idx in selected_segments:
            start = starts[seg_idx]
            end = ends[seg_idx]
            mask_start = max(start, end - mask_length)
            result[batch_idx, mask_start:end] = 1

    return result
