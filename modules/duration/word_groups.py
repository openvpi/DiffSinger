"""Helpers for the note/syllable groups used by duration prediction.

A word is the unit whose total duration is fixed by the score: the phonemes it
contains share one frame budget, and the duration predictor only decides how that
budget is split among them. Training marks the words with ``ph2word`` (index 0
means padding), while the exported graph derives them from ``word_div`` with the
length regulator. All helpers use only export-safe tensor operations, so the same
code path serves training, validation and the exported graph.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "word_membership",
    "word_sizes",
    "word_position_ids",
    "word_distribution",
    "word_log_softmax",
    "word_budget",
    "allocate_word_frames",
]

# Large negative logit used instead of -inf so that empty rows stay finite.
_MASKED_LOGIT = -1e4
_EPS = 1e-8


def word_membership(word_ids: Tensor, num_words=None, x_masks: Tensor = None) -> Tensor:
    """One-hot membership mask of the words.

    :param word_ids: [B, T] word index per item, 1-based, 0 for padding
    :param num_words: number of words; when omitted it is taken from the largest
        index in ``word_ids``, which keeps the mask compatible with a dynamic
        number of words
    :param x_masks: [B, T] bool mask, True for padded items
    :return: [B, T, W] bool mask
    """
    if num_words is None:
        num_words = word_ids.max()
    index = torch.arange(1, num_words + 1, device=word_ids.device)
    mask = word_ids[:, :, None] == index[None, None, :]
    if x_masks is not None:
        mask = mask & ~x_masks.bool()[:, :, None]
    return mask


def word_sizes(mask: Tensor) -> Tensor:
    """:param mask: [B, T, W] membership mask; :return: [B, W] items per word"""
    return mask.sum(dim=1)


def word_position_ids(mask: Tensor, max_position: int = None):
    """Positions of every item inside its word, counted forward and backward.

    :param mask: [B, T, W] membership mask
    :param max_position: optional clamp, so long words saturate instead of
        growing the embedding table
    :return: (forward, reverse) [B, T] long tensors; padded items map to 0
    """
    numeric = mask.to(torch.int64)  # CumSum has no boolean kernel
    running = numeric.cumsum(dim=1) * numeric  # [B, T, W]
    forward = running.sum(dim=2) - 1  # [B, T]
    size = (word_sizes(mask)[:, None, :] * numeric).sum(dim=2)  # [B, T]
    reverse = size - 1 - forward
    forward = forward.clamp_min(0)
    reverse = reverse.clamp_min(0)
    if max_position is not None:
        forward = forward.clamp_max(max_position)
        reverse = reverse.clamp_max(max_position)
    return forward, reverse


def word_distribution(duration: Tensor, word_ids: Tensor, mask: Tensor = None) -> Tensor:
    """Normalize durations into a distribution over the phonemes of each word.

    :param duration: [B, T] durations or scores
    :param word_ids: [B, T] word index per item, 0 for padding
    :param mask: optional [B, T] bool mask of items to include; by default every
        item with a positive word index is included
    :return: [B, T] share of each item inside its word, zero elsewhere
    """
    if mask is None:
        mask = word_ids > 0
    shape = duration.shape[0], word_ids.max() + 1
    total = duration.new_zeros(*shape).scatter_add(
        1, word_ids, duration * mask
    )[:, 1:]  # [B, T] => [B, W]
    per_word = total.gather(1, word_ids.clamp(min=1) - 1)  # word total per item
    return duration * mask / per_word.clamp_min(_EPS)


def word_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Log softmax of ``logits`` normalized within each word.

    :param logits: [B, T] unnormalized scores
    :param mask: [B, T, W] membership mask, padded items must be unset
    :return: [B, T] log share of each item inside its word
    """
    expanded = logits[:, :, None].masked_fill(~mask, _MASKED_LOGIT)  # [B, T, W]
    word_max = expanded.max(dim=1).values  # [B, W]
    shifted = (expanded - word_max[:, None, :]).exp() * mask
    log_norm = word_max + shifted.sum(dim=1).clamp_min(_EPS).log()  # [B, W]
    return ((expanded - log_norm[:, None, :]) * mask).sum(dim=2)  # [B, T]


def word_budget(durations: Tensor, word_ids: Tensor, num_words=None) -> Tensor:
    """Sum the durations of the items of every word into its frame budget.

    This is the single place where a frame budget is built, which guarantees the
    "constant inside a word" contract of :func:`allocate_word_frames`.

    :param durations: [B, T] per-item durations in frames
    :param word_ids: [B, T] word index per item, 1-based, 0 for padding
    :param num_words: number of words; when omitted it is taken from the largest
        index in ``word_ids``
    :return: [B, W] frame budget of every word
    """
    if num_words is None:
        num_words = word_ids.max()
    shape = durations.shape[0], num_words + 1
    return durations.new_zeros(*shape).scatter_add(1, word_ids, durations)[:, 1:]


def allocate_word_frames(prob: Tensor, budget: Tensor, word_ids: Tensor, x_masks: Tensor = None) -> Tensor:
    """Turn within-word shares and per-item budgets into frame counts.

    The cumulative share is scaled by the budget, rounded with ties away from zero
    and then differenced. Because the shares of a word sum to one, the boundary at
    the end of a word equals that word's budget, so the counts of every word sum to
    its budget without any repair step. That last boundary is pinned to the
    cumulative budget of the word rather than rounded, which makes the sum exact by
    construction instead of by floating-point luck.

    :param prob: [B, T] within-word shares
    :param budget: [B, T] frame budget of the word each item belongs to; it must be
        constant inside a word, so build it with :func:`word_budget` or by gathering
        the per-word budget with the word ids
    :param word_ids: [B, T] word index per item, 1-based, 0 for padding
    :param x_masks: [B, T] bool mask, True for padded items
    :return: [B, T] frame counts
    """
    # The counts have to be exact integers, so the accumulation is done in float32
    # whatever precision the shares arrive in.
    prob = prob.float()
    budget = budget.to(torch.float32)

    cumulative = (prob.clamp_min(0.) * budget).cumsum(dim=1)
    boundary = (cumulative + 0.5).floor()

    # A word ends where its index changes; the trailing zero is ``word_ids[:, 1:]``
    # shifted left, so the last item of the sequence closes its own word.
    next_ids = F.pad(word_ids[:, 1:], [0, 1])
    word_end = (word_ids != next_ids) & (word_ids > 0)  # [B, T]

    # A phoneme inherits the budget of its own word, so summing it over the word
    # ends restores the cumulative budget word by word, without reading any
    # per-word tensor back.
    end_budget = torch.where(word_end, budget, torch.zeros_like(budget))
    word_cumulative = end_budget.cumsum(dim=1)  # [B, T]
    boundary = torch.where(word_end, word_cumulative, boundary)

    previous = torch.cat([boundary.new_zeros(boundary.shape[0], 1), boundary[:, :-1]], dim=1)
    counts = boundary - previous
    if x_masks is not None:
        counts = counts.masked_fill(x_masks.bool(), 0.)
    return counts
