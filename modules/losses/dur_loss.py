import torch
import torch.nn as nn
from torch import Tensor

from modules.duration.word_groups import word_distribution


class DurationLoss(nn.Module):
    """
    Combines phoneme, word-allocation, word and sentence duration losses.

    The allocation term models each word's phonemes as a distribution over that
    word's frame budget and trains it with a cross entropy; it is scale invariant
    inside a word, so the word and sentence terms keep the absolute frame scale
    meaningful. When the predictor is given the budget it already reproduces both
    sums, so those terms are zero and :func:`build_duration_loss` disables them.
    """

    def __init__(self, offset, loss_type,
                 lambda_pdur=0.6, lambda_wdur=0.3, lambda_sdur=0.1, lambda_alloc=0.0):
        """Initialize the module.

        Args:
            offset (float): Offset of the log-domain transform.
            loss_type (str): Loss type, either ``'mse'`` or ``'huber'``.
            lambda_pdur (float): Weight of the phoneme term.
            lambda_wdur (float): Weight of the word term.
            lambda_sdur (float): Weight of the sentence term.
            lambda_alloc (float): Weight of the within-word allocation term.
                Zero disables it.
        """
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss = nn.MSELoss()
        elif self.loss_type == 'huber':
            self.loss = nn.HuberLoss()
        else:
            raise NotImplementedError()
        self.offset = offset

        self.lambda_pdur = lambda_pdur
        self.lambda_wdur = lambda_wdur
        self.lambda_sdur = lambda_sdur
        self.lambda_alloc = lambda_alloc

    def linear2log(self, any_dur):
        return torch.log(any_dur + self.offset)

    def forward(self, dur_pred: Tensor, dur_gt: Tensor, ph2word: Tensor) -> Tensor:
        """Calculate the duration loss.

        Args:
            dur_pred (Tensor): Predicted durations (B, Tmax).
            dur_gt (Tensor): Ground-truth durations (B, Tmax), in the same unit.
            ph2word (Tensor): Word index of every phoneme (B, Tmax), 1-based,
                0 for padding.

        Returns:
            Tensor: Scalar loss.
        """
        dur_gt = dur_gt.to(dtype=dur_pred.dtype)

        # pdur_loss
        pdur_loss = self.lambda_pdur * self.loss(self.linear2log(dur_pred), self.linear2log(dur_gt))

        dur_pred = dur_pred.clamp(min=0.)  # clip to avoid NaN loss

        # allocation loss
        alloc_loss = 0.
        if self.lambda_alloc > 0.:
            # float32: under fp16 ``_EPS`` underflows, the clamp no-ops and a
            # zero-duration word divides by zero.
            prob_pred = word_distribution(dur_pred.float(), ph2word)
            prob_gt = word_distribution(dur_gt.float(), ph2word)
            token_loss = -(prob_gt * prob_pred.clamp_min(1e-8).log()) * (ph2word > 0)
            n_tokens = (ph2word > 0).sum().clamp_min(1)
            alloc_loss = self.lambda_alloc * token_loss.sum() / n_tokens

        # wdur loss
        shape = dur_pred.shape[0], ph2word.max() + 1
        wdur_pred = dur_pred.new_zeros(*shape).scatter_add(
            1, ph2word, dur_pred
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_gt = dur_gt.new_zeros(*shape).scatter_add(
            1, ph2word, dur_gt
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_loss = self.lambda_wdur * self.loss(self.linear2log(wdur_pred), self.linear2log(wdur_gt))

        # sdur loss
        sdur_pred = dur_pred.sum(dim=1)
        sdur_gt = dur_gt.sum(dim=1)
        sdur_loss = self.lambda_sdur * self.loss(self.linear2log(sdur_pred), self.linear2log(sdur_gt))

        # combine
        dur_loss = alloc_loss + pdur_loss + wdur_loss + sdur_loss

        return dur_loss


def build_duration_loss(dur_hparams: dict, word_budget_given: bool) -> DurationLoss:
    """Build the duration loss from a flat ``dur_prediction_args`` block.

    Coefficients live here so they cannot drift from the trainer that applies
    them:

    * The word and sentence terms are switched off when the predictor is given
      the frame budget of every word (it then reproduces both sums by
      construction, so the weights cannot matter); the configured values still
      apply to a predictor that predicts absolute durations itself.
    * The allocation term is enabled exactly when the predictor allocates.

    Args:
        dur_hparams: The ``dur_prediction_args`` block.
        word_budget_given: Whether the duration predictor consumes the per-word
            frame budget (true for ``arch == 'attn'``, false for the
            convolutional architectures that predict absolute durations).

    Returns:
        The loss module to train with.
    """
    return DurationLoss(
        offset=dur_hparams['log_offset'],
        loss_type=dur_hparams['loss_type'],
        lambda_pdur=dur_hparams['lambda_pdur_loss'],
        lambda_wdur=0. if word_budget_given else dur_hparams['lambda_wdur_loss'],
        lambda_sdur=0. if word_budget_given else dur_hparams['lambda_sdur_loss'],
        lambda_alloc=1. if word_budget_given else 0.,
    )
