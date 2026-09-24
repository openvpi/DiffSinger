import torch
import torch.nn as nn
from torch import Tensor

from modules.duration.word_groups import word_distribution


class DurationLoss(nn.Module):
    """
    Loss module as combination of phone duration loss, word allocation loss,
    word duration loss and sentence duration loss.

    The allocation term treats the phones of a word (a note or syllable) as a
    distribution over the frame budget of that word and compares it with the
    target distribution with a cross entropy. It is scale invariant inside a
    word, so the word and sentence terms are the ones that keep the absolute
    frame scale meaningful, and they should stay non-zero as long as the model
    has to predict that scale by itself. When the model is handed the frame
    budget of every word instead (``use_allocation`` in the duration predictor),
    its output already carries the right word and sentence sums, so those two
    terms become constant and only the phone and allocation terms carry
    gradient.
    """

    def __init__(self, offset, loss_type,
                 lambda_pdur=0.6, lambda_wdur=0.3, lambda_sdur=0.1, lambda_alloc=0.0):
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
        dur_gt = dur_gt.to(dtype=dur_pred.dtype)

        # pdur_loss
        pdur_loss = self.lambda_pdur * self.loss(self.linear2log(dur_pred), self.linear2log(dur_gt))

        dur_pred = dur_pred.clamp(min=0.)  # clip to avoid NaN loss

        # allocation loss
        alloc_loss = 0.
        if self.lambda_alloc > 0.:
            prob_pred = word_distribution(dur_pred, ph2word)
            prob_gt = word_distribution(dur_gt, ph2word)
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
