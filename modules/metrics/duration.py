import torch
import torchmetrics
from torch import Tensor

from modules.commons.tts_modules import RhythmRegulator


def linguistic_checks(pred, target, ph2word, mask=None):
    if mask is None:
        assert pred.shape == target.shape == ph2word.shape, \
            f'shapes of pred, target and ph2word mismatch: {pred.shape}, {target.shape}, {ph2word.shape}'
    else:
        assert pred.shape == target.shape == ph2word.shape == mask.shape, \
            f'shapes of pred, target and mask mismatch: {pred.shape}, {target.shape}, {ph2word.shape}, {mask.shape}'
    assert pred.ndim == 2, f'all inputs should be 2D, but got {pred.shape}'
    assert torch.any(ph2word > 0), 'empty word sequence'
    assert torch.all(ph2word >= 0), 'unexpected negative word index'
    assert ph2word.max() <= pred.shape[1], f'word index out of range: {ph2word.max()} > {pred.shape[1]}'
    assert torch.all(pred >= 0.), f'unexpected negative ph_dur prediction'
    assert torch.all(target >= 0.), f'unexpected negative ph_dur target'


class RhythmCorrectness(torchmetrics.Metric):
    def __init__(self, *, tolerance, **kwargs):
        super().__init__(**kwargs)
        assert 0. < tolerance < 1., 'tolerance should be within (0, 1)'
        self.tolerance = tolerance
        self.add_state('correct', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        """

        :param pdur_pred: predicted ph_dur
        :param pdur_target: reference ph_dur
        :param ph2word: word division sequence
        :param mask: valid or non-padding mask
        """
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)

        shape = pdur_pred.shape[0], ph2word.max() + 1
        wdur_pred = pdur_pred.new_zeros(*shape).scatter_add(
            1, ph2word, pdur_pred
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_target = pdur_target.new_zeros(*shape).scatter_add(
            1, ph2word, pdur_target
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        if mask is None:
            wdur_mask = torch.ones_like(wdur_pred, dtype=torch.bool)
        else:
            wdur_mask = mask.new_zeros(*shape).scatter_add(
                1, ph2word, mask
            )[:, 1:].bool()  # [B, T_ph] => [B, T_w]

        correct = torch.abs(wdur_pred - wdur_target) <= wdur_target * self.tolerance
        correct &= wdur_mask

        self.correct += correct.sum()
        self.total += wdur_mask.sum()

    def compute(self) -> Tensor:
        return self.correct / self.total


class PhonemeDurationAccuracy(torchmetrics.Metric):
    def __init__(self, *, tolerance, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.rr = RhythmRegulator()
        self.add_state('accurate', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        """

        :param pdur_pred: predicted ph_dur
        :param pdur_target: reference ph_dur
        :param ph2word: word division sequence
        :param mask: valid or non-padding mask
        """
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)

        shape = pdur_pred.shape[0], ph2word.max() + 1
        wdur_target = pdur_target.new_zeros(*shape).scatter_add(
            1, ph2word, pdur_target
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        pdur_align = self.rr(pdur_pred, ph2word=ph2word, word_dur=wdur_target)

        accurate = torch.abs(pdur_align - pdur_target) <= pdur_target * self.tolerance
        if mask is not None:
            accurate &= mask

        self.accurate += accurate.sum()
        self.total += pdur_pred.numel() if mask is None else mask.sum()

    def compute(self) -> Tensor:
        return self.accurate / self.total
