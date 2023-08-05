import torch
import torchmetrics
from torch import Tensor


class RawCurveAccuracy(torchmetrics.Metric):
    def __init__(self, *, tolerance, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.add_state('close', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')

    def update(self, pred: Tensor, target: Tensor, mask=None) -> None:
        """

        :param pred: predicted curve
        :param target: reference curve
        :param mask: valid or non-padding mask
        """
        if mask is None:
            assert pred.shape == target.shape, f'shapes of pred and target mismatch: {pred.shape}, {target.shape}'
        else:
            assert pred.shape == target.shape == mask.shape, \
                f'shapes of pred, target and mask mismatch: {pred.shape}, {target.shape}, {mask.shape}'
        close = torch.abs(pred - target) <= self.tolerance
        if mask is not None:
            close &= mask

        self.close += close.sum()
        self.total += pred.numel() if mask is None else mask.sum()

    def compute(self) -> Tensor:
        return self.close / self.total
