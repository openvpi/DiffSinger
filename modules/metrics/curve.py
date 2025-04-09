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


class RawCurveR2Score(torchmetrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('sum_squared_error', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('sum_error', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('residual', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

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
            pred = pred[mask]
            target = target[mask]
        pred = pred.flatten()
        target = target.flatten()

        sum_error = torch.sum(target)
        sum_squared_error = torch.sum(target * target)
        residual = target - pred
        rss = torch.sum(residual * residual)
        total = target.numel() if mask is None else mask.sum()

        self.sum_squared_error += sum_squared_error
        self.sum_error += sum_error
        self.residual += rss
        self.total += total

    def compute(self) -> Tensor:
        return 1 - self.residual / (self.sum_squared_error - self.sum_error ** 2 / self.total)
