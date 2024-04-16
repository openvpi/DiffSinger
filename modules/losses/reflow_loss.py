import torch
import torch.nn as nn
from torch import Tensor


class RectifiedFlowLoss(nn.Module):
    def __init__(self, loss_type, log_norm=True):
        super().__init__()
        self.loss_type = loss_type
        self.log_norm = log_norm
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_non_padding(v_pred, v_gt, non_padding=None):
        if non_padding is not None:
            non_padding = non_padding.transpose(1, 2).unsqueeze(1)
            return v_pred * non_padding, v_gt * non_padding
        else:
            return v_pred, v_gt

    @staticmethod
    def get_weights(t):
        eps = 1e-7
        t = t.float()
        t = torch.clip(t, 0 + eps, 1 - eps)
        weights = 0.398942 / t / (1 - t) * torch.exp(
            -0.5 * torch.log(t / (1 - t)) ** 2
        ) + eps
        return weights[:, None, None, None]

    def _forward(self, v_pred, v_gt, t=None):
        if self.log_norm:
            return self.get_weights(t) * self.loss(v_pred, v_gt)
        else:
            return self.loss(v_pred, v_gt)

    def forward(self, v_pred: Tensor, v_gt: Tensor, t: Tensor, non_padding: Tensor = None) -> Tensor:
        """
        :param v_pred: [B, 1, M, T]
        :param v_gt: [B, 1, M, T]
        :param t: [B,]
        :param non_padding: [B, T, M]
        """
        v_pred, v_gt = self._mask_non_padding(v_pred, v_gt, non_padding)
        return self._forward(v_pred, v_gt, t=t).mean()
