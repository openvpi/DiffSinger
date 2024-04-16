import torch.nn as nn
from torch import Tensor


class DiffusionLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_non_padding(x_recon, noise, non_padding=None):
        if non_padding is not None:
            non_padding = non_padding.transpose(1, 2).unsqueeze(1)
            return x_recon * non_padding, noise * non_padding
        else:
            return x_recon, noise

    def _forward(self, x_recon, noise):
        return self.loss(x_recon, noise)

    def forward(self, x_recon: Tensor, noise: Tensor, non_padding: Tensor = None) -> Tensor:
        """
        :param x_recon: [B, 1, M, T]
        :param noise: [B, 1, M, T]
        :param non_padding: [B, T, M]
        """
        x_recon, noise = self._mask_non_padding(x_recon, noise, non_padding)
        return self._forward(x_recon, noise).mean()
