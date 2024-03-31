import torch.nn as nn
from torch import Tensor
import torch


class DiffusionNoiseLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        elif self.loss_type == 'l2_rf_norm':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_nonpadding(x_recon, noise, nonpadding=None):
        if nonpadding is not None:
            nonpadding = nonpadding.transpose(1, 2).unsqueeze(1)
            return x_recon * nonpadding, noise * nonpadding
        else:
            return x_recon, noise

    def l2_rf_norm(self, x_recon, noise, timestep):
        eps = 1e-8
        timestep = timestep.float()
        timestep=torch.clip(timestep, 0+eps, 1-eps)
        weights = 0.398942 / timestep / (1 - timestep) * torch.exp(
            -0.5 * torch.log(timestep / (1 - timestep)) ** 2) + eps
        return weights[:, None, None, None] * self.loss(x_recon, noise)

    def _forward(self, x_recon, noise, timestep=None):
        if self.loss_type == 'l2_rf_norm':
            return self.l2_rf_norm(x_recon, noise, timestep)
        return self.loss(x_recon, noise)

    def forward(self, x_recon: Tensor, noise: Tensor, timesteps: Tensor = None, nonpadding: Tensor = None) -> Tensor:
        """
        :param x_recon: [B, 1, M, T]
        :param noise: [B, 1, M, T]
        :param nonpadding: [B, T, M]
        """
        x_recon, noise = self._mask_nonpadding(x_recon, noise, nonpadding)
        return self._forward(x_recon, noise, timestep=timesteps).mean()
