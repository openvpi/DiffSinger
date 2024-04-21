from __future__ import annotations

import torch

from modules.core.ddpm import MultiVarianceDiffusion
from utils import filter_kwargs
from utils.hparams import hparams

VARIANCE_CHECKLIST = ['energy', 'breathiness', 'voicing', 'tension']


class ParameterAdaptorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.variance_prediction_list = []
        self.predict_energy = hparams.get('predict_energy', False)
        self.predict_breathiness = hparams.get('predict_breathiness', False)
        self.predict_voicing = hparams.get('predict_voicing', False)
        self.predict_tension = hparams.get('predict_tension', False)
        if self.predict_energy:
            self.variance_prediction_list.append('energy')
        if self.predict_breathiness:
            self.variance_prediction_list.append('breathiness')
        if self.predict_voicing:
            self.variance_prediction_list.append('voicing')
        if self.predict_tension:
            self.variance_prediction_list.append('tension')
        self.predict_variances = len(self.variance_prediction_list) > 0

    def build_adaptor(self, cls=MultiVarianceDiffusion):
        ranges = []
        clamps = []

        if self.predict_energy:
            ranges.append((
                hparams['energy_db_min'],
                hparams['energy_db_max']
            ))
            clamps.append((hparams['energy_db_min'], 0.))

        if self.predict_breathiness:
            ranges.append((
                hparams['breathiness_db_min'],
                hparams['breathiness_db_max']
            ))
            clamps.append((hparams['breathiness_db_min'], 0.))

        if self.predict_voicing:
            ranges.append((
                hparams['voicing_db_min'],
                hparams['voicing_db_max']
            ))
            clamps.append((hparams['voicing_db_min'], 0.))

        if self.predict_tension:
            ranges.append((
                hparams['tension_logit_min'],
                hparams['tension_logit_max']
            ))
            clamps.append((
                hparams['tension_logit_min'],
                hparams['tension_logit_max']
            ))

        variances_hparams = hparams['variances_prediction_args']
        total_repeat_bins = variances_hparams['total_repeat_bins']
        assert total_repeat_bins % len(self.variance_prediction_list) == 0, \
            f'Total number of repeat bins must be divisible by number of ' \
            f'variance parameters ({len(self.variance_prediction_list)}).'
        repeat_bins = total_repeat_bins // len(self.variance_prediction_list)
        kwargs = filter_kwargs(
            {
                'ranges': ranges,
                'clamps': clamps,
                'repeat_bins': repeat_bins,
                'timesteps': hparams.get('timesteps'),
                'time_scale_factor': hparams.get('time_scale_factor'),
                'backbone_type': hparams.get('backbone_type', hparams.get('diff_decoder_type')),
                'backbone_args': {
                    'n_layers': variances_hparams['residual_layers'],
                    'n_chans': variances_hparams['residual_channels'],
                    'n_dilates': variances_hparams['dilation_cycle_length'],
                }
            },
            cls
        )
        return cls(**kwargs)

    def collect_variance_inputs(self, **kwargs) -> list:
        return [kwargs.get(name) for name in self.variance_prediction_list]

    def collect_variance_outputs(self, variances: list | tuple) -> dict:
        return {
            name: pred
            for name, pred in zip(self.variance_prediction_list, variances)
        }
