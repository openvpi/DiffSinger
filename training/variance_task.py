import matplotlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from modules.losses.diff_loss import DiffusionNoiseLoss
from modules.losses.dur_loss import DurationLoss
from modules.toplevel import DiffSingerVariance
from utils.hparams import hparams
from utils.plot import dur_to_figure, curve_to_figure

matplotlib.use('Agg')


class VarianceDataset(BaseDataset):
    def collater(self, samples):
        batch = super().collater(samples)

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        ph_dur = utils.collate_nd([s['ph_dur'] for s in samples], 0)
        midi = utils.collate_nd([s['midi'] for s in samples], 0)
        ph2word = utils.collate_nd([s['ph2word'] for s in samples], 0)
        mel2ph = utils.collate_nd([s['mel2ph'] for s in samples], 0)
        base_pitch = utils.collate_nd([s['base_pitch'] for s in samples], 0)
        delta_pitch = utils.collate_nd([s['delta_pitch'] for s in samples], 0)
        batch.update({
            'tokens': tokens,
            'ph_dur': ph_dur,
            'midi': midi,
            'ph2word': ph2word,
            'mel2ph': mel2ph,
            'base_pitch': base_pitch,
            'delta_pitch': delta_pitch,
        })
        if hparams['predict_energy']:
            energy = utils.collate_nd([s['energy'] for s in samples], 0)
            batch['energy'] = energy
        if hparams['predict_breathiness']:
            breathiness = utils.collate_nd([s['breathiness'] for s in samples], 0)
            batch['breathiness'] = breathiness
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids

        return batch


class VarianceTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = VarianceDataset

        self.predict_dur = hparams['predict_dur']
        if self.predict_dur:
            self.lambda_dur_loss = hparams['lambda_dur_loss']

        self.predict_pitch = hparams['predict_pitch']
        if self.predict_pitch:
            self.lambda_pitch_loss = hparams['lambda_pitch_loss']

        predict_energy = hparams['predict_energy']
        predict_breathiness = hparams['predict_breathiness']
        self.variance_prediction_list = []
        if predict_energy:
            self.variance_prediction_list.append('energy')
        if predict_breathiness:
            self.variance_prediction_list.append('breathiness')
        self.predict_variances = len(self.variance_prediction_list) > 0
        self.lambda_var_loss = hparams['lambda_var_loss']

    def build_model(self):
        return DiffSingerVariance(
            vocab_size=len(self.phone_encoder),
        )

    # noinspection PyAttributeOutsideInit
    def build_losses(self):
        if self.predict_dur:
            dur_hparams = hparams['dur_prediction_args']
            self.dur_loss = DurationLoss(
                offset=dur_hparams['log_offset'],
                loss_type=dur_hparams['loss_type'],
                lambda_pdur=dur_hparams['lambda_pdur_loss'],
                lambda_wdur=dur_hparams['lambda_wdur_loss'],
                lambda_sdur=dur_hparams['lambda_sdur_loss']
            )
        if self.predict_pitch:
            self.pitch_loss = DiffusionNoiseLoss(
                loss_type=hparams['diff_loss_type'],
            )
        if self.predict_variances:
            self.var_loss = DiffusionNoiseLoss(
                loss_type=hparams['diff_loss_type'],
            )

    def run_model(self, sample, infer=False):
        txt_tokens = sample['tokens']  # [B, T_ph]
        midi = sample['midi']  # [B, T_ph]
        ph2word = sample['ph2word']  # [B, T_ph]
        ph_dur = sample['ph_dur']  # [B, T_ph]
        mel2ph = sample['mel2ph']  # [B, T_t]
        base_pitch = sample['base_pitch']  # [B, T_t]
        delta_pitch = sample['delta_pitch']  # [B, T_t]
        energy = sample.get('energy')  # [B, T_t]
        breathiness = sample.get('breathiness')  # [B, T_t]

        if (self.predict_pitch or self.predict_variances) and not infer:
            # randomly select continuous retaking regions
            b = sample['size']
            t = mel2ph.shape[1]
            device = mel2ph.device
            start, end = torch.sort(
                torch.randint(low=0, high=t + 1, size=(b, 2), device=device), dim=1
            )[0].split(1, dim=1)
            idx = torch.arange(0, t, dtype=torch.long, device=device)[None]
            retake = (idx >= start) & (idx < end)
        else:
            retake = None

        output = self.model(
            txt_tokens, midi=midi, ph2word=ph2word, ph_dur=ph_dur, mel2ph=mel2ph,
            base_pitch=base_pitch, delta_pitch=delta_pitch,
            energy=energy, breathiness=breathiness,
            retake=retake, infer=infer
        )

        dur_pred, pitch_pred, variances_pred = output
        if infer:
            if dur_pred is not None:
                dur_pred = dur_pred.round().long()
            return dur_pred, pitch_pred, variances_pred  # Tensor, Tensor, Dict[Tensor]
        else:
            losses = {}
            if dur_pred is not None:
                losses['dur_loss'] = self.lambda_dur_loss * self.dur_loss(dur_pred, ph_dur, ph2word=ph2word)
            nonpadding = (mel2ph > 0).unsqueeze(-1).float()
            if pitch_pred is not None:
                (pitch_x_recon, pitch_noise) = pitch_pred
                losses['pitch_loss'] = self.lambda_pitch_loss * self.pitch_loss(
                    pitch_x_recon, pitch_noise, nonpadding=nonpadding
                )
            if variances_pred is not None:
                (variance_x_recon, variance_noise) = variances_pred
                losses['var_loss'] = self.lambda_var_loss * self.var_loss(
                    variance_x_recon, variance_noise, nonpadding=nonpadding
                )
            return losses

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        total_loss = sum(losses.values())
        outputs = {
            'total_loss': total_loss
        }

        if batch_idx < hparams['num_valid_plots'] \
                and (self.trainer.distributed_sampler_kwargs or {}).get('rank', 0) == 0:
            dur_pred, pitch_pred, variances_pred = self.run_model(sample, infer=True)
            if dur_pred is not None:
                self.plot_dur(batch_idx, sample['ph_dur'], dur_pred, txt=sample['tokens'])
            if pitch_pred is not None:
                base_pitch = sample['base_pitch']
                delta_pitch = sample['delta_pitch']
                self.plot_curve(
                    batch_idx,
                    gt_curve=base_pitch + delta_pitch,
                    pred_curve=base_pitch + pitch_pred,
                    base_curve=base_pitch,
                    curve_name='pitch',
                    grid=1
                )
            for name in self.variance_prediction_list:
                variance = sample[name]
                variance_pred = variances_pred[name]
                self.plot_curve(
                    batch_idx,
                    gt_curve=variance,
                    pred_curve=variance_pred,
                    curve_name=name
                )

        return outputs, sample['size']

    ############
    # validation plots
    ############
    def plot_dur(self, batch_idx, gt_dur, pred_dur, txt=None):
        name = f'dur_{batch_idx}'
        gt_dur = gt_dur[0].cpu().numpy()
        pred_dur = pred_dur[0].cpu().numpy()
        txt = self.phone_encoder.decode(txt[0].cpu().numpy()).split()
        self.logger.experiment.add_figure(name, dur_to_figure(gt_dur, pred_dur, txt), self.global_step)

    def plot_curve(self, batch_idx, gt_curve, pred_curve, base_curve=None, grid=None, curve_name='curve'):
        name = f'{curve_name}_{batch_idx}'
        gt_curve = gt_curve[0].cpu().numpy()
        pred_curve = pred_curve[0].cpu().numpy()
        if base_curve is not None:
            base_curve = base_curve[0].cpu().numpy()
        self.logger.experiment.add_figure(name, curve_to_figure(
            gt_curve, pred_curve, base_curve, grid=grid
        ), self.global_step)
