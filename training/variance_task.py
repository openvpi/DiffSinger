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
from modules.metrics.curve import RawCurveAccuracy
from modules.metrics.duration import RhythmCorrectness, PhonemeDurationAccuracy
from modules.toplevel import DiffSingerVariance
from utils.hparams import hparams
from utils.plot import dur_to_figure, pitch_note_to_figure, curve_to_figure

matplotlib.use('Agg')


class VarianceDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        need_energy = hparams['predict_energy']
        need_breathiness = hparams['predict_breathiness']
        need_tension = hparams['predict_tension']
        self.predict_variances = need_energy or need_breathiness or need_tension

    def collater(self, samples):
        batch = super().collater(samples)

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        ph_dur = utils.collate_nd([s['ph_dur'] for s in samples], 0)
        batch.update({
            'tokens': tokens,
            'ph_dur': ph_dur
        })

        if hparams['use_spk_id']:
            batch['spk_ids'] = torch.LongTensor([s['spk_id'] for s in samples])
        if hparams['predict_dur']:
            batch['ph2word'] = utils.collate_nd([s['ph2word'] for s in samples], 0)
            batch['midi'] = utils.collate_nd([s['midi'] for s in samples], 0)
        if hparams['predict_pitch']:
            batch['note_midi'] = utils.collate_nd([s['note_midi'] for s in samples], -1)
            batch['note_rest'] = utils.collate_nd([s['note_rest'] for s in samples], True)
            batch['note_dur'] = utils.collate_nd([s['note_dur'] for s in samples], 0)
            if hparams['use_glide_embed']:
                batch['note_glide'] = utils.collate_nd([s['note_glide'] for s in samples], 0)
            batch['mel2note'] = utils.collate_nd([s['mel2note'] for s in samples], 0)
            batch['base_pitch'] = utils.collate_nd([s['base_pitch'] for s in samples], 0)
        if hparams['predict_pitch'] or self.predict_variances:
            batch['mel2ph'] = utils.collate_nd([s['mel2ph'] for s in samples], 0)
            batch['pitch'] = utils.collate_nd([s['pitch'] for s in samples], 0)
            batch['uv'] = utils.collate_nd([s['uv'] for s in samples], True)
        if hparams['predict_energy']:
            batch['energy'] = utils.collate_nd([s['energy'] for s in samples], 0)
        if hparams['predict_breathiness']:
            batch['breathiness'] = utils.collate_nd([s['breathiness'] for s in samples], 0)
        if hparams['predict_tension']:
            batch['tension'] = utils.collate_nd([s['tension'] for s in samples], 0)

        return batch


def random_retake_masks(b, t, device):
    # 1/4 segments are True in average
    B_masks = torch.randint(low=0, high=4, size=(b, 1), dtype=torch.long, device=device) == 0
    # 1/3 frames are True in average
    T_masks = utils.random_continuous_masks(b, t, dim=1, device=device)
    # 1/4 segments and 1/2 frames are True in average (1/4 + 3/4 * 1/3 = 1/2)
    return B_masks | T_masks


class VarianceTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = VarianceDataset

        self.use_spk_id = hparams['use_spk_id']

        self.predict_dur = hparams['predict_dur']
        if self.predict_dur:
            self.lambda_dur_loss = hparams['lambda_dur_loss']

        self.predict_pitch = hparams['predict_pitch']
        if self.predict_pitch:
            self.lambda_pitch_loss = hparams['lambda_pitch_loss']

        predict_energy = hparams['predict_energy']
        predict_breathiness = hparams['predict_breathiness']
        predict_tension = hparams['predict_tension']
        self.variance_prediction_list = []
        if predict_energy:
            self.variance_prediction_list.append('energy')
        if predict_breathiness:
            self.variance_prediction_list.append('breathiness')
        if predict_tension:
            self.variance_prediction_list.append('tension')
        self.predict_variances = len(self.variance_prediction_list) > 0
        self.lambda_var_loss = hparams['lambda_var_loss']

    def build_model(self):
        return DiffSingerVariance(
            vocab_size=len(self.phone_encoder),
        )

    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        if self.predict_dur:
            dur_hparams = hparams['dur_prediction_args']
            self.dur_loss = DurationLoss(
                offset=dur_hparams['log_offset'],
                loss_type=dur_hparams['loss_type'],
                lambda_pdur=dur_hparams['lambda_pdur_loss'],
                lambda_wdur=dur_hparams['lambda_wdur_loss'],
                lambda_sdur=dur_hparams['lambda_sdur_loss']
            )
            self.register_metric('rhythm_corr', RhythmCorrectness(tolerance=0.05))
            self.register_metric('ph_dur_acc', PhonemeDurationAccuracy(tolerance=0.2))
        if self.predict_pitch:
            self.pitch_loss = DiffusionNoiseLoss(
                loss_type=hparams['diff_loss_type'],
            )
            self.register_metric('pitch_acc', RawCurveAccuracy(tolerance=0.5))
        if self.predict_variances:
            self.var_loss = DiffusionNoiseLoss(
                loss_type=hparams['diff_loss_type'],
            )

    def run_model(self, sample, infer=False):
        spk_ids = sample['spk_ids'] if self.use_spk_id else None  # [B,]
        txt_tokens = sample['tokens']  # [B, T_ph]
        ph_dur = sample['ph_dur']  # [B, T_ph]
        ph2word = sample.get('ph2word')  # [B, T_ph]
        midi = sample.get('midi')  # [B, T_ph]
        mel2ph = sample.get('mel2ph')  # [B, T_s]

        note_midi = sample.get('note_midi')  # [B, T_n]
        note_rest = sample.get('note_rest')  # [B, T_n]
        note_dur = sample.get('note_dur')  # [B, T_n]
        note_glide = sample.get('note_glide')  # [B, T_n]
        mel2note = sample.get('mel2note')  # [B, T_s]

        base_pitch = sample.get('base_pitch')  # [B, T_s]
        pitch = sample.get('pitch')  # [B, T_s]
        energy = sample.get('energy')  # [B, T_s]
        breathiness = sample.get('breathiness')  # [B, T_s]
        tension = sample.get('tension')  # [B, T_s]

        pitch_retake = variance_retake = None
        if (self.predict_pitch or self.predict_variances) and not infer:
            # randomly select continuous retaking regions
            b = sample['size']
            t = mel2ph.shape[1]
            device = mel2ph.device
            if self.predict_pitch:
                pitch_retake = random_retake_masks(b, t, device)
            if self.predict_variances:
                variance_retake = {
                    v_name: random_retake_masks(b, t, device)
                    for v_name in self.variance_prediction_list
                }

        output = self.model(
            txt_tokens, midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, mel2ph=mel2ph,
            note_midi=note_midi, note_rest=note_rest,
            note_dur=note_dur, note_glide=note_glide, mel2note=mel2note,
            base_pitch=base_pitch, pitch=pitch,
            energy=energy, breathiness=breathiness, tension=tension,
            pitch_retake=pitch_retake, variance_retake=variance_retake,
            spk_id=spk_ids, infer=infer
        )

        dur_pred, pitch_pred, variances_pred = output
        if infer:
            if dur_pred is not None:
                dur_pred = dur_pred.round().long()
            return dur_pred, pitch_pred, variances_pred  # Tensor, Tensor, Dict[str, Tensor]
        else:
            losses = {}
            if dur_pred is not None:
                losses['dur_loss'] = self.lambda_dur_loss * self.dur_loss(dur_pred, ph_dur, ph2word=ph2word)
            nonpadding = (mel2ph > 0).unsqueeze(-1) if mel2ph is not None else None
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

        if batch_idx < hparams['num_valid_plots'] \
                and (self.trainer.distributed_sampler_kwargs or {}).get('rank', 0) == 0:
            dur_pred, pitch_pred, variances_pred = self.run_model(sample, infer=True)
            if dur_pred is not None:
                tokens = sample['tokens']
                dur_gt = sample['ph_dur']
                ph2word = sample['ph2word']
                mask = tokens != 0
                self.rhythm_corr.update(
                    pdur_pred=dur_pred, pdur_target=dur_gt, ph2word=ph2word, mask=mask
                )
                self.ph_dur_acc.update(
                    pdur_pred=dur_pred, pdur_target=dur_gt, ph2word=ph2word, mask=mask
                )
                self.plot_dur(batch_idx, dur_gt, dur_pred, txt=tokens)
            if pitch_pred is not None:
                pred_pitch = sample['base_pitch'] + pitch_pred
                gt_pitch = sample['pitch']
                mask = (sample['mel2ph'] > 0) & ~sample['uv']
                self.pitch_acc.update(pred=pred_pitch, target=gt_pitch, mask=mask)
                self.plot_pitch(
                    batch_idx,
                    gt_pitch=gt_pitch,
                    pred_pitch=pred_pitch,
                    note_midi=sample['note_midi'],
                    note_dur=sample['note_dur'],
                    note_rest=sample['note_rest']
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

        return losses, sample['size']

    ############
    # validation plots
    ############
    def plot_dur(self, batch_idx, gt_dur, pred_dur, txt=None):
        name = f'dur_{batch_idx}'
        gt_dur = gt_dur[0].cpu().numpy()
        pred_dur = pred_dur[0].cpu().numpy()
        txt = self.phone_encoder.decode(txt[0].cpu().numpy()).split()
        self.logger.experiment.add_figure(name, dur_to_figure(gt_dur, pred_dur, txt), self.global_step)

    def plot_pitch(self, batch_idx, gt_pitch, pred_pitch, note_midi, note_dur, note_rest):
        name = f'pitch_{batch_idx}'
        gt_pitch = gt_pitch[0].cpu().numpy()
        pred_pitch = pred_pitch[0].cpu().numpy()
        note_midi = note_midi[0].cpu().numpy()
        note_dur = note_dur[0].cpu().numpy()
        note_rest = note_rest[0].cpu().numpy()
        self.logger.experiment.add_figure(name, pitch_note_to_figure(
            gt_pitch, pred_pitch, note_midi, note_dur, note_rest
        ), self.global_step)

    def plot_curve(self, batch_idx, gt_curve, pred_curve, base_curve=None, grid=None, curve_name='curve'):
        name = f'{curve_name}_{batch_idx}'
        gt_curve = gt_curve[0].cpu().numpy()
        pred_curve = pred_curve[0].cpu().numpy()
        if base_curve is not None:
            base_curve = base_curve[0].cpu().numpy()
        self.logger.experiment.add_figure(name, curve_to_figure(
            gt_curve, pred_curve, base_curve, grid=grid
        ), self.global_step)
