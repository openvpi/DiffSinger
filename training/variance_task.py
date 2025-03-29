import matplotlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from modules.losses import DurationLoss, DiffusionLoss, RectifiedFlowLoss
from modules.metrics import (
    RawCurveAccuracy, RawCurveR2Score, RhythmCorrectness, PhonemeDurationAccuracy
)
from modules.toplevel import DiffSingerVariance
from utils.plot import dur_to_figure, pitch_note_to_figure, curve_to_figure

matplotlib.use('Agg')


class VarianceDataset(BaseDataset):
    def __init__(self, config, prefix, preload=False):
        super(VarianceDataset, self).__init__(config, prefix, config['dataset_size_key'], preload)
        need_energy = config['predict_energy']
        need_breathiness = config['predict_breathiness']
        need_voicing = config['predict_voicing']
        need_tension = config['predict_tension']
        self.predict_variances = need_energy or need_breathiness or need_voicing or need_tension

    def collater(self, samples):
        batch = super().collater(samples)
        if batch['size'] == 0:
            return batch

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        ph_dur = utils.collate_nd([s['ph_dur'] for s in samples], 0)
        batch.update({
            'tokens': tokens,
            'ph_dur': ph_dur
        })

        if self.config['use_spk_id']:
            batch['spk_ids'] = torch.LongTensor([s['spk_id'] for s in samples])
        if self.config['use_lang_id']:
            batch['languages'] = utils.collate_nd([s['languages'] for s in samples], 0)
        if self.config['predict_dur']:
            batch['ph2word'] = utils.collate_nd([s['ph2word'] for s in samples], 0)
            batch['midi'] = utils.collate_nd([s['midi'] for s in samples], 0)
        if self.config['predict_pitch']:
            batch['note_midi'] = utils.collate_nd([s['note_midi'] for s in samples], -1)
            batch['note_rest'] = utils.collate_nd([s['note_rest'] for s in samples], True)
            batch['note_dur'] = utils.collate_nd([s['note_dur'] for s in samples], 0)
            if self.config['use_glide_embed']:
                batch['note_glide'] = utils.collate_nd([s['note_glide'] for s in samples], 0)
            batch['mel2note'] = utils.collate_nd([s['mel2note'] for s in samples], 0)
            batch['base_pitch'] = utils.collate_nd([s['base_pitch'] for s in samples], 0)
        if self.config['predict_pitch'] or self.predict_variances:
            batch['mel2ph'] = utils.collate_nd([s['mel2ph'] for s in samples], 0)
            batch['pitch'] = utils.collate_nd([s['pitch'] for s in samples], 0)
            batch['uv'] = utils.collate_nd([s['uv'] for s in samples], True)
        if self.config['predict_energy']:
            batch['energy'] = utils.collate_nd([s['energy'] for s in samples], 0)
        if self.config['predict_breathiness']:
            batch['breathiness'] = utils.collate_nd([s['breathiness'] for s in samples], 0)
        if self.config['predict_voicing']:
            batch['voicing'] = utils.collate_nd([s['voicing'] for s in samples], 0)
        if self.config['predict_tension']:
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
    def __init__(self, config):
        super().__init__(config)
        self.dataset_cls = VarianceDataset

        self.diffusion_type = config['diffusion_type']

        self.use_spk_id = config['use_spk_id']
        self.use_lang_id = config['use_lang_id']

        self.predict_dur = config['predict_dur']
        if self.predict_dur:
            self.lambda_dur_loss = config['lambda_dur_loss']

        self.predict_pitch = config['predict_pitch']
        if self.predict_pitch:
            self.lambda_pitch_loss = config['lambda_pitch_loss']

        predict_energy = config['predict_energy']
        predict_breathiness = config['predict_breathiness']
        predict_voicing = config['predict_voicing']
        predict_tension = config['predict_tension']
        self.variance_prediction_list = []
        if predict_energy:
            self.variance_prediction_list.append('energy')
        if predict_breathiness:
            self.variance_prediction_list.append('breathiness')
        if predict_voicing:
            self.variance_prediction_list.append('voicing')
        if predict_tension:
            self.variance_prediction_list.append('tension')
        self.predict_variances = len(self.variance_prediction_list) > 0
        self.lambda_var_loss = config['lambda_var_loss']
        super()._finish_init()

    def _build_model(self):
        return DiffSingerVariance(
            config=self.config,
            vocab_size=len(self.phoneme_dictionary),
        )

    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        if self.predict_dur:
            dur_hparams = self.config['dur_prediction_args']
            self.dur_loss = DurationLoss(
                offset=dur_hparams['log_offset'],
                loss_type=dur_hparams['loss_type'],
                lambda_pdur=dur_hparams['lambda_pdur_loss'],
                lambda_wdur=dur_hparams['lambda_wdur_loss'],
                lambda_sdur=dur_hparams['lambda_sdur_loss']
            )
            self.register_validation_loss('dur_loss')
            self.register_validation_metric('rhythm_corr', RhythmCorrectness(tolerance=0.05))
            self.register_validation_metric('ph_dur_acc', PhonemeDurationAccuracy(tolerance=0.2))
        if self.predict_pitch:
            if self.diffusion_type == 'ddpm':
                self.pitch_loss = DiffusionLoss(loss_type=self.config['main_loss_type'])
            elif self.diffusion_type == 'reflow':
                self.pitch_loss = RectifiedFlowLoss(
                    loss_type=self.config['main_loss_type'], log_norm=self.config['main_loss_log_norm']
                )
            else:
                raise ValueError(f'Unknown diffusion type: {self.diffusion_type}')
            self.register_validation_loss('pitch_loss')
            self.register_validation_metric('pitch_acc', RawCurveAccuracy(tolerance=0.5))
            self.register_validation_metric('pitch_r2', RawCurveR2Score())
        if self.predict_variances:
            if self.diffusion_type == 'ddpm':
                self.var_loss = DiffusionLoss(loss_type=self.config['main_loss_type'])
            elif self.diffusion_type == 'reflow':
                self.var_loss = RectifiedFlowLoss(
                    loss_type=self.config['main_loss_type'], log_norm=self.config['main_loss_log_norm']
                )
            else:
                raise ValueError(f'Unknown diffusion type: {self.diffusion_type}')
            self.register_validation_loss('var_loss')
            for name in self.variance_prediction_list:
                self.register_validation_metric(f'{name}_r2', RawCurveR2Score())

    def run_model(self, sample, infer=False):
        spk_ids = sample['spk_ids'] if self.use_spk_id else None  # [B,]
        languages = sample['languages'] if self.use_lang_id else None  # [B,]
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
        voicing = sample.get('voicing')  # [B, T_s]
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
            txt_tokens, languages=languages,
            midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, mel2ph=mel2ph,
            note_midi=note_midi, note_rest=note_rest,
            note_dur=note_dur, note_glide=note_glide, mel2note=mel2note,
            base_pitch=base_pitch, pitch=pitch,
            energy=energy, breathiness=breathiness, voicing=voicing, tension=tension,
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
            non_padding = (mel2ph > 0).unsqueeze(-1) if mel2ph is not None else None
            if pitch_pred is not None:
                if self.diffusion_type == 'ddpm':
                    pitch_x_recon, pitch_noise = pitch_pred
                    pitch_loss = self.pitch_loss(
                        pitch_x_recon, pitch_noise, non_padding=non_padding
                    )
                elif self.diffusion_type == 'reflow':
                    pitch_v_pred, pitch_v_gt, t = pitch_pred
                    pitch_loss = self.pitch_loss(
                        pitch_v_pred, pitch_v_gt, t=t, non_padding=non_padding
                    )
                else:
                    raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")
                losses['pitch_loss'] = self.lambda_pitch_loss * pitch_loss
            if variances_pred is not None:
                if self.diffusion_type == 'ddpm':
                    var_x_recon, var_noise = variances_pred
                    var_loss = self.var_loss(
                        var_x_recon, var_noise, non_padding=non_padding
                    )
                elif self.diffusion_type == 'reflow':
                    var_v_pred, var_v_gt, t = variances_pred
                    var_loss = self.var_loss(
                        var_v_pred, var_v_gt, t=t, non_padding=non_padding
                    )
                else:
                    raise ValueError(f"Unknown diffusion type: {self.diffusion_type}")
                losses['var_loss'] = self.lambda_var_loss * var_loss

            return losses

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)
        if min(sample['indices']) < self.config['num_valid_plots']:
            def sample_get(key, idx, abs_idx):
                return sample[key][idx][:self.valid_dataset.metadata[key][abs_idx]].unsqueeze(0)

            dur_preds, pitch_preds, variances_preds = self.run_model(sample, infer=True)
            for i in range(len(sample['indices'])):
                data_idx = sample['indices'][i]
                if data_idx < self.config['num_valid_plots']:
                    if dur_preds is not None:
                        dur_len = self.valid_dataset.metadata['ph_dur'][data_idx]
                        tokens = sample_get('tokens', i, data_idx)
                        gt_dur = sample_get('ph_dur', i, data_idx)
                        pred_dur = dur_preds[i][:dur_len].unsqueeze(0)
                        ph2word = sample_get('ph2word', i, data_idx)
                        mask = tokens != 0
                        self.valid_metrics['rhythm_corr'].update(
                            pdur_pred=pred_dur, pdur_target=gt_dur, ph2word=ph2word, mask=mask
                        )
                        self.valid_metrics['ph_dur_acc'].update(
                            pdur_pred=pred_dur, pdur_target=gt_dur, ph2word=ph2word, mask=mask
                        )
                        self.plot_dur(
                            data_idx, gt_dur, pred_dur,
                            txt=self.valid_dataset.metadata['ph_texts'][data_idx].split()
                        )
                    if pitch_preds is not None:
                        pitch_len = self.valid_dataset.metadata['pitch'][data_idx]
                        pred_pitch = sample_get('base_pitch', i, data_idx) + pitch_preds[i][:pitch_len].unsqueeze(0)
                        gt_pitch = sample_get('pitch', i, data_idx)
                        mask = (sample_get('mel2ph', i, data_idx) > 0) & ~sample_get('uv', i, data_idx)
                        self.valid_metrics['pitch_acc'].update(pred=pred_pitch, target=gt_pitch, mask=mask)
                        self.plot_pitch(
                            data_idx,
                            gt_pitch=gt_pitch,
                            pred_pitch=pred_pitch,
                            note_midi=sample_get('note_midi', i, data_idx),
                            note_dur=sample_get('note_dur', i, data_idx),
                            note_rest=sample_get('note_rest', i, data_idx)
                        )
                    for name in self.variance_prediction_list:
                        variance_len = self.valid_dataset.metadata[name][data_idx]
                        gt_variances = sample[name][i][:variance_len].unsqueeze(0)
                        pred_variances = variances_preds[name][i][:variance_len].unsqueeze(0)
                        mask = (sample_get('mel2ph', i, data_idx) > 0) & ~sample_get('uv', i, data_idx)
                        self.valid_metrics[f'{name}_r2'].update(pred=pred_variances, target=gt_variances, mask=mask)
                        self.plot_curve(
                            data_idx,
                            gt_curve=gt_variances,
                            pred_curve=pred_variances,
                            curve_name=name
                        )
        return losses, sample['size']

    ############
    # validation plots
    ############
    def plot_dur(self, data_idx, gt_dur, pred_dur, txt=None):
        gt_dur = gt_dur[0].cpu().numpy()
        pred_dur = pred_dur[0].cpu().numpy()
        title_text = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(f'dur_{data_idx}', dur_to_figure(
            gt_dur, pred_dur, txt, title_text
        ), self.global_step)

    def plot_pitch(self, data_idx, gt_pitch, pred_pitch, note_midi, note_dur, note_rest):
        gt_pitch = gt_pitch[0].cpu().numpy()
        pred_pitch = pred_pitch[0].cpu().numpy()
        note_midi = note_midi[0].cpu().numpy()
        note_dur = note_dur[0].cpu().numpy()
        note_rest = note_rest[0].cpu().numpy()
        title_text = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(f'pitch_{data_idx}', pitch_note_to_figure(
            gt_pitch, pred_pitch, note_midi, note_dur, note_rest, title_text
        ), self.global_step)

    def plot_curve(self, data_idx, gt_curve, pred_curve, base_curve=None, grid=None, curve_name='curve'):
        gt_curve = gt_curve[0].cpu().numpy()
        pred_curve = pred_curve[0].cpu().numpy()
        if base_curve is not None:
            base_curve = base_curve[0].cpu().numpy()
        title_text = f"{self.valid_dataset.metadata['spk_names'][data_idx]} - {self.valid_dataset.metadata['names'][data_idx]}"
        self.logger.all_rank_experiment.add_figure(f'{curve_name}_{data_idx}', curve_to_figure(
            gt_curve, pred_curve, base_curve, grid=grid, title=title_text
        ), self.global_step)
