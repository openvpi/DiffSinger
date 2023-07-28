import matplotlib
import torch
import torch.distributions
import torch.optim
import torch.utils.data

import utils
import utils.infer_utils
from basics.base_dataset import BaseDataset
from basics.base_task import BaseTask
from basics.base_vocoder import BaseVocoder
from modules.losses.diff_loss import DiffusionNoiseLoss
from modules.toplevel import DiffSingerAcoustic
from modules.vocoders.registry import get_vocoder_cls
from utils.hparams import hparams
from utils.plot import spec_to_figure, curve_to_figure

matplotlib.use('Agg')


class AcousticDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_variances = {}  # key: variance name, value: padding value
        if hparams.get('use_energy_embed', False):
            self.required_variances['energy'] = 0.0
        if hparams.get('use_breathiness_embed', False):
            self.required_variances['breathiness'] = 0.0

        self.need_key_shift = hparams.get('use_key_shift_embed', False)
        self.need_speed = hparams.get('use_speed_embed', False)
        self.need_spk_id = hparams['use_spk_id']

    def collater(self, samples):
        batch = super().collater(samples)

        tokens = utils.collate_nd([s['tokens'] for s in samples], 0)
        f0 = utils.collate_nd([s['f0'] for s in samples], 0.0)
        mel2ph = utils.collate_nd([s['mel2ph'] for s in samples], 0)
        mel = utils.collate_nd([s['mel'] for s in samples], 0.0)
        batch.update({
            'tokens': tokens,
            'mel2ph': mel2ph,
            'mel': mel,
            'f0': f0,
        })
        for v_name, v_pad in self.required_variances.items():
            batch[v_name] = utils.collate_nd([s[v_name] for s in samples], v_pad)
        if self.need_key_shift:
            batch['key_shift'] = torch.FloatTensor([s['key_shift'] for s in samples])[:, None]
        if self.need_speed:
            batch['speed'] = torch.FloatTensor([s['speed'] for s in samples])[:, None]
        if self.need_spk_id:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        return batch


class AcousticTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = AcousticDataset
        self.use_vocoder = hparams['infer'] or hparams['val_with_vocoder']
        if self.use_vocoder:
            self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        self.logged_gt_wav = set()
        self.required_variances = []
        if hparams.get('use_energy_embed', False):
            self.required_variances.append('energy')
        if hparams.get('use_breathiness_embed', False):
            self.required_variances.append('breathiness')

    def build_model(self):
        return DiffSingerAcoustic(
            vocab_size=len(self.phone_encoder),
            out_dims=hparams['audio_num_mel_bins']
        )

    # noinspection PyAttributeOutsideInit
    def build_losses_and_metrics(self):
        self.mel_loss = DiffusionNoiseLoss(loss_type=hparams['diff_loss_type'])

    def run_model(self, sample, infer=False):
        txt_tokens = sample['tokens']  # [B, T_ph]
        target = sample['mel']  # [B, T_s, M]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        variances = {
            v_name: sample[v_name]
            for v_name in self.required_variances
        }
        key_shift = sample.get('key_shift')
        speed = sample.get('speed')

        if hparams['use_spk_id']:
            spk_embed_id = sample['spk_ids']
        else:
            spk_embed_id = None
        output = self.model(
            txt_tokens, mel2ph=mel2ph, f0=f0, **variances,
            key_shift=key_shift, speed=speed, spk_embed_id=spk_embed_id,
            gt_mel=target, infer=infer
        )

        if infer:
            return output  # mel_pred
        else:
            x_recon, x_noise = output
            mel_loss = self.mel_loss(x_recon, x_noise, nonpadding=(mel2ph > 0).unsqueeze(-1).float())
            losses = {
                'mel_loss': mel_loss
            }

            return losses

    def on_train_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)

    def _on_validation_start(self):
        if self.use_vocoder and self.vocoder.get_device() != self.device:
            self.vocoder.to_device(self.device)

    def _validation_step(self, sample, batch_idx):
        losses = self.run_model(sample, infer=False)

        if batch_idx < hparams['num_valid_plots'] \
                and (self.trainer.distributed_sampler_kwargs or {}).get('rank', 0) == 0:
            mel_pred = self.run_model(sample, infer=True)

            if self.use_vocoder:
                self.plot_wav(batch_idx, sample['mel'], mel_pred, f0=sample['f0'])
            self.plot_mel(batch_idx, sample['mel'], mel_pred, name=f'diffmel_{batch_idx}')

        return losses, sample['size']

    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_mel, pred_mel, f0=None):
        gt_mel = gt_mel[0].cpu().numpy()
        pred_mel = pred_mel[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if batch_idx not in self.logged_gt_wav:
            gt_wav = self.vocoder.spec2wav(gt_mel, f0=f0)
            self.logger.experiment.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'],
                                             global_step=self.global_step)
            self.logged_gt_wav.add(batch_idx)
        pred_wav = self.vocoder.spec2wav(pred_mel, f0=f0)
        self.logger.experiment.add_audio(f'pred_{batch_idx}', pred_wav, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)

    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        name = f'mel_{batch_idx}' if name is None else name
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        spec_cat = torch.cat([(spec_out - spec).abs() + vmin, spec, spec_out], -1)
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

    def plot_curve(self, batch_idx, gt_curve, pred_curve, curve_name='curve'):
        name = f'{curve_name}_{batch_idx}'
        gt_curve = gt_curve[0].cpu().numpy()
        pred_curve = pred_curve[0].cpu().numpy()
        self.logger.experiment.add_figure(name, curve_to_figure(gt_curve, pred_curve), self.global_step)
