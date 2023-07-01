from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from basics.base_module import CategorizedModule
from modules.commons.common_layers import (
    XavierUniformInitLinear as Linear,
    NormalInitEmbedding as Embedding
)
from modules.diffusion.ddpm import (
    GaussianDiffusion, PitchDiffusion
)
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from modules.fastspeech.param_adaptor import ParameterAdaptorModule
from modules.fastspeech.tts_modules import RhythmRegulator, LengthRegulator
from modules.fastspeech.variance_encoder import FastSpeech2Variance
from utils.hparams import hparams


class DiffSingerAcoustic(ParameterAdaptorModule, CategorizedModule):
    @property
    def category(self):
        return 'acoustic'

    def __init__(self, vocab_size, out_dims):
        super().__init__()
        self.fs2 = FastSpeech2Acoustic(
            vocab_size=vocab_size
        )

        self.diffusion = GaussianDiffusion(
            out_dims=out_dims,
            num_feats=1,
            timesteps=hparams['timesteps'],
            k_step=hparams['K_step'],
            denoiser_type=hparams['diff_decoder_type'],
            denoiser_args={
                'n_layers': hparams['residual_layers'],
                'n_chans': hparams['residual_channels'],
                'n_dilates': hparams['dilation_cycle_length'],
            },
            spec_min=hparams['spec_min'],
            spec_max=hparams['spec_max']
        )

    def forward(
            self, txt_tokens, mel2ph, f0, key_shift=None, speed=None,
            spk_embed_id=None, gt_mel=None, infer=True, **kwargs
    ):
        condition = self.fs2(
            txt_tokens, mel2ph, f0, key_shift=key_shift, speed=speed,
            spk_embed_id=spk_embed_id, **kwargs
        )

        if infer:
            mel_pred = self.diffusion(condition, infer=True)
            mel_pred *= ((mel2ph > 0).float()[:, :, None])
            return mel_pred
        else:
            x_recon, noise = self.diffusion(condition, gt_spec=gt_mel, infer=False)
            return x_recon, noise


class DiffSingerVariance(ParameterAdaptorModule, CategorizedModule):
    @property
    def category(self):
        return 'variance'

    def __init__(self, vocab_size):
        super().__init__()
        self.predict_dur = hparams['predict_dur']
        self.predict_pitch = hparams['predict_pitch']

        self.use_spk_id = hparams['use_spk_id']
        if self.use_spk_id:
            self.spk_embed = Embedding(hparams['num_spk'], hparams['hidden_size'])

        self.fs2 = FastSpeech2Variance(
            vocab_size=vocab_size
        )
        self.rr = RhythmRegulator()
        self.lr = LengthRegulator()

        if self.predict_pitch:
            self.pitch_retake_embed = Embedding(2, hparams['hidden_size'])
            pitch_hparams = hparams['pitch_prediction_args']
            self.base_pitch_embed = Linear(1, hparams['hidden_size'])
            self.pitch_predictor = PitchDiffusion(
                vmin=pitch_hparams['pitd_norm_min'],
                vmax=pitch_hparams['pitd_norm_max'],
                cmin=pitch_hparams['pitd_clip_min'],
                cmax=pitch_hparams['pitd_clip_max'],
                repeat_bins=pitch_hparams['repeat_bins'],
                timesteps=hparams['timesteps'],
                k_step=hparams['K_step'],
                denoiser_type=hparams['diff_decoder_type'],
                denoiser_args={
                    'n_layers': pitch_hparams['residual_layers'],
                    'n_chans': pitch_hparams['residual_channels'],
                    'n_dilates': pitch_hparams['dilation_cycle_length'],
                }
            )

        if self.predict_variances:
            self.pitch_embed = Linear(1, hparams['hidden_size'])
            self.variance_embeds = nn.ModuleDict({
                v_name: Linear(1, hparams['hidden_size'])
                for v_name in self.variance_prediction_list
            })
            self.variance_predictor = self.build_adaptor()

    def forward(
            self, txt_tokens, midi, ph2word, ph_dur=None, word_dur=None, mel2ph=None,
            base_pitch=None, pitch=None, pitch_retake=None, variance_retake: Dict[str, Tensor] = None,
            spk_id=None, infer=True, **kwargs
    ):
        if self.use_spk_id:
            ph_spk_mix_embed = kwargs.get('ph_spk_mix_embed')
            spk_mix_embed = kwargs.get('spk_mix_embed')
            if ph_spk_mix_embed is not None and spk_mix_embed is not None:
                ph_spk_embed = ph_spk_mix_embed
                spk_embed = spk_mix_embed
            else:
                ph_spk_embed = spk_embed = self.spk_embed(spk_id)[:, None, :]  # [B,] => [B, T=1, H]
        else:
            ph_spk_embed = spk_embed = None

        encoder_out, dur_pred_out = self.fs2(
            txt_tokens, midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, word_dur=word_dur,
            spk_embed=ph_spk_embed, infer=infer
        )

        if not self.predict_pitch and not self.predict_variances:
            return dur_pred_out, None, ({} if infer else None)

        if mel2ph is None and word_dur is not None:  # inference from file
            dur_pred_align = self.rr(dur_pred_out, ph2word, word_dur)
            mel2ph = self.lr(dur_pred_align)
            mel2ph = F.pad(mel2ph, [0, base_pitch.shape[1] - mel2ph.shape[1]])

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, hparams['hidden_size']])
        condition = torch.gather(encoder_out, 1, mel2ph_)

        if self.use_spk_id:
            condition += spk_embed

        if self.predict_pitch:
            if pitch_retake is None:
                pitch_retake = torch.ones_like(mel2ph, dtype=torch.bool)
            pitch_cond = condition + self.pitch_retake_embed(pitch_retake.long())
            base_pitch = base_pitch * pitch_retake + pitch * ~pitch_retake
            pitch_cond += self.base_pitch_embed(base_pitch[:, :, None])
            if infer:
                pitch_pred_out = self.pitch_predictor(pitch_cond, infer=True)
            else:
                pitch_pred_out = self.pitch_predictor(pitch_cond, pitch - base_pitch, infer=False)
        else:
            pitch_pred_out = None

        if not self.predict_variances:
            return dur_pred_out, pitch_pred_out, ({} if infer else None)

        if pitch is None:
            pitch = base_pitch + pitch_pred_out
        condition += self.pitch_embed(pitch[:, :, None])

        variance_inputs = self.collect_variance_inputs(**kwargs)
        if variance_retake is not None:
            variance_embeds = [
                self.variance_embeds[v_name](v_input[:, :, None]) * ~variance_retake[v_name][:, :, None]
                for v_name, v_input in zip(self.variance_prediction_list, variance_inputs)
            ]
            condition += torch.stack(variance_embeds, dim=-1).sum(-1)

        variance_outputs = self.variance_predictor(condition, variance_inputs, infer)

        if infer:
            variances_pred_out = self.collect_variance_outputs(variance_outputs)
        else:
            variances_pred_out = variance_outputs

        return dur_pred_out, pitch_pred_out, variances_pred_out
