import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.base_module import CategorizedModule
from modules.commons.common_layers import (
    XavierUniformInitLinear as Linear,
    NormalInitEmbedding as Embedding
)
from modules.diffusion.ddpm import (
    GaussianDiffusion, PitchDiffusion
)
from modules.fastspeech.acoustic_encoder import FastSpeech2Acoustic
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST, ParameterAdaptorModule
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

        if self.predict_variances:
            self.variance_adaptor = self.build_adaptor()
        variances_to_embed = set()
        if hparams.get('use_energy_embed', False) and not self.predict_energy:
            # energy is embedded but not predicted
            variances_to_embed.add('energy')
        if hparams.get('use_breathiness_embed', False) and not self.predict_breathiness:
            # breathiness is embedded but not predicted
            variances_to_embed.add('breathiness')
        self.embed_variances = len(variances_to_embed) > 0
        self.variance_aware_list = [
            v_name for v_name in VARIANCE_CHECKLIST
            if v_name in self.variance_prediction_list or v_name in self.variances_to_embed
        ]
        if self.embed_variances or self.predict_variances:
            self.variance_embeds = nn.ModuleDict({
                v_name: Linear(1, hparams['hidden_size'])
                for v_name in self.variance_aware_list
            })

        self.diffusion = GaussianDiffusion(
            out_dims=out_dims,
            num_feats=1,
            timesteps=hparams['timesteps'],
            k_step=hparams['K_step'],
            denoiser_type=hparams['diff_decoder_type'],
            denoiser_args=(
                hparams['residual_layers'],
                hparams['residual_channels']
            ),
            spec_min=hparams['spec_min'],
            spec_max=hparams['spec_max']
        )

    def forward(
            self, txt_tokens, mel2ph, f0, key_shift=None, speed=None,
            spk_embed_id=None, gt_mel=None, infer=True, **kwargs
    ):
        adaptor_cond, mel_cond = self.fs2(
            txt_tokens, mel2ph, f0, key_shift=key_shift, speed=speed,
            spk_embed_id=spk_embed_id, infer=infer, **kwargs
        )

        variance_embed_inputs = {
            v_name: kwargs.get(v_name) for v_name in self.variance_aware_list
        }  # all possible variance inputs

        if infer:
            if self.predict_variances:
                # get variance predictor inputs
                variance_preset_inputs = self.collect_variance_inputs(**variance_embed_inputs)
                if not all([v is not None for v in variance_preset_inputs]):  # need to predict some variances
                    variance_pred_outputs = self.collect_variance_outputs(
                        self.variance_adaptor(adaptor_cond, infer=True)
                    )  # dict of predictions
                    variance_embed_inputs = {
                        v_name: (
                            variance_embed_inputs[v_name] if variance_embed_inputs[v_name] is not None
                            else variance_pred_outputs[v_name]
                        )
                        for v_name in self.variance_aware_list
                    }  # merge presets and predictions, should contain no NoneType
                variance_pred_out = self.collect_variance_outputs(variance_embed_inputs)  # collect from embed inputs
            else:
                variance_pred_out = {}
        else:
            if self.predict_variances:
                # use gt variances to train the predictor
                variance_inputs = self.collect_variance_inputs(**variance_embed_inputs)
                variance_pred_out = self.variance_adaptor(adaptor_cond, variance_inputs, infer=False)
            else:
                variance_pred_out = None

        if self.predict_variances or self.embed_variances:
            # embed variances into mel condition
            variance_embeds = torch.stack([
                self.variance_embeds[v_name](variance_embed_inputs[v_name][:, :, None])  # [B, T] => [B, T, H]
                for v_name in self.variance_aware_list
            ], dim=-1).sum(-1)
            mel_cond += variance_embeds

        if infer:
            mel_pred_out = self.diffusion(mel_cond, infer=True)
            mel_pred_out *= ((mel2ph > 0).float()[:, :, None])
        else:
            mel_pred_out = self.diffusion(mel_cond, gt_spec=gt_mel, infer=False)

        return mel_pred_out, variance_pred_out


class DiffSingerVariance(ParameterAdaptorModule, CategorizedModule):
    @property
    def category(self):
        return 'variance'

    def __init__(self, vocab_size):
        super().__init__()
        self.predict_dur = hparams['predict_dur']
        self.fs2 = FastSpeech2Variance(
            vocab_size=vocab_size
        )
        self.rr = RhythmRegulator()
        self.lr = LengthRegulator()

        self.predict_pitch = hparams['predict_pitch']

        predict_energy = hparams.get('predict_energy', False)
        predict_breathiness = hparams.get('predict_breathiness', False)
        self.variance_prediction_list = []
        if predict_energy:
            self.variance_prediction_list.append('energy')
        if predict_breathiness:
            self.variance_prediction_list.append('breathiness')
        self.predict_variances = len(self.variance_prediction_list) > 0

        if self.predict_pitch or self.predict_variances:
            self.retake_embed = Embedding(2, hparams['hidden_size'])

        if self.predict_pitch:
            pitch_hparams = hparams['pitch_prediction_args']
            self.base_pitch_embed = Linear(1, hparams['hidden_size'])
            self.pitch_predictor = PitchDiffusion(
                vmin=pitch_hparams['pitch_delta_vmin'],
                vmax=pitch_hparams['pitch_delta_vmax'],
                repeat_bins=pitch_hparams['num_pitch_bins'],
                timesteps=hparams['timesteps'],
                k_step=hparams['K_step'],
                denoiser_type=hparams['diff_decoder_type'],
                denoiser_args=(
                    pitch_hparams['residual_layers'],
                    pitch_hparams['residual_channels']
                )
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
            base_pitch=None, delta_pitch=None, retake=None, infer=True, **kwargs
    ):
        encoder_out, dur_pred_out = self.fs2(
            txt_tokens, midi=midi, ph2word=ph2word,
            ph_dur=ph_dur, word_dur=word_dur, infer=infer
        )

        if not self.predict_pitch and not self.predict_variances:
            return dur_pred_out, None, None, ({} if infer else None)

        if mel2ph is None and word_dur is not None:  # inference from file
            dur_pred_align = self.rr(dur_pred_out, ph2word, word_dur)
            mel2ph = self.lr(dur_pred_align)
            mel2ph = F.pad(mel2ph, [0, base_pitch.shape[1] - mel2ph.shape[1]])

        encoder_out = F.pad(encoder_out, [0, 0, 1, 0])
        mel2ph_ = mel2ph[..., None].repeat([1, 1, hparams['hidden_size']])
        condition = torch.gather(encoder_out, 1, mel2ph_)

        if self.predict_pitch or self.predict_variances:
            if retake is None:
                retake_embed = self.retake_embed(torch.ones_like(mel2ph))
            else:
                retake_embed = self.retake_embed(retake.long())
            condition += retake_embed

        if self.predict_pitch:
            if retake is not None:
                base_pitch = base_pitch + delta_pitch * ~retake
                delta_pitch = delta_pitch * retake
            pitch_cond = condition + self.base_pitch_embed(base_pitch[:, :, None])
            pitch_pred_out = self.pitch_predictor(pitch_cond, delta_pitch, infer)
        else:
            pitch_pred_out = None

        if not self.predict_variances:
            return dur_pred_out, pitch_pred_out, ({} if infer else None)

        if delta_pitch is None:
            pitch = base_pitch + pitch_pred_out
        else:
            pitch = base_pitch + delta_pitch
        condition += self.pitch_embed(pitch[:, :, None])

        variance_inputs = self.collect_variance_inputs(**kwargs)

        if retake is None:
            variance_embeds = [
                self.variance_embeds[v_name](torch.zeros_like(pitch))
                for v_name in self.variance_prediction_list
            ]
        else:
            variance_embeds = [
                self.variance_embeds[v_name]((v_input * retake)[:, :, None])
                for v_name, v_input in zip(self.variance_prediction_list, variance_inputs)
            ]
        condition += torch.stack(variance_embeds, dim=-1).sum(-1)

        variance_outputs = self.variance_predictor(condition, variance_inputs, infer)

        if infer:
            variances_pred_out = self.collect_variance_outputs(variance_outputs)
        else:
            variances_pred_out = variance_outputs

        return dur_pred_out, pitch_pred_out, variances_pred_out
