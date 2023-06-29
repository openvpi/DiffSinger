import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from deployment.modules.diffusion import (
    GaussianDiffusionONNX, PitchDiffusionONNX, MultiVarianceDiffusionONNX
)
from deployment.modules.fastspeech2 import FastSpeech2AcousticONNX, FastSpeech2VarianceONNX
from modules.toplevel import DiffSingerAcoustic, DiffSingerVariance
from utils.hparams import hparams


class DiffSingerAcousticONNX(DiffSingerAcoustic):
    def __init__(self, vocab_size, out_dims):
        super().__init__(vocab_size, out_dims)
        del self.fs2
        del self.diffusion
        self.fs2 = FastSpeech2AcousticONNX(
            vocab_size=vocab_size
        )
        self.diffusion = GaussianDiffusionONNX(
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

    def forward_fs2(
            self,
            tokens: Tensor,
            durations: Tensor,
            f0: Tensor,
            variances: dict,
            gender: Tensor = None,
            velocity: Tensor = None,
            spk_embed: Tensor = None
    ) -> Tensor:
        return self.fs2(
            tokens, durations, f0, variances=variances,
            gender=gender, velocity=velocity, spk_embed=spk_embed
        )

    def forward_diffusion(self, condition: Tensor, speedup: int) -> Tensor:
        return self.diffusion(condition, speedup)

    def view_as_fs2(self) -> nn.Module:
        model = copy.deepcopy(self)
        try:
            del model.variance_embeds
            del model.variance_adaptor
        except AttributeError:
            pass
        del model.diffusion
        model.forward = model.forward_fs2
        return model

    def view_as_adaptor(self) -> nn.Module:
        model = copy.deepcopy(self)
        del model.fs2
        del model.diffusion
        raise NotImplementedError()

    def view_as_diffusion(self) -> nn.Module:
        model = copy.deepcopy(self)
        del model.fs2
        try:
            del model.variance_embeds
            del model.variance_adaptor
        except AttributeError:
            pass
        model.forward = model.forward_diffusion
        return model


class DiffSingerVarianceONNX(DiffSingerVariance):
    def __init__(self, vocab_size):
        super().__init__(vocab_size=vocab_size)
        del self.fs2
        self.fs2 = FastSpeech2VarianceONNX(
            vocab_size=vocab_size
        )
        self.hidden_size = hparams['hidden_size']
        if self.predict_pitch:
            del self.pitch_predictor
            self.smooth: nn.Conv1d = None
            pitch_hparams = hparams['pitch_prediction_args']
            self.pitch_predictor = PitchDiffusionONNX(
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
                },
            )
        if self.predict_variances:
            del self.variance_predictor
            self.variance_predictor = self.build_adaptor(cls=MultiVarianceDiffusionONNX)

    def build_smooth_op(self, device):
        smooth_kernel_size = round(hparams['midi_smooth_width'] * hparams['audio_sample_rate'] / hparams['hop_size'])
        smooth = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=smooth_kernel_size,
            bias=False,
            padding='same',
            padding_mode='replicate'
        ).eval()
        smooth_kernel = torch.sin(torch.from_numpy(
            np.linspace(0, 1, smooth_kernel_size).astype(np.float32) * np.pi
        ))
        smooth_kernel /= smooth_kernel.sum()
        smooth.weight.data = smooth_kernel[None, None]
        self.smooth = smooth.to(device)

    def forward_linguistic_encoder_word(self, tokens, word_div, word_dur):
        return self.fs2.forward_encoder_word(tokens, word_div, word_dur)

    def forward_linguistic_encoder_phoneme(self, tokens, ph_dur):
        return self.fs2.forward_encoder_phoneme(tokens, ph_dur)

    def forward_dur_predictor(self, encoder_out, x_masks, ph_midi):
        return self.fs2.forward_dur_predictor(encoder_out, x_masks, ph_midi)

    def forward_mel2x_gather(self, x_src, x_dur, x_dim=None):
        mel2x = self.lr(x_dur)
        if x_dim is not None:
            x_src = F.pad(x_src, [0, 0, 1, 0])
            mel2x = mel2x[..., None].repeat([1, 1, x_dim])
        else:
            x_src = F.pad(x_src, [1, 0])
        x_cond = torch.gather(x_src, 1, mel2x)
        return x_cond

    def forward_pitch_preprocess(
            self, encoder_out, ph_dur, note_midi, note_dur,
            pitch=None, retake=None
    ):
        condition = self.forward_mel2x_gather(encoder_out, ph_dur, x_dim=self.hidden_size)
        condition += self.retake_embed(retake.long())
        frame_midi_pitch = self.forward_mel2x_gather(note_midi, note_dur, x_dim=None)
        base_pitch = self.smooth(frame_midi_pitch)
        base_pitch += (pitch - base_pitch) * ~retake
        pitch_cond = condition + self.base_pitch_embed(base_pitch[:, :, None])
        return pitch_cond, base_pitch

    def forward_pitch_diffusion(
            self, pitch_cond, speedup: int = 1
    ):
        x_pred = self.pitch_predictor(pitch_cond, speedup)
        return x_pred

    def forward_pitch_postprocess(self, x_pred, base_pitch):
        pitch_pred = self.pitch_predictor.clamp_spec(x_pred) + base_pitch
        return pitch_pred

    def forward_variance_preprocess(
            self, encoder_out, ph_dur, pitch, variances: dict = None, retake=None
    ):
        condition = self.forward_mel2x_gather(encoder_out, ph_dur, x_dim=self.hidden_size)
        condition += self.retake_embed(retake.long())
        variance_cond = condition + self.pitch_embed(pitch[:, :, None])
        non_retake_masks = (~retake)[:, :, None].float()
        variance_embeds = [
            self.variance_embeds[v_name](variances[v_name][:, :, None]) * non_retake_masks
            for v_name in self.variance_prediction_list
        ]
        variance_cond += torch.stack(variance_embeds, dim=-1).sum(-1)
        return variance_cond

    def forward_variance_diffusion(self, variance_cond, speedup: int = 1):
        xs_pred = self.variance_predictor(variance_cond, speedup)
        return xs_pred

    def forward_variance_postprocess(self, xs_pred):
        if self.variance_predictor.num_feats == 1:
            xs_pred = [xs_pred]
        else:
            xs_pred = xs_pred.unbind(dim=1)
        variance_pred = self.variance_predictor.clamp_spec(xs_pred)
        return tuple(variance_pred)

    def view_as_linguistic_encoder(self):
        model = copy.deepcopy(self)
        if self.predict_pitch:
            del model.pitch_predictor
        if self.predict_variances:
            del model.variance_predictor
        model.fs2 = model.fs2.view_as_encoder()
        if self.predict_dur:
            model.forward = model.forward_linguistic_encoder_word
        else:
            model.forward = model.forward_linguistic_encoder_phoneme
        return model

    def view_as_dur_predictor(self):
        model = copy.deepcopy(self)
        if self.predict_pitch:
            del model.pitch_predictor
        if self.predict_variances:
            del model.variance_predictor
        assert self.predict_dur
        model.fs2 = model.fs2.view_as_dur_predictor()
        model.forward = model.forward_dur_predictor
        return model

    def view_as_pitch_preprocess(self):
        model = copy.deepcopy(self)
        del model.fs2
        if self.predict_pitch:
            del model.pitch_predictor
        if self.predict_variances:
            del model.variance_predictor
        model.forward = model.forward_pitch_preprocess
        return model

    def view_as_pitch_diffusion(self):
        model = copy.deepcopy(self)
        del model.fs2
        del model.lr
        if self.predict_variances:
            del model.variance_predictor
        assert self.predict_pitch
        model.forward = model.forward_pitch_diffusion
        return model

    def view_as_pitch_postprocess(self):
        model = copy.deepcopy(self)
        del model.fs2
        if self.predict_variances:
            del model.variance_predictor
        model.forward = model.forward_pitch_postprocess
        return model

    def view_as_variance_preprocess(self):
        model = copy.deepcopy(self)
        del model.fs2
        if self.predict_pitch:
            del model.pitch_predictor
        if self.predict_variances:
            del model.variance_predictor
        model.forward = model.forward_variance_preprocess
        return model

    def view_as_variance_diffusion(self):
        model = copy.deepcopy(self)
        del model.fs2
        del model.lr
        if self.predict_pitch:
            del model.pitch_predictor
        assert self.predict_variances
        model.forward = model.forward_variance_diffusion
        return model

    def view_as_variance_postprocess(self):
        model = copy.deepcopy(self)
        del model.fs2
        if self.predict_pitch:
            del model.pitch_predictor
        model.forward = model.forward_variance_postprocess
        return model
