import copy
import json

import tqdm
import pathlib
from collections import OrderedDict

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate

from basics.base_svs_infer import BaseSVSInfer
from modules.fastspeech.tts_modules import (
    LengthRegulator, RhythmRegulator,
    mel2ph_to_dur
)
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from modules.toplevel import DiffSingerVariance
from utils import load_ckpt
from utils.hparams import hparams
from utils.infer_utils import resample_align_curve
from utils.phoneme_utils import build_phoneme_list
from utils.pitch_utils import interp_f0
from utils.text_encoder import TokenTextEncoder


class DiffSingerVarianceInfer(BaseSVSInfer):
    def __init__(
            self, device=None, ckpt_steps=None,
            predictions: set = None
    ):
        super().__init__(device=device)
        self.ph_encoder = TokenTextEncoder(vocab_list=build_phoneme_list())
        if hparams['use_spk_id']:
            with open(pathlib.Path(hparams['work_dir']) / 'spk_map.json', 'r', encoding='utf8') as f:
                self.spk_map = json.load(f)
            assert isinstance(self.spk_map, dict) and len(self.spk_map) > 0, 'Invalid or empty speaker map!'
            assert len(self.spk_map) == len(set(self.spk_map.values())), 'Duplicate speaker id in speaker map!'
        self.model: DiffSingerVariance = self.build_model(ckpt_steps=ckpt_steps)
        self.lr = LengthRegulator()
        self.rr = RhythmRegulator()
        smooth_kernel_size = round(hparams['midi_smooth_width'] / self.timestep)
        self.smooth = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=smooth_kernel_size,
            bias=False,
            padding='same',
            padding_mode='replicate'
        ).eval().to(self.device)
        smooth_kernel = torch.sin(torch.from_numpy(
            np.linspace(0, 1, smooth_kernel_size).astype(np.float32) * np.pi
        ).to(self.device))
        smooth_kernel /= smooth_kernel.sum()
        self.smooth.weight.data = smooth_kernel[None, None]

        self.auto_completion_mode = len(predictions) == 0
        self.global_predict_dur = 'dur' in predictions and hparams['predict_dur']
        self.global_predict_pitch = 'pitch' in predictions and hparams['predict_pitch']
        self.variance_prediction_set = predictions.intersection(VARIANCE_CHECKLIST)
        self.global_predict_variances = len(self.variance_prediction_set) > 0

    def build_model(self, ckpt_steps=None):
        model = DiffSingerVariance(
            vocab_size=len(self.ph_encoder)
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], ckpt_steps=ckpt_steps,
                  prefix_in_ckpt='model', strict=True, device=self.device)
        return model

    @torch.no_grad()
    def preprocess_input(
            self, param, idx=0,
            load_dur: bool = False,
            load_pitch: bool = False
    ):
        """
        :param param: one segment in the .ds file
        :param idx: index of the segment
        :param load_dur: whether ph_dur is loaded
        :param load_pitch: whether pitch is loaded
        :return: batch of the model inputs
        """
        batch = {}
        summary = OrderedDict()
        txt_tokens = torch.LongTensor([self.ph_encoder.encode(param['ph_seq'].split())]).to(self.device)  # [B=1, T_ph]
        T_ph = txt_tokens.shape[1]
        batch['tokens'] = txt_tokens
        ph_num = torch.from_numpy(np.array([param['ph_num'].split()], np.int64)).to(self.device)  # [B=1, T_w]
        ph2word = self.lr(ph_num)  # => [B=1, T_ph]
        T_w = int(ph2word.max())
        batch['ph2word'] = ph2word

        note_seq = torch.FloatTensor(
            [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in param['note_seq'].split()]
        ).to(self.device)[None]  # [B=1, T_n]
        note_dur_sec = torch.from_numpy(np.array([param['note_dur'].split()], np.float32)).to(self.device)  # [B=1, T_n]
        note_acc = torch.round(torch.cumsum(note_dur_sec, dim=1) / self.timestep + 0.5).long()
        note_dur = torch.diff(note_acc, dim=1, prepend=note_acc.new_zeros(1, 1))
        mel2note = self.lr(note_dur)  # [B=1, T_s]
        T_s = mel2note.shape[1]

        summary['words'] = T_w
        summary['notes'] = note_seq.shape[1]
        summary['tokens'] = T_ph
        summary['frames'] = T_s
        summary['seconds'] = '%.2f' % (T_s * self.timestep)

        if hparams['use_spk_id']:
            ph_spk_mix_id, ph_spk_mix_value = self.load_speaker_mix(
                param_src=param, summary_dst=summary, mix_mode='token', mix_length=T_ph
            )
            spk_mix_id, spk_mix_value = self.load_speaker_mix(
                param_src=param, summary_dst=summary, mix_mode='frame', mix_length=T_s
            )
            batch['ph_spk_mix_id'] = ph_spk_mix_id
            batch['ph_spk_mix_value'] = ph_spk_mix_value
            batch['spk_mix_id'] = spk_mix_id
            batch['spk_mix_value'] = spk_mix_value

        if load_dur:
            # Get mel2ph if ph_dur is needed
            ph_dur_sec = torch.from_numpy(
                np.array([param['ph_dur'].split()], np.float32)
            ).to(self.device)  # [B=1, T_ph]
            ph_acc = torch.round(torch.cumsum(ph_dur_sec, dim=1) / self.timestep + 0.5).long()
            ph_dur = torch.diff(ph_acc, dim=1, prepend=ph_acc.new_zeros(1, 1))
            mel2ph = self.lr(ph_dur, txt_tokens == 0)
            if mel2ph.shape[1] != T_s:  # Align phones with notes
                mel2ph = F.pad(mel2ph, [0, T_s - mel2ph.shape[1]], value=mel2ph[0, -1])
                ph_dur = mel2ph_to_dur(mel2ph, T_ph)
            # Get word_dur from ph_dur and ph_num
            word_dur = note_dur.new_zeros(1, T_w + 1).scatter_add(
                1, ph2word, ph_dur
            )[:, 1:]  # => [B=1, T_w]
        else:
            ph_dur = None
            mel2ph = None
            # Get word_dur from note_dur and note_slur
            is_slur = torch.BoolTensor([[int(s) for s in param['note_slur'].split()]]).to(self.device)  # [B=1, T_n]
            note2word = torch.cumsum(~is_slur, dim=1)  # [B=1, T_n]
            word_dur = note_dur.new_zeros(1, T_w + 1).scatter_add(
                1, note2word, note_dur
            )[:, 1:]  # => [B=1, T_w]

        batch['ph_dur'] = ph_dur
        batch['mel2ph'] = mel2ph

        mel2word = self.lr(word_dur)  # [B=1, T_s]
        if mel2word.shape[1] != T_s:  # Align words with notes
            mel2word = F.pad(mel2word, [0, T_s - mel2word.shape[1]], value=mel2word[0, -1])
            word_dur = mel2ph_to_dur(mel2word, T_w)
        batch['word_dur'] = word_dur

        # Calculate frame-level MIDI pitch, which is a step function curve
        frame_midi_pitch = torch.gather(
            F.pad(note_seq, [1, 0]), 1, mel2note
        )  # => frame-level MIDI pitch, [B=1, T_s]
        rest = (frame_midi_pitch < 0)[0].cpu().numpy()
        frame_midi_pitch = frame_midi_pitch[0].cpu().numpy()
        interp_func = interpolate.interp1d(
            np.where(~rest)[0], frame_midi_pitch[~rest],
            kind='nearest', fill_value='extrapolate'
        )
        frame_midi_pitch[rest] = interp_func(np.where(rest)[0])
        frame_midi_pitch = torch.from_numpy(frame_midi_pitch[None]).to(self.device)
        base_pitch = self.smooth(frame_midi_pitch)
        batch['base_pitch'] = base_pitch

        if ph_dur is not None:
            # Phone durations are available, calculate phoneme-level MIDI.
            mel2pdur = torch.gather(F.pad(ph_dur, [1, 0], value=1), 1, mel2ph)  # frame-level phone duration
            ph_midi = frame_midi_pitch.new_zeros(1, T_ph + 1).scatter_add(
                1, mel2ph, frame_midi_pitch / mel2pdur
            )[:, 1:]
        else:
            # Phone durations are not available, calculate word-level MIDI instead.
            mel2wdur = torch.gather(F.pad(word_dur, [1, 0], value=1), 1, mel2word)
            w_midi = frame_midi_pitch.new_zeros(1, T_w + 1).scatter_add(
                1, mel2word, frame_midi_pitch / mel2wdur
            )[:, 1:]
            # Convert word-level MIDI to phoneme-level MIDI
            ph_midi = torch.gather(F.pad(w_midi, [1, 0]), 1, ph2word)
        ph_midi = ph_midi.round().long()
        batch['midi'] = ph_midi

        if load_pitch:
            f0 = resample_align_curve(
                np.array(param['f0_seq'].split(), np.float32),
                original_timestep=float(param['f0_timestep']),
                target_timestep=self.timestep,
                align_length=T_s
            )
            batch['pitch'] = torch.from_numpy(
                librosa.hz_to_midi(interp_f0(f0)[0]).astype(np.float32)
            ).to(self.device)[None]

        if self.model.predict_dur:
            if load_dur:
                summary['ph_dur'] = 'manual'
            elif self.auto_completion_mode or self.global_predict_dur:
                summary['ph_dur'] = 'auto'
            else:
                summary['ph_dur'] = 'ignored'

        if self.model.predict_pitch:
            if load_pitch:
                summary['pitch'] = 'manual'
            elif self.auto_completion_mode or self.global_predict_pitch:
                summary['pitch'] = 'auto'

                # Load expressiveness
                expressiveness = param.get('expressiveness', 1.)
                if isinstance(expressiveness, (int, float, bool)):
                    summary['expressiveness'] = f'static({expressiveness:.3f})'
                    batch['expressiveness'] = torch.FloatTensor([expressiveness]).to(self.device)[:, None]  # [B=1, T=1]
                else:
                    summary['expressiveness'] = 'dynamic'
                    expressiveness = resample_align_curve(
                        np.array(expressiveness.split(), np.float32),
                        original_timestep=float(param['expressiveness_timestep']),
                        target_timestep=self.timestep,
                        align_length=T_s
                    )
                    batch['expressiveness'] = torch.from_numpy(expressiveness.astype(np.float32)).to(self.device)[None]

            else:
                summary['pitch'] = 'ignored'

        if self.model.predict_variances:
            for v_name in self.model.variance_prediction_list:
                if self.auto_completion_mode and param.get(v_name) is None or v_name in self.variance_prediction_set:
                    summary[v_name] = 'auto'
                else:
                    summary[v_name] = 'ignored'

        print(f'[{idx}]\t' + ', '.join(f'{k}: {v}' for k, v in summary.items()))

        return batch

    @torch.no_grad()
    def forward_model(self, sample):
        txt_tokens = sample['tokens']
        midi = sample['midi']
        ph2word = sample['ph2word']
        word_dur = sample['word_dur']
        ph_dur = sample['ph_dur']
        mel2ph = sample['mel2ph']
        base_pitch = sample['base_pitch']
        expressiveness = sample.get('expressiveness')
        pitch = sample.get('pitch')

        if hparams['use_spk_id']:
            ph_spk_mix_id = sample['ph_spk_mix_id']
            ph_spk_mix_value = sample['ph_spk_mix_value']
            spk_mix_id = sample['spk_mix_id']
            spk_mix_value = sample['spk_mix_value']
            ph_spk_mix_embed = torch.sum(
                self.model.spk_embed(ph_spk_mix_id) * ph_spk_mix_value.unsqueeze(3),  # => [B, T_ph, N, H]
                dim=2, keepdim=False
            )  # => [B, T_ph, H]
            spk_mix_embed = torch.sum(
                self.model.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(3),  # => [B, T_s, N, H]
                dim=2, keepdim=False
            )  # [B, T_s, H]
        else:
            ph_spk_mix_embed = spk_mix_embed = None

        dur_pred, pitch_pred, variance_pred = self.model(
            txt_tokens, midi=midi, ph2word=ph2word, word_dur=word_dur, ph_dur=ph_dur,
            mel2ph=mel2ph, base_pitch=base_pitch, pitch=pitch,
            ph_spk_mix_embed=ph_spk_mix_embed, spk_mix_embed=spk_mix_embed,
            retake=None, expressiveness=expressiveness, infer=True
        )
        if dur_pred is not None:
            dur_pred = self.rr(dur_pred, ph2word, word_dur)
        if pitch_pred is not None:
            pitch_pred = base_pitch + pitch_pred
        return dur_pred, pitch_pred, variance_pred

    def infer_once(self, param):
        batch = self.preprocess_input(param)
        dur_pred, pitch_pred, variance_pred = self.forward_model(batch)
        if dur_pred is not None:
            dur_pred = dur_pred[0].cpu().numpy()
        if pitch_pred is not None:
            pitch_pred = pitch_pred[0].cpu().numpy()
            f0_pred = librosa.midi_to_hz(pitch_pred)
        else:
            f0_pred = None
        variance_pred = {
            k: v[0].cpu().numpy()
            for k, v in variance_pred.items()
        }
        return dur_pred, f0_pred, variance_pred

    def run_inference(
            self, params,
            out_dir: pathlib.Path = None,
            title: str = None,
            num_runs: int = 1,
            seed: int = -1
    ):
        batches = []
        predictor_flags: list[tuple[bool, bool, bool]] = []

        for i, param in enumerate(params):
            param: dict
            if self.auto_completion_mode:
                flag = (
                    self.model.fs2.predict_dur and param.get('ph_dur') is None,
                    self.model.predict_pitch and param.get('f0_seq') is None,
                    self.model.predict_variances and any(
                        param.get(v_name) is None for v_name in self.model.variance_prediction_list
                    )
                )
            else:
                predict_variances = self.model.predict_variances and self.global_predict_variances
                predict_pitch = self.model.predict_pitch and (
                        self.global_predict_pitch or (param.get('f0_seq') is None and predict_variances)
                )
                predict_dur = self.model.predict_dur and (
                        self.global_predict_dur or (
                            param.get('ph_dur') is None and (predict_pitch or predict_variances))
                )
                flag = (predict_dur, predict_pitch, predict_variances)
            predictor_flags.append(flag)
            batches.append(self.preprocess_input(
                param, idx=i,
                load_dur=not flag[0] and (flag[1] or flag[2]),
                load_pitch=not flag[1] and flag[2]
            ))

        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_runs):
            results = []
            for param, flag, batch in tqdm.tqdm(
                    zip(params, predictor_flags, batches), desc='infer segments', total=len(params)
            ):
                if 'seed' in param:
                    torch.manual_seed(param["seed"] & 0xffff_ffff)
                    torch.cuda.manual_seed_all(param["seed"] & 0xffff_ffff)
                elif seed >= 0:
                    torch.manual_seed(seed & 0xffff_ffff)
                    torch.cuda.manual_seed_all(seed & 0xffff_ffff)
                param_copy = copy.deepcopy(param)

                flag_saved = (
                    self.model.fs2.predict_dur,
                    self.model.predict_pitch,
                    self.model.predict_variances
                )
                (
                    self.model.fs2.predict_dur,
                    self.model.predict_pitch,
                    self.model.predict_variances
                ) = flag
                dur_pred, pitch_pred, variance_pred = self.forward_model(batch)
                (
                    self.model.fs2.predict_dur,
                    self.model.predict_pitch,
                    self.model.predict_variances
                ) = flag_saved

                if dur_pred is not None and (self.auto_completion_mode or self.global_predict_dur):
                    dur_pred = dur_pred[0].cpu().numpy()
                    param_copy['ph_dur'] = ' '.join(str(round(dur, 6)) for dur in (dur_pred * self.timestep).tolist())
                if pitch_pred is not None and (self.auto_completion_mode or self.global_predict_pitch):
                    pitch_pred = pitch_pred[0].cpu().numpy()
                    f0_pred = librosa.midi_to_hz(pitch_pred)
                    param_copy['f0_seq'] = ' '.join([str(round(freq, 1)) for freq in f0_pred.tolist()])
                    param_copy['f0_timestep'] = str(self.timestep)
                variance_pred = {
                    k: v[0].cpu().numpy()
                    for k, v in variance_pred.items()
                    if (self.auto_completion_mode and param.get(k) is None) or k in self.variance_prediction_set
                }
                for v_name, v_pred in variance_pred.items():
                    param_copy[v_name] = ' '.join([str(round(v, 4)) for v in v_pred.tolist()])
                    param_copy[f'{v_name}_timestep'] = str(self.timestep)

                # Restore ph_spk_mix and spk_mix
                if 'ph_spk_mix' in param_copy and 'spk_mix' in param_copy:
                    if 'ph_spk_mix_backup' in param_copy:
                        if param_copy['ph_spk_mix_backup'] is None:
                            del param_copy['ph_spk_mix']
                        else:
                            param_copy['ph_spk_mix'] = param_copy['ph_spk_mix_backup']
                        del param['ph_spk_mix_backup']
                    if 'spk_mix_backup' in param_copy:
                        if param_copy['ph_spk_mix_backup'] is None:
                            del param_copy['spk_mix']
                        else:
                            param_copy['spk_mix'] = param_copy['spk_mix_backup']
                        del param['spk_mix_backup']

                results.append(param_copy)

            if num_runs > 1:
                filename = f'{title}-{str(i).zfill(3)}.ds'
            else:
                filename = f'{title}.ds'
            save_path = out_dir / filename
            with open(save_path, 'w', encoding='utf8') as f:
                print(f'| save params: {save_path}')
                json.dump(results, f, ensure_ascii=False, indent=2)
