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
from modules.toplevel import DiffSingerVariance
from utils import load_ckpt
from utils.hparams import hparams
from utils.infer_utils import resample_align_curve
from utils.phoneme_utils import build_phoneme_list
from utils.pitch_utils import interp_f0
from utils.text_encoder import TokenTextEncoder


class DiffSingerVarianceInfer(BaseSVSInfer):
    def __init__(self, device=None, ckpt_steps=None):
        super().__init__(device=device)
        self.ph_encoder = TokenTextEncoder(vocab_list=build_phoneme_list())
        self.model = self.build_model(ckpt_steps=ckpt_steps)
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

    def build_model(self, ckpt_steps=None):
        model = DiffSingerVariance(
            vocab_size=len(self.ph_encoder)
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], ckpt_steps=ckpt_steps, required_category='variance',
                  prefix_in_ckpt='model', strict=True, device=self.device)
        return model

    @torch.no_grad()
    def preprocess_input(self, param):
        """
        :param param: one segment in the .ds file
        :return: batch of the model inputs
        """
        batch = {}
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
        mel2note = self.lr(note_dur)  # [B=1, T_t]
        T_t = mel2note.shape[1]
        is_slur = torch.BoolTensor([[int(s) for s in param['note_slur'].split()]]).to(self.device)  # [B=1, T_n]
        note2word = torch.cumsum(~is_slur, dim=1)  # [B=1, T_n]
        word_dur = note_dur.new_zeros(1, T_w + 1).scatter_add(
            1, note2word, note_dur
        )[:, 1:]  # => [B=1, T_w]
        mel2word = self.lr(word_dur)  # [B=1, T_t]

        print(f'Length: {T_w} word(s), {note_seq.shape[1]} note(s), {T_ph} token(s), '
              f'{T_t} frame(s), {T_t * self.timestep:.2f} second(s)')

        if mel2word.shape[1] != T_t:  # Align words with notes
            mel2word = F.pad(mel2word, [0, T_t - mel2word.shape[1]], value=mel2word[0, -1])
            word_dur = mel2ph_to_dur(mel2word, T_w)
        batch['word_dur'] = word_dur

        if param.get('ph_dur'):  # Get mel2ph if ph_dur is given
            ph_dur_sec = torch.from_numpy(
                np.array([param['ph_dur'].split()], np.float32)
            ).to(self.device)  # [B=1, T_ph]
            ph_acc = torch.round(torch.cumsum(ph_dur_sec, dim=1) / self.timestep + 0.5).long()
            ph_dur = torch.diff(ph_acc, dim=1, prepend=ph_acc.new_zeros(1, 1))
            mel2ph = self.lr(ph_dur, txt_tokens == 0)
            if mel2ph.shape[1] != T_t:  # Align phones with notes
                mel2ph = F.pad(mel2ph, [0, T_t - mel2ph.shape[1]], value=mel2ph[0, -1])
                ph_dur = mel2ph_to_dur(mel2ph, T_ph)
        else:
            ph_dur = None
            mel2ph = None
        batch['ph_dur'] = ph_dur
        batch['mel2ph'] = mel2ph

        # Calculate frame-level MIDI pitch, which is a step function curve
        frame_midi_pitch = torch.gather(
            F.pad(note_seq, [1, 0]), 1, mel2note
        )  # => frame-level MIDI pitch, [B=1, T_t]
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

        if param.get('f0_seq'):
            f0 = resample_align_curve(
                np.array(param['f0_seq'].split(), np.float32),
                original_timestep=float(param['f0_timestep']),
                target_timestep=self.timestep,
                align_length=T_t
            )
            batch['delta_pitch'] = torch.from_numpy(
                librosa.hz_to_midi(interp_f0(f0)[0]).astype(np.float32)
            ).to(self.device)[None] - base_pitch

        return batch

    @torch.no_grad()
    def run_model(self, sample):
        txt_tokens = sample['tokens']
        midi = sample['midi']
        ph2word = sample['ph2word']
        word_dur = sample['word_dur']
        ph_dur = sample['ph_dur']
        mel2ph = sample['mel2ph']
        base_pitch = sample['base_pitch']
        delta_pitch = sample.get('delta_pitch')

        dur_pred, pitch_pred, variance_pred = self.model(
            txt_tokens, midi=midi, ph2word=ph2word, word_dur=word_dur, ph_dur=ph_dur,
            mel2ph=mel2ph, base_pitch=base_pitch, delta_pitch=delta_pitch,
            retake=None, infer=True
        )
        if dur_pred is not None:
            dur_pred = self.rr(dur_pred, ph2word, word_dur)
        if pitch_pred is not None:
            pitch_pred = base_pitch + pitch_pred
        return dur_pred, pitch_pred, variance_pred

    def infer_once(self, param):
        batch = self.preprocess_input(param)
        dur_pred, pitch_pred, variance_pred = self.run_model(batch)
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
