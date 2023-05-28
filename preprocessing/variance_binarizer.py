import csv
import os
import pathlib

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate

from basics.base_binarizer import BaseBinarizer
from modules.fastspeech.tts_modules import LengthRegulator
from utils.binarizer_utils import (
    SinusoidalSmoothingConv1d,
    get_mel2ph_torch,
    get_pitch_parselmouth,
    get_energy_librosa,
    get_breathiness_pyworld
)
from utils.hparams import hparams

os.environ["OMP_NUM_THREADS"] = "1"
VARIANCE_ITEM_ATTRIBUTES = [
    'spk_id',  # index number of dataset/speaker, int64
    'tokens',  # index numbers of phonemes, int64[T_ph,]
    'ph_dur',  # durations of phonemes, in number of frames, int64[T_ph,]
    'midi',  # phoneme-level mean MIDI pitch, int64[T_ph,]
    'ph2word',  # similar to mel2ph format, representing number of phones within each note, int64[T_ph,]
    'mel2ph',  # mel2ph format representing number of frames within each phone, int64[T_s,]
    'base_pitch',  # interpolated and smoothed frame-level MIDI pitch, float32[T_s,]
    'pitch',  # actual pitch in semitones, float32[T_s,]
    'energy',  # frame-level RMS, float32[T_s,]
    'breathiness',  # frame-level RMS of aperiodic parts, float32[T_s,]
]

# These operators are used as global variable due to a PyTorch shared memory bug on Windows.
# See https://github.com/pytorch/pytorch/issues/100358
midi_smooth: SinusoidalSmoothingConv1d = None
energy_smooth: SinusoidalSmoothingConv1d = None
breathiness_smooth: SinusoidalSmoothingConv1d = None


class VarianceBinarizer(BaseBinarizer):
    def __init__(self):
        super().__init__(data_attrs=VARIANCE_ITEM_ATTRIBUTES)

        predict_energy = hparams['predict_energy']
        predict_breathiness = hparams['predict_breathiness']
        self.predict_variances = predict_energy or predict_breathiness
        self.lr = LengthRegulator().to(self.device)
        # self.smooth: nn.Conv1d = None

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id):
        meta_data_dict = {}
        for utterance_label in csv.DictReader(
                open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf8')
        ):
            item_name = utterance_label['name']
            temp_dict = {
                'spk_id': ds_id,
                'wav_fn': str(raw_data_dir / 'wavs' / f'{item_name}.wav'),
                'ph_seq': utterance_label['ph_seq'].split(),
                'ph_dur': [float(x) for x in utterance_label['ph_dur'].split()]
            }

            assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']), \
                f'Lengths of ph_seq and ph_dur mismatch in \'{item_name}\'.'

            if hparams['predict_dur']:
                temp_dict['ph_num'] = [int(x) for x in utterance_label['ph_num'].split()]
                assert len(temp_dict['ph_seq']) == sum(temp_dict['ph_num']), \
                    f'Sum of ph_num does not equal length of ph_seq in \'{item_name}\'.'

            if hparams['predict_pitch']:
                temp_dict['note_seq'] = utterance_label['note_seq'].split()
                temp_dict['note_dur'] = [float(x) for x in utterance_label['note_dur'].split()]
                assert len(temp_dict['note_seq']) == len(temp_dict['note_dur']), \
                    f'Lengths of note_seq and note_dur mismatch in \'{item_name}\'.'
                assert any([note != 'rest' for note in temp_dict['note_seq']]), \
                    f'All notes are rest in \'{item_name}\'.'

            meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict

        self.items.update(meta_data_dict)

    def check_coverage(self):
        print('Coverage checks are temporarily skipped.')
        pass

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        seconds = sum(meta_data['ph_dur'])
        length = round(seconds / self.timestep)
        T_ph = len(meta_data['ph_seq'])
        processed_input = {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
            'spk_id': meta_data['spk_id'],
            'seconds': seconds,
            'length': length,
            'tokens': np.array(self.phone_encoder.encode(meta_data['ph_seq']), dtype=np.int64)
        }

        ph_dur_sec = torch.FloatTensor(meta_data['ph_dur']).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur_sec, dim=0) / self.timestep + 0.5).long()
        ph_dur = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))
        processed_input['ph_dur'] = ph_dur.cpu().numpy()

        mel2ph = get_mel2ph_torch(
            self.lr, ph_dur_sec, length, self.timestep, device=self.device
        )

        if hparams['predict_dur'] and (hparams['predict_pitch'] or self.predict_variances):
            processed_input['mel2ph'] = mel2ph.cpu().numpy()

        # Below: extract actual f0, convert to pitch and calculate delta pitch
        waveform, _ = librosa.load(meta_data['wav_fn'], sr=hparams['audio_sample_rate'], mono=True)
        f0, uv = get_pitch_parselmouth(waveform, length, hparams, interp_uv=True)
        if uv.all():  # All unvoiced
            print(f'Skipped \'{item_name}\': empty gt f0')
            return None
        pitch = torch.from_numpy(librosa.hz_to_midi(f0.astype(np.float32))).to(self.device)

        if hparams['predict_dur']:
            ph_num = torch.LongTensor(meta_data['ph_num']).to(self.device)
            ph2word = self.lr(ph_num[None])[0]
            processed_input['ph2word'] = ph2word.cpu().numpy()
            mel2dur = torch.gather(F.pad(ph_dur, [1, 0], value=1), 0, mel2ph)  # frame-level phone duration
            ph_midi = pitch.new_zeros(T_ph + 1).scatter_add(
                0, mel2ph, pitch / mel2dur
            )[1:]
            processed_input['midi'] = ph_midi.round().long().clamp(min=0, max=127).cpu().numpy()

        if hparams['predict_pitch']:
            # Below: calculate and interpolate frame-level MIDI pitch, which is a step function curve
            note_dur = torch.FloatTensor(meta_data['note_dur']).to(self.device)
            mel2note = get_mel2ph_torch(
                self.lr, note_dur, mel2ph.shape[0], self.timestep, device=self.device
            )
            note_pitch = torch.FloatTensor(
                [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in meta_data['note_seq']]
            ).to(self.device)
            frame_midi_pitch = torch.gather(F.pad(note_pitch, [1, 0], value=0), 0, mel2note)
            rest = (frame_midi_pitch < 0).cpu().numpy()
            frame_midi_pitch = frame_midi_pitch.cpu().numpy()
            interp_func = interpolate.interp1d(
                np.where(~rest)[0], frame_midi_pitch[~rest],
                kind='nearest', fill_value='extrapolate'
            )
            frame_midi_pitch[rest] = interp_func(np.where(rest)[0])
            frame_midi_pitch = torch.from_numpy(frame_midi_pitch).to(self.device)

            # Below: smoothen the pitch step curve as the base pitch curve
            global midi_smooth
            if midi_smooth is None:
                midi_smooth = SinusoidalSmoothingConv1d(
                    round(hparams['midi_smooth_width'] / self.timestep)
                ).eval().to(self.device)
            smoothed_midi_pitch = midi_smooth(frame_midi_pitch[None])[0]
            processed_input['base_pitch'] = smoothed_midi_pitch.cpu().numpy()

        if hparams['predict_pitch'] or self.predict_variances:
            processed_input['pitch'] = pitch.cpu().numpy()

        # Below: extract energy
        if hparams['predict_energy']:
            energy = get_energy_librosa(waveform, length, hparams).astype(np.float32)

            global energy_smooth
            if energy_smooth is None:
                energy_smooth = SinusoidalSmoothingConv1d(
                    round(hparams['energy_smooth_width'] / self.timestep)
                ).eval().to(self.device)
            energy = energy_smooth(torch.from_numpy(energy).to(self.device)[None])[0]

            processed_input['energy'] = energy.cpu().numpy()

        # Below: extract breathiness
        if hparams['predict_breathiness']:
            breathiness = get_breathiness_pyworld(waveform, f0 * ~uv, length, hparams).astype(np.float32)

            global breathiness_smooth
            if breathiness_smooth is None:
                breathiness_smooth = SinusoidalSmoothingConv1d(
                    round(hparams['breathiness_smooth_width'] / self.timestep)
                ).eval().to(self.device)
            breathiness = breathiness_smooth(torch.from_numpy(breathiness).to(self.device)[None])[0]

            processed_input['breathiness'] = breathiness.cpu().numpy()

        return processed_input

    def arrange_data_augmentation(self, data_iterator):
        return {}
