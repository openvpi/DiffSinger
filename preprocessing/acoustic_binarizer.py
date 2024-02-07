"""
    item: one piece of data
    item_name: data id
    wav_fn: wave file path
    spk: dataset name
    ph_seq: phoneme sequence
    ph_dur: phoneme durations
"""
import csv
import os
import pathlib
import random
from copy import deepcopy

import numpy as np
import torch

from basics.base_binarizer import BaseBinarizer
from basics.base_pe import BasePE
from modules.fastspeech.tts_modules import LengthRegulator
from modules.pe import initialize_pe
from modules.vocoders.registry import VOCODERS
from utils.binarizer_utils import (
    SinusoidalSmoothingConv1d,
    get_mel2ph_torch,
    get_energy_librosa,
    get_breathiness_pyworld
)
from utils.hparams import hparams

os.environ["OMP_NUM_THREADS"] = "1"
ACOUSTIC_ITEM_ATTRIBUTES = [
    'spk_id',
    'mel',
    'tokens',
    'mel2ph',
    'f0',
    'energy',
    'breathiness',
    'key_shift',
    'speed'
]

pitch_extractor: BasePE = None
energy_smooth: SinusoidalSmoothingConv1d = None
breathiness_smooth: SinusoidalSmoothingConv1d = None


class AcousticBinarizer(BaseBinarizer):
    def __init__(self):
        super().__init__(data_attrs=ACOUSTIC_ITEM_ATTRIBUTES)
        self.lr = LengthRegulator()
        self.need_energy = hparams.get('use_energy_embed', False)
        self.need_breathiness = hparams.get('use_breathiness_embed', False)

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id, spk_id):
        meta_data_dict = {}
        if (raw_data_dir / 'transcriptions.csv').exists():
            with open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf-8') as f:
                for utterance_label in csv.DictReader(f):
                    item_name = utterance_label['name']
                    temp_dict = {
                        'wav_fn': str(raw_data_dir / 'wavs' / f'{item_name}.wav'),
                        'ph_seq': utterance_label['ph_seq'].split(),
                        'ph_dur': [float(x) for x in utterance_label['ph_dur'].split()],
                        'spk_id': spk_id,
                        'spk_name': self.speakers[ds_id],
                    }
                    assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']), \
                        f'Lengths of ph_seq and ph_dur mismatch in \'{item_name}\'.'
                    meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict
        else:
            raise FileNotFoundError(
                f'transcriptions.csv not found in {raw_data_dir}. '
                'If this is a dataset with the old transcription format, please consider '
                'migrating it to the new format via the following command:\n'
                'python scripts/migrate.py txt <INPUT_TXT>'
            )
        self.items.update(meta_data_dict)

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        if hparams['vocoder'] in VOCODERS:
            wav, mel = VOCODERS[hparams['vocoder']].wav2spec(meta_data['wav_fn'])
        else:
            wav, mel = VOCODERS[hparams['vocoder'].split('.')[-1]].wav2spec(meta_data['wav_fn'])
        length = mel.shape[0]
        seconds = length * hparams['hop_size'] / hparams['audio_sample_rate']
        processed_input = {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
            'spk_id': meta_data['spk_id'],
            'spk_name': meta_data['spk_name'],
            'seconds': seconds,
            'length': length,
            'mel': mel,
            'tokens': np.array(self.phone_encoder.encode(meta_data['ph_seq']), dtype=np.int64),
            'ph_dur': np.array(meta_data['ph_dur']).astype(np.float32),
        }

        # get ground truth dur
        processed_input['mel2ph'] = get_mel2ph_torch(
            self.lr, torch.from_numpy(processed_input['ph_dur']), length, self.timestep, device=self.device
        ).cpu().numpy()

        # get ground truth f0
        global pitch_extractor
        if pitch_extractor is None:
            pitch_extractor = initialize_pe()
        gt_f0, uv = pitch_extractor.get_pitch(
            wav, length, hparams, interp_uv=True
        )
        if uv.all():  # All unvoiced
            print(f'Skipped \'{item_name}\': empty gt f0')
            return None
        processed_input['f0'] = gt_f0.astype(np.float32)

        if self.need_energy:
            # get ground truth energy
            energy = get_energy_librosa(wav, length, hparams).astype(np.float32)

            global energy_smooth
            if energy_smooth is None:
                energy_smooth = SinusoidalSmoothingConv1d(
                    round(hparams['energy_smooth_width'] / self.timestep)
                ).eval().to(self.device)
            energy = energy_smooth(torch.from_numpy(energy).to(self.device)[None])[0]

            processed_input['energy'] = energy.cpu().numpy()

        if self.need_breathiness:
            # get ground truth breathiness
            breathiness = get_breathiness_pyworld(wav, gt_f0 * ~uv, length, hparams).astype(np.float32)

            global breathiness_smooth
            if breathiness_smooth is None:
                breathiness_smooth = SinusoidalSmoothingConv1d(
                    round(hparams['breathiness_smooth_width'] / self.timestep)
                ).eval().to(self.device)
            breathiness = breathiness_smooth(torch.from_numpy(breathiness).to(self.device)[None])[0]

            processed_input['breathiness'] = breathiness.cpu().numpy()

        if hparams.get('use_key_shift_embed', False):
            processed_input['key_shift'] = 0.

        if hparams.get('use_speed_embed', False):
            processed_input['speed'] = 1.

        return processed_input

    def arrange_data_augmentation(self, data_iterator):
        aug_map = {}
        aug_list = []
        all_item_names = [item_name for item_name, _ in data_iterator]
        total_scale = 0
        aug_pe = initialize_pe()
        if self.augmentation_args['random_pitch_shifting']['enabled']:
            from augmentation.spec_stretch import SpectrogramStretchAugmentation
            aug_args = self.augmentation_args['random_pitch_shifting']
            key_shift_min, key_shift_max = aug_args['range']
            assert hparams.get('use_key_shift_embed', False), \
                'Random pitch shifting augmentation requires use_key_shift_embed == True.'
            assert key_shift_min < 0 < key_shift_max, \
                'Random pitch shifting augmentation must have a range where min < 0 < max.'

            aug_ins = SpectrogramStretchAugmentation(self.raw_data_dirs, aug_args, pe=aug_pe)
            scale = aug_args['scale']
            aug_item_names = random.choices(all_item_names, k=int(scale * len(all_item_names)))

            for aug_item_name in aug_item_names:
                rand = random.uniform(-1, 1)
                if rand < 0:
                    key_shift = key_shift_min * abs(rand)
                else:
                    key_shift = key_shift_max * rand
                aug_task = {
                    'name': aug_item_name,
                    'func': aug_ins.process_item,
                    'kwargs': {'key_shift': key_shift}
                }
                if aug_item_name in aug_map:
                    aug_map[aug_item_name].append(aug_task)
                else:
                    aug_map[aug_item_name] = [aug_task]
                aug_list.append(aug_task)

            total_scale += scale

        if self.augmentation_args['fixed_pitch_shifting']['enabled']:
            from augmentation.spec_stretch import SpectrogramStretchAugmentation
            aug_args = self.augmentation_args['fixed_pitch_shifting']
            targets = aug_args['targets']
            scale = aug_args['scale']
            spk_id_size = max(self.spk_ids) + 1
            min_num_spk = (1 + len(targets)) * spk_id_size
            assert not self.augmentation_args['random_pitch_shifting']['enabled'], \
                'Fixed pitch shifting augmentation is not compatible with random pitch shifting.'
            assert len(targets) == len(set(targets)), \
                'Fixed pitch shifting augmentation requires having no duplicate targets.'
            assert hparams['use_spk_id'], 'Fixed pitch shifting augmentation requires use_spk_id == True.'
            assert hparams['num_spk'] >= min_num_spk, \
                f'Fixed pitch shifting augmentation requires num_spk >= (1 + len(targets)) * (max(spk_ids) + 1).'
            assert scale < 1, 'Fixed pitch shifting augmentation requires scale < 1.'

            aug_ins = SpectrogramStretchAugmentation(self.raw_data_dirs, aug_args, pe=aug_pe)
            for i, target in enumerate(targets):
                aug_item_names = random.choices(all_item_names, k=int(scale * len(all_item_names)))
                for aug_item_name in aug_item_names:
                    replace_spk_id = self.spk_ids[int(aug_item_name.split(':', maxsplit=1)[0])] + (i + 1) * spk_id_size
                    aug_task = {
                        'name': aug_item_name,
                        'func': aug_ins.process_item,
                        'kwargs': {'key_shift': target, 'replace_spk_id': replace_spk_id}
                    }
                    if aug_item_name in aug_map:
                        aug_map[aug_item_name].append(aug_task)
                    else:
                        aug_map[aug_item_name] = [aug_task]
                    aug_list.append(aug_task)

            total_scale += scale * len(targets)

        if self.augmentation_args['random_time_stretching']['enabled']:
            from augmentation.spec_stretch import SpectrogramStretchAugmentation
            aug_args = self.augmentation_args['random_time_stretching']
            speed_min, speed_max = aug_args['range']
            assert hparams.get('use_speed_embed', False), \
                'Random time stretching augmentation requires use_speed_embed == True.'
            assert 0 < speed_min < 1 < speed_max, \
                'Random time stretching augmentation must have a range where 0 < min < 1 < max.'

            aug_ins = SpectrogramStretchAugmentation(self.raw_data_dirs, aug_args, pe=aug_pe)
            scale = aug_args['scale']
            k_from_raw = int(scale / (1 + total_scale) * len(all_item_names))
            k_from_aug = int(total_scale * scale / (1 + total_scale) * len(all_item_names))
            k_mutate = int(total_scale * scale / (1 + scale) * len(all_item_names))
            aug_types = [0] * k_from_raw + [1] * k_from_aug + [2] * k_mutate
            aug_items = random.choices(all_item_names, k=k_from_raw) + random.choices(aug_list, k=k_from_aug + k_mutate)

            for aug_type, aug_item in zip(aug_types, aug_items):
                # Uniform distribution in log domain
                speed = speed_min * (speed_max / speed_min) ** random.random()
                if aug_type == 0:
                    aug_task = {
                        'name': aug_item,
                        'func': aug_ins.process_item,
                        'kwargs': {'speed': speed}
                    }
                    if aug_item in aug_map:
                        aug_map[aug_item].append(aug_task)
                    else:
                        aug_map[aug_item] = [aug_task]
                    aug_list.append(aug_task)
                elif aug_type == 1:
                    aug_task = {
                        'name': aug_item,
                        'func': aug_item['func'],
                        'kwargs': deepcopy(aug_item['kwargs'])
                    }
                    aug_task['kwargs']['speed'] = speed
                    if aug_item['name'] in aug_map:
                        aug_map[aug_item['name']].append(aug_task)
                    else:
                        aug_map[aug_item['name']] = [aug_task]
                    aug_list.append(aug_task)
                elif aug_type == 2:
                    aug_item['kwargs']['speed'] = speed

            total_scale += scale

        return aug_map
