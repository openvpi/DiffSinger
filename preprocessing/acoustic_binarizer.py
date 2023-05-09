"""
    item: one piece of data
    item_name: data id
    wav_fn: wave file path
    spk: dataset name
    ph_seq: phoneme sequence
    ph_dur: phoneme durations
"""
import csv
import json
import os
import pathlib
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from basics.base_binarizer import BaseBinarizer, BinarizationError
from modules.fastspeech.tts_modules import LengthRegulator
from modules.vocoders.registry import VOCODERS
from utils.binarizer_utils import get_pitch_parselmouth, get_mel2ph_torch
from utils.hparams import hparams
from utils.phoneme_utils import build_phoneme_list

os.environ["OMP_NUM_THREADS"] = "1"
ACOUSTIC_ITEM_ATTRIBUTES = ['spk_id', 'mel', 'tokens', 'mel2ph', 'f0', 'key_shift', 'speed']


class AcousticBinarizer(BaseBinarizer):
    def __init__(self):
        super().__init__(data_attrs=ACOUSTIC_ITEM_ATTRIBUTES)
        self.lr = LengthRegulator()

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id):
        meta_info = {
            'category': 'acoustic',
            'format': 'grid'
        }
        meta_file = raw_data_dir / 'meta.json'
        if meta_file.exists():
            meta_info.update(json.load(open(meta_file, 'r', encoding='utf8')))
        category = meta_info['category']
        assert category == 'acoustic', \
            f'Dataset in \'{raw_data_dir}\' is of category \'{category}\', ' \
            f'but a dataset of category \'acoustic\' is required.'

        meta_data_dict = {}
        if meta_info['format'] == 'csv' or os.path.exists(raw_data_dir / 'transcriptions.csv'):
            for utterance_label in csv.DictReader(
                    open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf-8')
            ):
                item_name = utterance_label['name']
                temp_dict = {
                    'wav_fn': str(raw_data_dir / 'wavs' / f'{item_name}.wav'),
                    'ph_seq': utterance_label['ph_seq'].split(),
                    'ph_dur': [float(x) for x in utterance_label['ph_dur'].split()],
                    'spk_id': ds_id
                }
                assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']), \
                    f'Lengths of ph_seq and ph_dur mismatch in \'{item_name}\'.'
                meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict
        else:
            utterance_labels = open(raw_data_dir / 'transcriptions.txt', 'r', encoding='utf-8').readlines()
            for utterance_label in utterance_labels:
                song_info = utterance_label.split('|')
                item_name = song_info[0]
                temp_dict = {
                    'wav_fn': f'{raw_data_dir}/wavs/{item_name}.wav',
                    'ph_seq': song_info[2].split(),
                    'ph_dur': [float(x) for x in song_info[5].split()],
                    'spk_id': ds_id
                }
                assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']), \
                    f'Lengths of ph_seq and ph_dur mismatch in \'{item_name}\'.'
                meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict
        self.items.update(meta_data_dict)

    def check_coverage(self):
        # Group by phonemes in the dictionary.
        ph_required = set(build_phoneme_list())
        phoneme_map = {}
        for ph in ph_required:
            phoneme_map[ph] = 0
        ph_occurred = []
        # Load and count those phones that appear in the actual data
        for item_name in self.items:
            ph_occurred += self.items[item_name]['ph_seq']
            if len(ph_occurred) == 0:
                raise BinarizationError(f'Empty tokens in {item_name}.')
        for ph in ph_occurred:
            if ph not in ph_required:
                continue
            phoneme_map[ph] += 1
        ph_occurred = set(ph_occurred)

        print('===== Phoneme Distribution Summary =====')
        for i, key in enumerate(sorted(phoneme_map.keys())):
            if i == len(ph_required) - 1:
                end = '\n'
            elif i % 10 == 9:
                end = ',\n'
            else:
                end = ', '
            print(f'\'{key}\': {phoneme_map[key]}', end=end)

        # Draw graph.
        plt.figure(figsize=(int(len(ph_required) * 0.8), 10))
        x = sorted(phoneme_map.keys())
        values = [phoneme_map[k] for k in x]
        plt.bar(x=x, height=values)
        plt.tick_params(labelsize=15)
        plt.xlim(-1, len(ph_required))
        for a, b in zip(x, values):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
        plt.grid()
        plt.title('Phoneme Distribution Summary', fontsize=30)
        plt.xlabel('Phoneme', fontsize=20)
        plt.ylabel('Number of occurrences', fontsize=20)
        filename = self.binary_data_dir / 'phoneme_distribution.jpg'
        plt.savefig(fname=filename,
                    bbox_inches='tight',
                    pad_inches=0.25)
        print(f'| save summary to \'{filename}\'')
        # Check unrecognizable or missing phonemes
        if ph_occurred != ph_required:
            unrecognizable_phones = ph_occurred.difference(ph_required)
            missing_phones = ph_required.difference(ph_occurred)
            raise BinarizationError('transcriptions and dictionary mismatch.\n'
                                    f' (+) {sorted(unrecognizable_phones)}\n'
                                    f' (-) {sorted(missing_phones)}')

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
            'seconds': seconds,
            'length': length,
            'mel': mel,
            'tokens': np.array(self.phone_encoder.encode(meta_data['ph_seq']), dtype=np.int64),
            'ph_dur': np.array(meta_data['ph_dur']).astype(np.float32),
        }

        # get ground truth f0
        gt_f0, uv = get_pitch_parselmouth(
            wav, length, hparams, interp_uv=hparams['interp_uv']
        )
        if uv.all():  # All unvoiced
            print(f'Skipped \'{item_name}\': empty gt f0')
            return None
        processed_input['f0'] = gt_f0.astype(np.float32)

        # get ground truth dur
        processed_input['mel2ph'] = get_mel2ph_torch(
            self.lr, torch.from_numpy(processed_input['ph_dur']), length, self.timestep, device=self.device
        ).cpu().numpy()

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
        if self.augmentation_args.get('random_pitch_shifting') is not None:
            from augmentation.spec_stretch import SpectrogramStretchAugmentation
            aug_args = self.augmentation_args['random_pitch_shifting']
            key_shift_min, key_shift_max = aug_args['range']
            assert hparams.get('use_key_shift_embed', False), \
                'Random pitch shifting augmentation requires use_key_shift_embed == True.'
            assert key_shift_min < 0 < key_shift_max, \
                'Random pitch shifting augmentation must have a range where min < 0 < max.'

            aug_ins = SpectrogramStretchAugmentation(self.raw_data_dirs, aug_args)
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

        if self.augmentation_args.get('fixed_pitch_shifting') is not None:
            from augmentation.spec_stretch import SpectrogramStretchAugmentation
            aug_args = self.augmentation_args['fixed_pitch_shifting']
            targets = aug_args['targets']
            scale = aug_args['scale']
            assert self.augmentation_args.get('random_pitch_shifting') is None, \
                'Fixed pitch shifting augmentation is not compatible with random pitch shifting.'
            assert len(targets) == len(set(targets)), \
                'Fixed pitch shifting augmentation requires having no duplicate targets.'
            assert hparams['use_spk_id'], 'Fixed pitch shifting augmentation requires use_spk_id == True.'
            assert hparams['num_spk'] >= (1 + len(targets)) * len(self.spk_map), \
                'Fixed pitch shifting augmentation requires num_spk >= (1 + len(targets)) * len(speakers).'
            assert scale < 1, 'Fixed pitch shifting augmentation requires scale < 1.'

            aug_ins = SpectrogramStretchAugmentation(self.raw_data_dirs, aug_args)
            for i, target in enumerate(targets):
                aug_item_names = random.choices(all_item_names, k=int(scale * len(all_item_names)))
                for aug_item_name in aug_item_names:
                    replace_spk_id = int(aug_item_name.split(':', maxsplit=1)[0]) + (i + 1) * len(self.spk_map)
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

        if self.augmentation_args.get('random_time_stretching') is not None:
            from augmentation.spec_stretch import SpectrogramStretchAugmentation
            aug_args = self.augmentation_args['random_time_stretching']
            speed_min, speed_max = aug_args['range']
            domain = aug_args['domain']
            assert hparams.get('use_speed_embed', False), \
                'Random time stretching augmentation requires use_speed_embed == True.'
            assert 0 < speed_min < 1 < speed_max, \
                'Random time stretching augmentation must have a range where 0 < min < 1 < max.'
            assert domain in ['log', 'linear'], 'domain must be \'log\' or \'linear\'.'

            aug_ins = SpectrogramStretchAugmentation(self.raw_data_dirs, aug_args)
            scale = aug_args['scale']
            k_from_raw = int(scale / (1 + total_scale) * len(all_item_names))
            k_from_aug = int(total_scale * scale / (1 + total_scale) * len(all_item_names))
            k_mutate = int(total_scale * scale / (1 + scale) * len(all_item_names))
            aug_types = [0] * k_from_raw + [1] * k_from_aug + [2] * k_mutate
            aug_items = random.choices(all_item_names, k=k_from_raw) + random.choices(aug_list, k=k_from_aug + k_mutate)

            for aug_type, aug_item in zip(aug_types, aug_items):
                if domain == 'log':
                    # Uniform distribution in log domain
                    speed = speed_min * (speed_max / speed_min) ** random.random()
                else:
                    # Uniform distribution in linear domain
                    rand = random.uniform(-1, 1)
                    speed = 1 + (speed_max - 1) * rand if rand >= 0 else 1 + (1 - speed_min) * rand
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
                    aug_task = deepcopy(aug_item)
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
