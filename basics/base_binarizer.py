import json
import pathlib
import random
import shutil
import warnings
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from utils.hparams import hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.multiprocess_utils import chunked_multiprocess_run
from utils.phoneme_utils import build_phoneme_list, locate_dictionary
from utils.plot import distribution_to_figure
from utils.text_encoder import TokenTextEncoder


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    """
        Base class for data processing.
        1. *process* and *process_data_split*:
            process entire data, generate the train-test split (support parallel processing);
        2. *process_item*:
            process singe piece of data;
        3. *get_pitch*:
            infer the pitch using some algorithm;
        4. *get_align*:
            get the alignment using 'mel2ph' format (see https://arxiv.org/abs/1905.09263).
        5. phoneme encoder, voice encoder, etc.

        Subclasses should define:
        1. *load_metadata*:
            how to read multiple datasets from files;
        2. *train_item_names*, *valid_item_names*, *test_item_names*:
            how to split the dataset;
        3. load_ph_set:
            the phoneme set.
    """

    def __init__(self, data_dir=None, data_attrs=None):
        if data_dir is None:
            data_dir = hparams['raw_data_dir']
        if not isinstance(data_dir, list):
            data_dir = [data_dir]

        self.raw_data_dirs = [pathlib.Path(d) for d in data_dir]
        self.binary_data_dir = pathlib.Path(hparams['binary_data_dir'])
        self.data_attrs = [] if data_attrs is None else data_attrs

        self.binarization_args = hparams['binarization_args']
        self.augmentation_args = hparams.get('augmentation_args', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.spk_map = None
        self.spk_ids = hparams['spk_ids']
        self.speakers = hparams['speakers']
        self.build_spk_map()

        self.items = {}
        self.phone_encoder = TokenTextEncoder(vocab_list=build_phoneme_list())
        self.timestep = hparams['hop_size'] / hparams['audio_sample_rate']

        # load each dataset
        for ds_id, spk_id, data_dir in zip(range(len(self.raw_data_dirs)), self.spk_ids, self.raw_data_dirs):
            self.load_meta_data(pathlib.Path(data_dir), ds_id=ds_id, spk_id=spk_id)
        self.item_names = sorted(list(self.items.keys()))
        self._train_item_names, self._valid_item_names = self.split_train_valid_set()

        if self.binarization_args['shuffle']:
            random.seed(hparams['seed'])
            random.shuffle(self.item_names)

    def build_spk_map(self):
        assert isinstance(self.speakers, list), 'Speakers must be a list'
        assert len(self.speakers) == len(self.raw_data_dirs), \
            'Number of raw data dirs must equal number of speaker names!'
        if not self.spk_ids:
            self.spk_ids = list(range(len(self.raw_data_dirs)))
        else:
            assert len(self.spk_ids) == len(self.raw_data_dirs), \
                'Length of explicitly given spk_ids must equal the number of raw datasets.'
        assert max(self.spk_ids) < hparams['num_spk'], \
            f'Index in spk_id sequence {self.spk_ids} is out of range. All values should be smaller than num_spk.'

        self.spk_map = {}
        for spk_name, spk_id in zip(self.speakers, self.spk_ids):
            if spk_name in self.spk_map and self.spk_map[spk_name] != spk_id:
                raise ValueError(f'Invalid speaker ID assignment. Name \'{spk_name}\' is assigned '
                                 f'with different speaker IDs: {self.spk_map[spk_name]} and {spk_id}.')
            self.spk_map[spk_name] = spk_id

        print("| spk_map: ", self.spk_map)

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id, spk_id):
        raise NotImplementedError()

    def split_train_valid_set(self):
        """
        Split the dataset into training set and validation set.
        :return: train_item_names, valid_item_names
        """
        prefixes = set([str(pr) for pr in hparams['test_prefixes']])
        valid_item_names = set()
        # Add prefixes that specified speaker index and matches exactly item name to test set
        for prefix in deepcopy(prefixes):
            if prefix in self.item_names:
                valid_item_names.add(prefix)
                prefixes.remove(prefix)
        # Add prefixes that exactly matches item name without speaker id to test set
        for prefix in deepcopy(prefixes):
            matched = False
            for name in self.item_names:
                if name.split(':')[-1] == prefix:
                    valid_item_names.add(name)
                    matched = True
            if matched:
                prefixes.remove(prefix)
        # Add names with one of the remaining prefixes to test set
        for prefix in deepcopy(prefixes):
            matched = False
            for name in self.item_names:
                if name.startswith(prefix):
                    valid_item_names.add(name)
                    matched = True
            if matched:
                prefixes.remove(prefix)
        for prefix in deepcopy(prefixes):
            matched = False
            for name in self.item_names:
                if name.split(':')[-1].startswith(prefix):
                    valid_item_names.add(name)
                    matched = True
            if matched:
                prefixes.remove(prefix)

        if len(prefixes) != 0:
            warnings.warn(
                f'The following rules in test_prefixes have no matching names in the dataset: {sorted(prefixes)}',
                category=UserWarning
            )
            warnings.filterwarnings('default')

        valid_item_names = sorted(list(valid_item_names))
        assert len(valid_item_names) > 0, 'Validation set is empty!'
        train_item_names = [x for x in self.item_names if x not in set(valid_item_names)]
        assert len(train_item_names) > 0, 'Training set is empty!'

        return train_item_names, valid_item_names

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._valid_item_names

    def meta_data_iterator(self, prefix):
        if prefix == 'train':
            item_names = self.train_item_names
        else:
            item_names = self.valid_item_names
        for item_name in item_names:
            meta_data = self.items[item_name]
            yield item_name, meta_data

    def process(self):
        self.binary_data_dir.mkdir(parents=True, exist_ok=True)

        # Copy spk_map and dictionary to binary data dir
        spk_map_fn = self.binary_data_dir / 'spk_map.json'
        with open(spk_map_fn, 'w', encoding='utf-8') as f:
            json.dump(self.spk_map, f)
        shutil.copy(locate_dictionary(), self.binary_data_dir / 'dictionary.txt')
        self.check_coverage()

        # Process valid set and train set
        try:
            self.process_dataset('valid')
            self.process_dataset(
                'train',
                num_workers=int(self.binarization_args['num_workers']),
                apply_augmentation=any(args['enabled'] for args in self.augmentation_args.values())
            )
        except KeyboardInterrupt:
            exit(-1)

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
        x = sorted(phoneme_map.keys())
        values = [phoneme_map[k] for k in x]
        plt = distribution_to_figure(
            title='Phoneme Distribution Summary',
            x_label='Phoneme', y_label='Number of occurrences',
            items=x, values=values
        )
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

    def process_dataset(self, prefix, num_workers=0, apply_augmentation=False):
        args = []
        builder = IndexedDatasetBuilder(self.binary_data_dir, prefix=prefix, allowed_attr=self.data_attrs)
        lengths = []
        total_sec = 0
        total_raw_sec = 0

        for item_name, meta_data in self.meta_data_iterator(prefix):
            args.append([item_name, meta_data, self.binarization_args])

        aug_map = self.arrange_data_augmentation(self.meta_data_iterator(prefix)) if apply_augmentation else {}

        def postprocess(_item):
            nonlocal total_sec, total_raw_sec
            if _item is None:
                return
            builder.add_item(_item)
            lengths.append(_item['length'])
            total_sec += _item['seconds']
            total_raw_sec += _item['seconds']

            for task in aug_map.get(_item['name'], []):
                aug_item = task['func'](_item, **task['kwargs'])
                builder.add_item(aug_item)
                lengths.append(aug_item['length'])
                total_sec += aug_item['seconds']

        try:
            if num_workers > 0:
                # code for parallel processing
                for item in tqdm(
                        chunked_multiprocess_run(self.process_item, args, num_workers=num_workers),
                        total=len(list(self.meta_data_iterator(prefix)))
                ):
                    postprocess(item)
            else:
                # code for single cpu processing
                for a in tqdm(args):
                    item = self.process_item(*a)
                    postprocess(item)
        except KeyboardInterrupt:
            builder.finalize()
            raise

        builder.finalize()
        with open(self.binary_data_dir / f'{prefix}.lengths', 'wb') as f:
            # noinspection PyTypeChecker
            np.save(f, lengths)

        if apply_augmentation:
            print(f'| {prefix} total duration (before augmentation): {total_raw_sec:.2f}s')
            print(
                f'| {prefix} total duration (after augmentation): {total_sec:.2f}s ({total_sec / total_raw_sec:.2f}x)')
        else:
            print(f'| {prefix} total duration: {total_raw_sec:.2f}s')

    def arrange_data_augmentation(self, data_iterator):
        """
        Code for all types of data augmentation should be added here.
        """
        raise NotImplementedError()

    def process_item(self, item_name, meta_data, binarization_args):
        raise NotImplementedError()
