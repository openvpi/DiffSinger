import collections
import copy
import csv
import pathlib
import random
from dataclasses import dataclass

import dask
import librosa
import numpy

from lib.conf.schema import DataSourceConfig
from .binarizer_base import MetadataItem, BaseBinarizer, DataSample

ACOUSTIC_ITEM_ATTRIBUTES = [
    "spk_id",
    "languages",
    "tokens",
    "mel",
    "mel2ph",
    "f0",
    "energy",
    "breathiness",
    "voicing",
    "tension",
    "key_shift",
    "speed",
]


@dataclass
class AcousticMetadataItem(MetadataItem):
    wav_fn: pathlib.Path
    ph_dur: list[float]


class AcousticBinarizer(BaseBinarizer):
    __data_attrs__ = ACOUSTIC_ITEM_ATTRIBUTES

    def load_metadata(self, data_source_config: DataSourceConfig):
        metadata_dict = collections.OrderedDict()
        raw_data_dir = data_source_config.raw_data_dir_resolved
        with open(raw_data_dir / "transcriptions.csv", "r", encoding="utf8") as f:
            transcriptions = list(csv.DictReader(f))
        for transcription in transcriptions:
            item_name = transcription["name"]
            spk_name = data_source_config.speaker
            spk_id = data_source_config.spk_id
            ph_text = transcription["ph_seq"].split()
            lang_seq = []
            for ph in ph_text:
                if self.phoneme_dictionary.is_cross_lingual(ph):
                    if "/" in ph:
                        lang_name = ph.split("/")[0]
                        if lang_name not in self.lang_map:
                            raise ValueError(
                                f"Invalid language tag found in raw dataset '{raw_data_dir.as_posix()}':\n"
                                f"item '{item_name}', phoneme '{ph}'"
                            )
                        # noinspection PyUnresolvedReferences
                        lang_id = self.lang_map[lang_name]
                    else:
                        # noinspection PyUnresolvedReferences
                        lang_id = self.lang_map[data_source_config.language]
                else:
                    lang_id = 0
                lang_seq.append(lang_id)
            ph_seq = self.phoneme_dictionary.encode(ph_text, lang=data_source_config.language)
            wav_fn = raw_data_dir / "wavs" / f"{item_name}.wav"
            ph_dur = []
            for dur in transcription["ph_dur"].split():
                dur_float = float(dur)
                if dur_float < 0:
                    raise ValueError(
                        f"Negative duration found in raw dataset '{raw_data_dir.as_posix()}':\n"
                        f"item '{item_name}', duration '{dur}'"
                    )
                ph_dur.append(dur_float)
            if len(ph_seq) != len(ph_dur):
                raise ValueError(
                    f"Unaligned ph_seq and ph_dur found in raw dataset '{raw_data_dir.as_posix()}':\n"
                    f"item '{item_name}', ph_seq length {len(ph_seq)}, ph_dur length {len(ph_dur)}"
                )
            metadata_dict[item_name] = AcousticMetadataItem(
                item_name=item_name,
                spk_name=spk_name,
                spk_id=spk_id,
                lang_seq=lang_seq,
                ph_text=" ".join(ph_text),
                ph_seq=ph_seq,
                wav_fn=wav_fn,
                ph_dur=ph_dur,
            )
        test_prefixes = data_source_config.test_prefixes
        for prefix in test_prefixes:
            if prefix in metadata_dict:
                self.valid_items.append(metadata_dict.pop(prefix))
            else:
                hit = False
                for key in list(metadata_dict.keys()):
                    if key.startswith(prefix):
                        self.valid_items.append(metadata_dict.pop(key))
                        hit = True
                if not hit:
                    # TODO: change to warning
                    raise ValueError(
                        f"Test prefix does not hit any item in raw dataset '{raw_data_dir.as_posix()}':\n"
                        f"prefix '{prefix}'"
                    )
        for item in metadata_dict.values():
            self.train_items.append(item)

    def process_item(self, item: AcousticMetadataItem, augmentation=False) -> list[DataSample]:
        ph_dur = numpy.array(item.ph_dur, dtype=numpy.float32)
        waveform, _ = librosa.load(item.wav_fn, sr=self.config.features.audio_sample_rate, mono=True)
        mel, length = self.get_mel(waveform)
        mel2ph = self.get_mel2ph(ph_dur, length)
        f0, uv = self.get_f0(waveform, length)
        energy = self.get_energy(waveform, length, smooth_fn_name="energy")
        hn_sep_method = self.config.extractors.harmonic_noise_separation.method
        if hn_sep_method == "world":
            sp, ap = self.world_analyze(waveform, f0)
            noise = self.world_synthesize_aperiodic(f0, sp, ap)
            harmonic = self.world_synthesize_harmonics(f0, sp, ap)
        elif hn_sep_method == "vr":
            harmonic, noise = self.harmonic_noise_separation(waveform)
        else:
            raise ValueError(f"Unknown harmonic-noise separation method: {hn_sep_method}")
        breathiness = self.get_energy(noise, length, smooth_fn_name="breathiness")
        voicing = self.get_energy(harmonic, length, smooth_fn_name="voicing")
        base_harmonic = self.get_kth_harmonic(harmonic, f0, k=0)
        tension = self.get_tension(harmonic, base_harmonic, length)

        data = {
            "spk_id": item.spk_id,
            "languages": numpy.array(item.lang_seq, dtype=numpy.int64),
            "tokens": numpy.array(item.ph_seq, dtype=numpy.int64),
            "mel": mel,
            "mel2ph": mel2ph,
            "f0": f0,
            "key_shift": 0.,
            "speed": 1.,
        }
        variance_names = []
        if self.config.features.energy.used:
            data["energy"] = energy
            variance_names.append("energy")
        if self.config.features.breathiness.used:
            data["breathiness"] = breathiness
            variance_names.append("breathiness")
        if self.config.features.voicing.used:
            data["voicing"] = voicing
            variance_names.append("voicing")
        if self.config.features.tension.used:
            data["tension"] = tension
            variance_names.append("tension")
        length, data = dask.compute(length, data)

        if uv.compute().all():
            print(f"Skipped \'{item.item_name}\': empty gt f0")
            return []
        sample = DataSample(
            name=item.item_name,
            spk_name=item.spk_name,
            spk_id=item.spk_id,
            ph_text=item.ph_text,
            length=length,
            augmented=False,
            data=data
        )

        samples = [sample]
        if not augmentation:
            return samples

        augmentation_params: list[tuple[int, float, float]] = []  # (ori_idx, shift, speed)
        shift_scale = self.config.augmentation.random_pitch_shifting.scale
        if self.config.augmentation.random_pitch_shifting.enabled:
            shift_ids = [0] * int(shift_scale)
            if numpy.random.rand() < shift_scale % 1:
                shift_ids.append(0)
        else:
            shift_ids = []
        key_shift_min, key_shift_max = self.config.augmentation.random_pitch_shifting.range
        for i in shift_ids:
            rand = random.uniform(-1, 1)
            if rand < 0:
                shift = key_shift_min * abs(rand)
            else:
                shift = key_shift_max * rand
            augmentation_params.append((i, shift, 1))
        stretch_scale = self.config.augmentation.random_time_stretching.scale
        if self.config.augmentation.random_time_stretching.enabled:
            randoms = numpy.random.rand(1 + len(shift_ids))
            stretch_ids = list(numpy.where(randoms < stretch_scale % 1)[0])
            if stretch_scale > 1:
                stretch_ids.extend([0] * int(stretch_scale))
                stretch_ids.sort()
        else:
            stretch_ids = []
        speed_min, speed_max = self.config.augmentation.random_time_stretching.range
        for i in stretch_ids:
            # Uniform distribution in log domain
            speed = speed_min * (speed_max / speed_min) ** random.random()
            if i == 0:
                augmentation_params.append((i, 0, speed))
            else:
                ori_idx, shift, _ = augmentation_params[i - 1]
                augmentation_params[i - 1] = (ori_idx, shift, speed)

        if not augmentation_params:
            return samples

        length_transforms = []
        data_transforms = []
        for ori_idx, shift, speed in augmentation_params:
            mel_transform, length_transform = self.get_mel(waveform, shift=shift, speed=speed)
            mel2ph_transform = self.get_mel2ph(ph_dur / speed, length_transform)
            f0_transform = self.resize_curve(f0.compute() * 2 ** (shift / 12), length_transform)
            v_transform = {
                v_name: self.resize_curve(data[v_name], length_transform)
                for v_name in variance_names
            }
            data_transform = samples[ori_idx].data.copy()
            data_transform["mel"] = mel_transform
            data_transform["mel2ph"] = mel2ph_transform
            data_transform["f0"] = f0_transform
            data_transform["key_shift"] = shift
            data_transform["speed"] = (  # real speed
                dask.delayed(lambda x: samples[ori_idx].length / x)(length_transform)
            )
            for v_name in variance_names:
                data_transform[v_name] = v_transform[v_name]
            length_transforms.append(length_transform)
            data_transforms.append(data_transform)

        length_transforms, data_transforms = dask.compute(length_transforms, data_transforms)
        for i, (ori_idx, shift, speed) in enumerate(augmentation_params):
            sample_transform = copy.copy(samples[ori_idx])
            sample_transform.length = length_transforms[i]
            sample_transform.data = data_transforms[i]
            sample_transform.augmented = True
            samples.append(sample_transform)

        return samples

    @dask.delayed
    def resize_curve(self, curve: numpy.ndarray, target_length: int):
        original_length = len(curve)
        original_indices = numpy.linspace(0, original_length - 1, num=original_length)
        target_indices = numpy.linspace(0, original_length - 1, num=target_length)
        interpolated_curve = numpy.interp(target_indices, original_indices, curve)
        return interpolated_curve
