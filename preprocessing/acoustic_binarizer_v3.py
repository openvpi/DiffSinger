import collections
import copy
import csv
import random
from dataclasses import dataclass

import dask
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
    pass


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
            succeeded, parse_results = self.parse_language_phoneme_sequences(
                transcription, language=data_source_config.language
            )
            if not succeeded:
                raise ValueError(
                    parse_results.format(raw_data_dir.as_posix(), item_name)
                )
            ph_text, lang_seq, ph_seq, ph_dur = parse_results
            wav_fn = raw_data_dir / "wavs" / f"{item_name}.wav"
            if not wav_fn.exists():
                raise ValueError(
                    f"Waveform file missing in raw dataset \'{raw_data_dir.as_posix()}\':\n"
                    f"item {item_name}, wav file \'{wav_fn.as_posix()}\'."
                )
            metadata_dict[item_name] = AcousticMetadataItem(
                item_name=item_name,
                estimated_duration=sum(ph_dur),
                spk_name=spk_name,
                spk_id=spk_id,
                ph_text=ph_text,
                lang_seq=lang_seq,
                ph_seq=ph_seq,
                ph_dur=ph_dur,
                wav_fn=wav_fn,
            )
        return metadata_dict

    def process_item(self, item: AcousticMetadataItem, augmentation=False) -> list[DataSample]:
        waveform = self.load_waveform(item.wav_fn)
        ph_dur = numpy.array(item.ph_dur, dtype=numpy.float32)
        mel, length = self.get_mel(waveform)
        mel2ph = self.get_mel2ph(ph_dur, length)
        f0, uv = self.get_f0(waveform, length)
        energy = self.get_energy(waveform, length, smooth_fn_name="energy")
        harmonic, noise = self.harmonic_noise_separation(waveform, f0)
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
