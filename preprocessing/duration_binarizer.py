import collections
import csv
from dataclasses import dataclass

import dask
import librosa
import numpy
import torch

from lib.conf.schema import DataSourceConfig
from preprocessing.binarizer_base import MetadataItem, BaseBinarizer, DataSample
from lib.feature.pitch import interp_f0


@dataclass
class DurationMetadataItem(MetadataItem):
    ph_num: list[int]
    external_labels: dict


class DurationBinarizer(BaseBinarizer):
    def load_metadata(self, data_source_config: DataSourceConfig) -> dict[str, MetadataItem]:
        metadata_dict = collections.OrderedDict()
        raw_data_dir = data_source_config.raw_data_dir_resolved
        with open(raw_data_dir / "transcriptions.csv", "r", encoding="utf8") as f:
            transcriptions = list(csv.DictReader(f))
        for transcription in transcriptions:
            item_name = transcription["name"]
            spk_name = data_source_config.speaker
            spk_id = data_source_config.spk_id
            wav_fn = raw_data_dir / "wavs" / f"{item_name}.wav"
            loaded, transcription = self.try_load_external_labels_if_allowed(raw_data_dir, item_name, transcription)
            if not loaded and not wav_fn.exists():
                if self.config.prefer_ds:
                    raise ValueError(
                        f"Both waveform and external labels are missing in raw dataset '{raw_data_dir.as_posix()}':\n"
                        f"item {item_name}, wav file '{wav_fn.as_posix()}'"
                    )
                else:
                    raise ValueError(
                        f"Waveform file missing in raw dataset '{raw_data_dir.as_posix()}':\n"
                        f"item {item_name}, wav file '{wav_fn.as_posix()}'."
                    )
            succeeded, parse_results = self.parse_language_phoneme_sequences(
                transcription, language=data_source_config.language
            )
            if not succeeded:
                raise ValueError(
                    parse_results.format(raw_data_dir.as_posix(), item_name)
                )
            ph_text, lang_seq, ph_seq, ph_dur = parse_results
            ph_num = []
            for num in transcription["ph_num"].split():
                num_int = int(num)
                if num_int < 0:
                    raise ValueError(
                        f"Negative phoneme division found in raw dataset '{raw_data_dir.as_posix()}'\n",
                        f"item {item_name}, ph_num '{num}'"
                    )
                ph_num.append(num_int)
            if sum(ph_num) != len(ph_seq):
                raise ValueError(
                    f"Phoneme division does not cover all phonemes in raw dataset '{raw_data_dir.as_posix()}'\n",
                    f"item {item_name}, sum(ph_num) = {sum(ph_num)}, len(ph_seq) = {len(ph_seq)}\n"
                )
            metadata_dict[item_name] = DurationMetadataItem(
                item_name=item_name,
                estimated_duration=sum(ph_dur),
                spk_name=spk_name,
                spk_id=spk_id,
                ph_text=ph_text,
                lang_seq=lang_seq,
                ph_seq=ph_seq,
                ph_dur=ph_dur,
                ph_num=ph_num,
                wav_fn=wav_fn,
                external_labels=transcription
            )
        return metadata_dict

    def process_item(self, item: DurationMetadataItem, augmentation=False) -> list[DataSample]:
        label = item.external_labels
        length = round(sum(item.ph_dur) / self.timestep)
        ph_dur_sec = numpy.array(item.ph_dur, dtype=numpy.float32)
        ph_dur = self.sec_dur_to_frame_dur(ph_dur_sec)
        mel2ph = self.get_mel2ph(ph_dur_sec, length)
        ph_num = numpy.array(item.ph_num, dtype=numpy.int64)
        ph2word = self.get_ph2word(ph_num)
        waveform = self.load_waveform(item.wav_fn)
        f0 = self.try_load_curve_from_label_if_allowed(
            label, "f0_seq", "f0_timestep", length
        )
        if f0 is not None:
            f0, uv = dask.delayed(lambda f: interp_f0(f, f == 0), nout=2)(f0)
        else:
            f0, uv = self.get_f0(waveform, length)
        pitch = dask.delayed(librosa.hz_to_midi)(f0)
        ph_midi = self.get_ph_midi(ph_dur, mel2ph, pitch)

        data = {
            "spk_id": item.spk_id,
            "languages": numpy.array(item.lang_seq, dtype=numpy.int64),
            "tokens": numpy.array(item.ph_seq, dtype=numpy.int64),
            "ph_dur": ph_dur,
            "ph_midi": ph_midi,
            "ph2word": ph2word,
        }
        (data,) = dask.compute(data)
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

        # No augmentation supported yet
        return [sample]

    @dask.delayed
    def get_ph2word(self, ph_num: numpy.ndarray):
        ph_num = torch.from_numpy(ph_num).to(self.device)[None]
        ph2word = self.lr(ph_num)
        return ph2word[0].cpu().numpy()

    @dask.delayed
    def get_ph_midi(self, ph_dur: numpy.ndarray, mel2ph: numpy.ndarray, pitch: numpy.ndarray):
        mel2dur = numpy.take_along_axis(
            numpy.pad(ph_dur, (1, 0), mode="constant", constant_values=(1, 1)),
            indices=mel2ph, axis=0
        )  # frame-level phone duration
        ph_midi = numpy.zeros(ph_dur.shape[0] + 1, dtype=pitch.dtype)
        numpy.add.at(ph_midi, mel2ph, pitch / mel2dur)
        ph_midi = ph_midi[1:]
        return ph_midi
