import collections
import csv
from dataclasses import dataclass

import dask
import librosa
import numpy
import scipy

from lib.conf.schema import DataSourceConfig, DataConfig
from preprocessing.binarizer_base import MetadataItem, BaseBinarizer, DataSample
from utils.infer_utils import resample_align_curve
from utils.pitch_utils import interp_f0
from utils.plot import distribution_to_figure

VARIANCE_ITEM_ATTRIBUTES = [
    "spk_id",  # index number of dataset/speaker, int64
    "languages",  # index numbers of phoneme languages, int64[T_ph,]
    "tokens",  # index numbers of phonemes, int64[T_ph,]
    "ph_dur",  # durations of phonemes, in number of frames, int64[T_ph,]
    "mel2ph",  # mel2ph format representing number of frames within each phone, int64[T_s,]
    "note_midi",  # note-level MIDI pitch, float32[T_n,]
    "note_rest",  # flags for rest notes, bool[T_n,]
    "note_dur",  # durations of notes, in number of frames, int64[T_n,]
    "note_glide",  # flags for glides, 0 = none, 1 = up, 2 = down, int64[T_n,]
    "mel2note",  # mel2ph format representing number of frames within each note, int64[T_s,]
    "base_pitch",  # interpolated and smoothed frame-level MIDI pitch, float32[T_s,]
    "pitch",  # actual pitch in semitones, float32[T_s,]
    "uv",  # unvoiced masks (only for objective evaluation metrics), bool[T_s,]
    "energy",  # frame-level RMS (dB), float32[T_s,]
    "breathiness",  # frame-level RMS of aperiodic parts (dB), float32[T_s,]
    "voicing",  # frame-level RMS of harmonic parts (dB), float32[T_s,]
    "tension",  # frame-level tension (logit), float32[T_s,]
]


@dataclass
class VarianceMetadataItem(MetadataItem):
    note_midi: list[int]
    note_rest: list[bool]
    note_dur: list[float]
    note_glide: list[int]
    manual_labels: dict


class VarianceBinarizer(BaseBinarizer):
    __data_attrs__ = VARIANCE_ITEM_ATTRIBUTES

    def __init__(self, data_config: DataConfig, *args, **kwargs):
        self.glide_map = data_config.glide_map
        super().__init__(data_config, *args, **kwargs)
        self.smooth_widths["midi"] = self.config.midi.smooth_width

    def load_metadata(self, data_source_config: DataSourceConfig):
        metadata_dict = collections.OrderedDict()
        raw_data_dir = data_source_config.raw_data_dir_resolved
        with open(raw_data_dir / "transcriptions.csv", "r", encoding="utf8") as f:
            transcriptions = list(csv.DictReader(f))
        for transcription in transcriptions:
            item_name = transcription["name"]
            spk_name = data_source_config.speaker
            spk_id = data_source_config.spk_id
            wav_fn = raw_data_dir / "wavs" / f"{item_name}.wav"
            if self.config.prefer_ds:
                ds_fn = raw_data_dir / "wavs" / f"{item_name}.ds"
                if ds_fn.exists() and ds_fn.is_file():
                    with open(ds_fn, "r", encoding="utf8") as ds_f:
                        ds_obj = ds_f.read()
                    if isinstance(ds_obj, list):
                        if len(ds_obj) == 0:
                            raise ValueError(f"Empty ds content encountered in \'{ds_fn}\'.")
                        elif len(ds_obj) > 1:
                            raise ValueError(
                                f"Unsupported multiple segments in \'{ds_fn}\' (found {len(ds_obj)} segments).")
                        ds_obj = ds_obj[0]
                    transcription = {
                        **transcription,
                        **{k: v for k, v in ds_obj if v}
                    }
                elif not wav_fn.exists():
                    raise ValueError(
                        f"Both waveform and ds files are missing in raw dataset \'{raw_data_dir.as_posix()}\':\n"
                        f"item {item_name}, wav file \'{wav_fn.as_posix()}\', ds file \'{ds_fn.as_posix()}\'"
                    )
            elif not wav_fn.exists():
                raise ValueError(
                    f"Waveform file missing in raw dataset \'{raw_data_dir.as_posix()}\':\n"
                    f"item {item_name}, wav file \'{wav_fn.as_posix()}\'."
                )
            succeeded, parse_results = self.parse_language_phoneme_sequences(
                transcription, language=data_source_config.language
            )
            if not succeeded:
                raise ValueError(
                    parse_results.format(raw_data_dir.as_posix(), item_name)
                )
            ph_text, lang_seq, ph_seq, ph_dur = parse_results
            if self.config.midi.used:
                note_text = transcription["note_seq"].split()
                note_midi = []
                note_rest = []
                for note in note_text:
                    if note == "rest":
                        note_midi.append(-1)
                        note_rest.append(True)
                    else:
                        note_midi.append(librosa.note_to_midi(note, round_midi=False))
                        note_rest.append(False)
                note_dur = []
                for dur in transcription["note_dur"].split():
                    dur_float = float(dur)
                    if dur_float < 0:
                        raise ValueError(
                            f"Negative note duration found in raw dataset '{raw_data_dir.as_posix()}':\n"
                            f"item '{item_name}', duration '{dur}'"
                        )
                    note_dur.append(dur_float)
                if len(note_midi) != len(note_dur):
                    raise ValueError(
                        f"Unaligned note_seq and note_dur found in raw dataset '{raw_data_dir.as_posix()}':\n"
                        f"item '{item_name}', note_seq length {len(note_midi)}, note_dur length {len(note_dur)}"
                    )
                note_glide = transcription.get("note_glide")
                if note_glide is None:
                    note_glide = [0] * len(note_glide)
                else:
                    note_glide = [self.glide_map[g] for g in note_glide.split()]
                    if len(note_glide) != len(note_glide):
                        raise ValueError(
                            f"Unaligned note_seq and note_glide found in raw dataset '{raw_data_dir.as_posix()}':\n"
                            f"item '{item_name}', note_seq length {len(note_midi)}, note_glide length {len(note_glide)}"
                        )
            else:
                note_midi = note_rest = note_dur = note_glide = None
            metadata_dict[item_name] = VarianceMetadataItem(
                item_name=item_name,
                spk_name=spk_name,
                spk_id=spk_id,
                ph_text=ph_text,
                lang_seq=lang_seq,
                ph_seq=ph_seq,
                ph_dur=ph_dur,
                note_midi=note_midi,
                note_rest=note_rest,
                note_dur=note_dur,
                note_glide=note_glide,
                wav_fn=wav_fn,
                manual_labels=transcription
            )
        return metadata_dict

    def check_coverage(self):
        super().check_coverage()
        if not self.config.midi.used:
            return

        # MIDI pitch distribution summary
        midi_map = {}
        for item in self.train_items:
            item: VarianceMetadataItem
            for midi in item.note_midi:
                if midi < 0:
                    continue
                midi = round(midi)
                if midi in midi_map:
                    midi_map[midi] += 1
                else:
                    midi_map[midi] = 1

        print("===== MIDI Pitch Distribution Summary =====")
        for i, key in enumerate(sorted(midi_map.keys())):
            if i == len(midi_map) - 1:
                end = "\n"
            elif i % 10 == 9:
                end = ",\n"
            else:
                end = ", "
            print(f"\'{librosa.midi_to_note(key, unicode=False)}\': {midi_map[key]}", end=end)

        # Draw graph.
        midis = sorted(midi_map.keys())
        notes = [librosa.midi_to_note(m, unicode=False) for m in range(midis[0], midis[-1] + 1)]
        plt = distribution_to_figure(
            title="MIDI Pitch Distribution Summary",
            x_label="MIDI Key", y_label="Number of occurrences",
            items=notes, values=[midi_map.get(m, 0) for m in range(midis[0], midis[-1] + 1)]
        )
        filename = self.binary_data_dir / "midi_distribution.jpg"
        plt.savefig(fname=filename,
                    bbox_inches="tight",
                    pad_inches=0.25)
        print(f"| save summary to \'{filename}\'")

        if self.config.midi.with_glide:
            # Glide type distribution summary
            glide_inverted_idx = {
                idx: g
                for g, idx in self.glide_map.items()
            }
            glide_count = {
                g: 0
                for g in self.glide_map
            }
            for item in self.train_items:
                for glide in item.note_glide:
                    glide_count[glide_inverted_idx[glide]] += 1

            print("===== Glide Type Distribution Summary =====")
            for i, key in enumerate(sorted(glide_count.keys(), key=lambda k: self.glide_map[k])):
                if i == len(glide_count) - 1:
                    end = "\n"
                elif i % 10 == 9:
                    end = ",\n"
                else:
                    end = ", "
                print(f"\'{key}\': {glide_count[key]}", end=end)

            if any(n == 0 for _, n in glide_count.items()):
                raise RuntimeError(
                    f"Missing glide types in dataset: "
                    f"{sorted([g for g, n in glide_count.items() if n == 0], key=lambda k: self.glide_map[k])}"
                )

    def process_item(self, item: VarianceMetadataItem, augmentation=False) -> list[DataSample]:
        label = item.manual_labels
        length = round(sum(item.ph_dur) / self.timestep)
        ph_dur_sec = numpy.array(item.ph_dur, dtype=numpy.float32)
        ph_dur = self.sec_dur_to_frame_dur(ph_dur_sec)
        mel2ph = self.get_mel2ph(ph_dur_sec, length)
        waveform = self.load_waveform(item.wav_fn)
        f0 = self.try_load_curve_from_label_if_allowed(
            label, "f0_seq", "f0_timestep", length
        )
        if f0 is not None:
            f0, uv = dask.delayed(lambda f: interp_f0(f, f == 0), nout=2)(f0)
        else:
            f0, uv = self.get_f0(waveform, length)
        pitch = dask.delayed(librosa.hz_to_midi)(f0)
        if self.config.midi.used:
            note_rest = numpy.array(item.note_rest, dtype=bool)
            note_midi = self.interp_midi(numpy.array(item.note_midi, dtype=numpy.float32), note_rest)
            note_dur_sec = numpy.array(item.note_dur, dtype=numpy.float32)
            note_dur = self.sec_dur_to_frame_dur(note_dur_sec)
            mel2note = self.get_mel2ph(note_dur_sec, length)
            base_pitch = self.get_base_pitch(note_midi, mel2note)
        else:
            note_midi = note_rest = note_dur = mel2note = base_pitch = None
        energy = self.try_load_curve_from_label_if_allowed(
            label, "energy_seq", "energy_timestep", length
        )
        if energy is None:
            energy = self.get_energy(waveform, length, smooth_fn_name="energy")
        harmonic, noise = self.harmonic_noise_separation(waveform, f0)
        breathiness = self.try_load_curve_from_label_if_allowed(
            label, "breathiness", "breathiness_timestep", length
        )
        if breathiness is None:
            breathiness = self.get_energy(noise, length, smooth_fn_name="breathiness")
        voicing = self.try_load_curve_from_label_if_allowed(
            label, "voicing", "voicing_timestep", length
        )
        if voicing is None:
            voicing = self.get_energy(noise, length, smooth_fn_name="voicing")
        tension = self.try_load_curve_from_label_if_allowed(
            label, "tension", "tension_timestep", length
        )
        if tension is None:
            base_harmonic = self.get_kth_harmonic(harmonic, f0, k=0)
            tension = self.get_tension(harmonic, base_harmonic, length)

        data = {
            "spk_id": item.spk_id,
            "languages": numpy.array(item.lang_seq, dtype=numpy.int64),
            "tokens": numpy.array(item.ph_seq, dtype=numpy.int64),
            "ph_dur": ph_dur,
            "mel2ph": mel2ph,
            "pitch": pitch,
            "uv": uv,
        }
        if self.config.midi.used:
            data.update({
                "note_midi": note_midi,
                "note_rest": note_rest,
                "note_dur": note_dur,
                "mel2note": mel2note,
                "base_pitch": base_pitch,
            })
            if self.config.midi.with_glide:
                data["note_glide"] = numpy.array(item.note_glide, dtype=numpy.int64)
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

    def try_load_curve_from_label_if_allowed(self, label: dict, curve_key: str, timestep_key: str, length: int):
        if not self.config.prefer_ds:
            return None
        curve_text = label.get(curve_key)
        if curve_text is None:
            return None
        curve = self.resample_align_curve(
            numpy.array(curve_text.split(), numpy.float32),
            original_timestep=float(label[timestep_key]),
            target_timestep=self.timestep,
            align_length=length
        )
        return curve

    @dask.delayed
    def sec_dur_to_frame_dur(self, dur_sec: numpy.ndarray):
        dur_frame = numpy.diff(
            numpy.round(numpy.cumsum(dur_sec, axis=0) / self.timestep + 0.5).astype(numpy.int64),
            axis=0, prepend=numpy.array([0])
        )
        return dur_frame

    @dask.delayed
    def resample_align_curve(
            self, curve: numpy.ndarray,
            original_timestep: float, target_timestep: float, align_length: int
    ):
        resample_aligned_curve = resample_align_curve(
            curve, original_timestep, target_timestep, align_length
        )
        return resample_aligned_curve

    @dask.delayed
    def interp_midi(self, note_midi: numpy.ndarray, note_rest: numpy.ndarray):
        interp_func = scipy.interpolate.interp1d(
            numpy.where(~note_rest)[0], note_midi[~note_rest],
            kind="nearest", fill_value="extrapolate"
        )
        note_midi[note_rest] = interp_func(numpy.where(note_rest)[0])
        return note_midi

    @dask.delayed
    def get_base_pitch(self, note_midi: numpy.ndarray, mel2note: numpy.ndarray):
        frame_midi_pitch = numpy.take_along_axis(
            numpy.pad(note_midi, (1, 0), mode="constant"), indices=mel2note, axis=0
        )
        smoothed_midi_pitch = self.smooth_curve(frame_midi_pitch, smooth_fn_name="midi")
        return smoothed_midi_pitch
