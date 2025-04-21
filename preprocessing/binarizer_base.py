import abc
import pathlib
from dataclasses import dataclass

import dask
import librosa
import numpy
import torch
import tqdm

from lib.conf.schema import DataConfig, BinarizerConfig
from lib.conf.schema import DataSourceConfig
from lib.feature.binarizer_utils import (
    get_pitch_parselmouth,
    get_pitch_harvest,
    get_energy_librosa,
    get_tension, SinusoidalSmoothingConv1d,
)
from lib.feature.decomposed_waveform import world_analyze, world_synthesize_harmonics, world_synthesize_aperiodic, \
    get_kth_harmonic
from lib.feature.mel_spec import StretchableMelSpectrogram
from lib.functional import dur_to_mel2ph
from modules.fastspeech.tts_modules import LengthRegulator
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.infer_utils import resample_align_curve
from utils.multiprocess_utils import chunked_multiprocess_run
from utils.phoneme_utils import PhonemeDictionary
from utils.plot import distribution_to_figure


@dataclass
class MetadataItem(abc.ABC):
    item_name: str
    estimated_duration: float
    spk_name: str
    spk_id: int
    ph_text: str
    lang_seq: list[int]
    ph_seq: list[int]
    ph_dur: list[float]
    wav_fn: pathlib.Path


@dataclass
class DataSample:
    name: str
    spk_name: str
    spk_id: int
    ph_text: str
    length: int
    augmented: bool
    data: dict[str, int | float | numpy.ndarray]


class BaseBinarizer(abc.ABC):
    __data_attrs__: list[str] = None

    def __init__(self, data_config: DataConfig, binarizer_config: BinarizerConfig):
        self.phoneme_dictionary = PhonemeDictionary(
            dictionaries=data_config.dictionaries,
            extra_phonemes=data_config.extra_phonemes,
            merged_groups=data_config.merged_phoneme_groups
        )
        self.spk_map = data_config.spk_map
        self.lang_map = data_config.lang_map
        self.sources = data_config.sources
        self.config = binarizer_config
        self.binary_data_dir: pathlib.Path = binarizer_config.binary_data_dir_resolved
        self.binary_data_dir.mkdir(parents=True, exist_ok=True)
        self.timestep = binarizer_config.features.hop_size / binarizer_config.features.audio_sample_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = LengthRegulator()
        # Lazy-initialized modules
        self.mel_spec = None
        self.rmvpe = None
        self.hn_sep_model = None
        self.smooth_fns = {}
        self.smooth_widths = {
            "energy": self.config.features.energy.smooth_width,
            "breathiness": self.config.features.breathiness.smooth_width,
            "voicing": self.config.features.voicing.smooth_width,
            "tension": self.config.features.tension.smooth_width
        }

        self.valid_items: list[MetadataItem] = []
        self.train_items: list[MetadataItem] = []

    @abc.abstractmethod
    def load_metadata(self, data_source_config: DataSourceConfig) -> dict[str, MetadataItem]:
        pass

    @abc.abstractmethod
    def process_item(self, item: MetadataItem, augmentation=False) -> list[DataSample]:
        pass

    def try_load_external_labels_if_allowed(self, raw_data_dir, item_name, transcription) -> tuple[bool, dict]:
        """
        Try to load external labels from the waveform file if allowed (prefer_ds=true).
        :return: (loaded, overridden transcription)
        """
        if self.config.prefer_ds:
            ds_fn = raw_data_dir / "wavs" / f"{item_name}.ds"
            if ds_fn.exists() and ds_fn.is_file():
                with open(ds_fn, "r", encoding="utf8") as ds_f:
                    ds_obj = ds_f.read()
                if isinstance(ds_obj, list):
                    if len(ds_obj) == 0:
                        raise ValueError(f"Empty ds content encountered in '{ds_fn}'.")
                    elif len(ds_obj) > 1:
                        raise ValueError(
                            f"Unsupported multiple segments in '{ds_fn}' (found {len(ds_obj)} segments).")
                    ds_obj = ds_obj[0]
                transcription = {
                    **transcription,
                    **{k: v for k, v in ds_obj if v}
                }
                return True, transcription
        return False, transcription

    def parse_language_phoneme_sequences(self, transcription: dict, language: str) -> tuple[bool, tuple | str]:
        """
        Parse the language and phoneme sequences from transcriptions
        :param transcription: dict
        :param language: language context
        :return: (succeeded, results).
          if succeeded, results = (ph_text, lang_seq, ph_seq, ph_dur);
          if failed, results = error message template with args (raw_data_dir, item_name)
        """
        ph_text = transcription["ph_seq"].split()
        lang_seq = []
        for ph in ph_text:
            if self.phoneme_dictionary.is_cross_lingual(ph):
                if "/" in ph:
                    lang_name = ph.split("/")[0]
                    if lang_name not in self.lang_map:
                        return False, (
                            "Invalid language tag found in raw dataset '{}':\n"
                            f"item '{{}}', phoneme '{ph}'"
                        )
                    # noinspection PyUnresolvedReferences
                    lang_id = self.lang_map[lang_name]
                else:
                    # noinspection PyUnresolvedReferences
                    lang_id = self.lang_map[data_source_config.language]
            else:
                lang_id = 0
            lang_seq.append(lang_id)
        ph_seq = self.phoneme_dictionary.encode(ph_text, lang=language)
        ph_dur = []
        for dur in transcription["ph_dur"].split():
            dur_float = float(dur)
            if dur_float < 0:
                return False, (
                    "Negative phoneme duration found in raw dataset '{}':\n"
                    f"item '{{}}', duration '{dur}'"
                )
            ph_dur.append(dur_float)
        if len(ph_seq) != len(ph_dur):
            raise ValueError(
                "Unaligned ph_seq and ph_dur found in raw dataset '{}':\n"
                f"item '{{}}', ph_seq length {len(ph_seq)}, ph_dur length {len(ph_dur)}"
            )
        ph_text = " ".join(ph_text)
        return True, (ph_text, lang_seq, ph_seq, ph_dur)

    def split_train_and_valid_set(self, metadata_dict: dict[str, MetadataItem], prefixes: list[str]):
        for prefix in prefixes:
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
                        f"Test prefix does not hit any item:\n"
                        f"prefix '{prefix}'"
                    )
        for item in metadata_dict.values():
            self.train_items.append(item)

    def check_coverage(self):
        # TODO refactor this
        # Group by phonemes in the dictionary.
        ph_idx_required = set(range(1, len(self.phoneme_dictionary)))
        ph_idx_occurred = set()
        ph_idx_count_map = {
            idx: 0
            for idx in ph_idx_required
        }

        # Load and count those phones that appear in the actual data
        for item in self.train_items:
            ph_idx_occurred.update(item.ph_seq)
            for idx in item.ph_seq:
                ph_idx_count_map[idx] += 1
        ph_count_map = {
            self.phoneme_dictionary.decode_one(idx, scalar=False): count
            for idx, count in ph_idx_count_map.items()
        }

        def display_phoneme(phoneme):
            if isinstance(phoneme, tuple):
                return f"({', '.join(phoneme)})"
            return phoneme

        print("===== Phoneme Distribution Summary =====")
        keys = sorted(ph_count_map.keys(), key=lambda v: v[0] if isinstance(v, tuple) else v)
        for i, key in enumerate(keys):
            if i == len(ph_count_map) - 1:
                end = "\n"
            elif i % 10 == 9:
                end = ",\n"
            else:
                end = ", "
            key_disp = display_phoneme(key)
            print(f"{key_disp}: {ph_count_map[key]}", end=end)

        # Draw graph.
        xs = [display_phoneme(k) for k in keys]
        ys = [ph_count_map[k] for k in keys]
        plt = distribution_to_figure(
            title="Phoneme Distribution Summary",
            x_label="Phoneme", y_label="Number of occurrences",
            items=xs, values=ys, rotate=len(self.lang_map) > 1
        )
        filename = self.binary_data_dir / "phoneme_distribution.jpg"
        plt.savefig(fname=filename,
                    bbox_inches="tight",
                    pad_inches=0.25)
        print(f"| save summary to '{filename}'")

        # Check unrecognizable or missing phonemes
        if ph_idx_occurred != ph_idx_required:
            missing_phones = sorted({
                self.phoneme_dictionary.decode_one(idx, scalar=False)
                for idx in ph_idx_required.difference(ph_idx_occurred)
            }, key=lambda v: v[0] if isinstance(v, tuple) else v)
            raise RuntimeError(
                f"The following phonemes are not covered in transcriptions: {missing_phones}"
            )

    def free_lazy_modules(self):
        """
        The lazy-initialized PyTorch modules should be freed before multiprocessing,
        because of CUDA IPC (shared memory) issues on Windows platforms.
        Reference:
          - https://github.com/pytorch/pytorch/issues/100358
          - https://github.com/Xiao-Chenguang/FedMind/issues/61
        """
        self.mel_spec = None
        self.rmvpe = None
        self.hn_sep_model = None
        self.smooth_fns.clear()

    def process_items(self, items: list[MetadataItem], prefix: str, augmentation=False, multiprocessing=True):
        builder = IndexedDatasetBuilder(
            path=self.binary_data_dir, prefix=prefix, allowed_attr=self.__data_attrs__
        )
        if multiprocessing and self.config.num_workers > 0:
            self.free_lazy_modules()
            iterable = chunked_multiprocess_run(
                self.process_item, [(item, augmentation) for item in items], num_workers=self.config.num_workers
            )
        else:
            iterable = (self.process_item(item, augmentation) for item in items)
        names = []
        ph_texts = []
        spk_ids = []
        spk_names = []
        lengths = []
        attr_lengths = {}
        total_duration_before_aug = {k: 0 for k in self.spk_map}
        total_duration = {k: 0 for k in self.spk_map}
        for samples in tqdm.tqdm(iterable, total=len(items), desc=f"Processing {prefix} items"):
            for sample in samples:
                builder.add_item(sample.data)
                names.append(sample.name)
                ph_texts.append(sample.ph_text)
                spk_ids.append(sample.spk_id)
                spk_names.append(sample.spk_name)
                lengths.append(sample.length)
                for k, v in sample.data.items():
                    if isinstance(v, numpy.ndarray):
                        if k not in attr_lengths:
                            attr_lengths[k] = []
                        attr_lengths[k].append(v.shape[0])
                duration = sample.length * self.timestep
                if not sample.augmented:
                    total_duration_before_aug[sample.spk_name] += duration
                total_duration[sample.spk_name] += duration
        builder.finalize()
        metadata = {
            "names": names,
            "ph_texts": ph_texts,
            "spk_ids": spk_ids,
            "spk_names": spk_names,
            "lengths": lengths,
            **attr_lengths
        }
        if prefix == "train":
            metadata.pop("names")
            metadata.pop("ph_texts")
            metadata.pop("spk_names")
        metadata = {
            k: numpy.array(v)
            for k, v in metadata.items()
        }
        with open(self.binary_data_dir / f"{prefix}.info.npz", "wb") as f:
            numpy.savez(f, **metadata)
        dur_before = sum(total_duration_before_aug.values())
        dur_after = sum(total_duration.values())
        if augmentation:
            print(f"| {prefix} total duration (before augmentation): {dur_before:.2f}s")
            print(
                f"| {prefix} respective duration (before augmentation): "
                + ", ".join(f"{k}={v:.2f}s" for k, v in total_duration_before_aug.items() if v > 0)
            )
            print(
                f"| {prefix} total duration (after augmentation): "
                f"{dur_after:.2f}s ({dur_after / dur_before:.2f}x)"
            )
            print(
                f"| {prefix} respective duration (after augmentation): "
                + ", ".join(f"{k}={v:.2f}s" for k, v in total_duration.items())
            )
        else:
            print(f"| {prefix} total duration: {dur_before:.2f}s")
            print(
                f"| {prefix} respective duration: "
                + ", ".join(f"{k}={v:.2f}s" for k, v in total_duration_before_aug.items() if v > 0)
            )

    def process(self):
        for source in self.sources:
            metadata_dict = self.load_metadata(source)
            self.split_train_and_valid_set(metadata_dict, source.test_prefixes)
        self.check_coverage()
        self.train_items.sort(key=lambda i: i.estimated_duration, reverse=True)
        self.process_items(self.valid_items, prefix="valid", augmentation=False, multiprocessing=False)
        self.process_items(self.train_items, prefix="train", augmentation=True, multiprocessing=True)

    @dask.delayed
    def load_waveform(self, wav_fn: pathlib.Path):
        waveform, _ = librosa.load(wav_fn, sr=self.config.features.audio_sample_rate, mono=True)
        return waveform

    @dask.delayed(nout=2)
    @torch.no_grad()
    def get_mel(self, waveform: numpy.ndarray, shift: float = 0., speed: float = 1.):
        if self.mel_spec is None:
            self.mel_spec = StretchableMelSpectrogram(
                sample_rate=self.config.features.audio_sample_rate,
                n_mels=self.config.features.spectrogram.num_bins,
                n_fft=self.config.features.fft_size,
                win_length=self.config.features.win_size,
                hop_length=self.config.features.hop_size,
                fmin=self.config.features.spectrogram.fmin,
                fmax=self.config.features.spectrogram.fmax
            ).eval().to(self.device)
        mel = self.mel_spec(
            torch.from_numpy(waveform).to(self.device).unsqueeze(0),
            key_shift=shift, speed=speed
        ).squeeze(0).T.cpu().numpy()
        return mel, mel.shape[0]

    @dask.delayed
    def get_mel2ph(self, ph_dur: numpy.ndarray, length: int):
        mel2ph = dur_to_mel2ph(
            self.lr, torch.from_numpy(ph_dur).to(self.device), length, self.timestep
        ).cpu().numpy()
        return mel2ph

    @dask.delayed
    def sec_dur_to_frame_dur(self, dur_sec: numpy.ndarray):
        dur_frame = numpy.diff(
            numpy.round(numpy.cumsum(dur_sec, axis=0) / self.timestep + 0.5).astype(numpy.int64),
            axis=0, prepend=numpy.array([0])
        )
        return dur_frame

    @torch.no_grad()
    def smooth_curve(self, curve: numpy.ndarray, smooth_fn_name: str):
        if smooth_fn_name not in self.smooth_fns:
            self.smooth_fns[smooth_fn_name] = SinusoidalSmoothingConv1d(
                round(self.smooth_widths[smooth_fn_name] / self.timestep)
            ).eval().to(self.device)
        return self.smooth_fns[smooth_fn_name](torch.from_numpy(curve)[None].to(self.device))[0].cpu().numpy()

    @dask.delayed(nout=2)
    def get_f0(self, waveform: numpy.ndarray, length: int):
        pe_method = self.config.extractors.pitch_extraction.method
        if pe_method == "parselmouth":
            f0, uv = get_pitch_parselmouth(
                waveform,
                samplerate=self.config.features.audio_sample_rate,
                length=length,
                hop_size=self.config.features.hop_size,
                f0_min=self.config.extractors.pitch_extraction.f0_min,
                f0_max=self.config.extractors.pitch_extraction.f0_max,
                speed=1,
                interp_uv=True
            )
        elif pe_method == "harvest":
            f0, uv = get_pitch_harvest(
                waveform,
                samplerate=self.config.features.audio_sample_rate,
                length=length,
                hop_size=self.config.features.hop_size,
                f0_min=self.config.extractors.pitch_extraction.f0_min,
                f0_max=self.config.extractors.pitch_extraction.f0_max,
                speed=1,
                interp_uv=True
            )
        elif pe_method == "rmvpe":
            if self.rmvpe is None:
                from modules.rmvpe.inference import RMVPE
                self.rmvpe = RMVPE(self.config.extractors.pitch_extraction.model_path)
            f0, uv = self.rmvpe.get_pitch(
                waveform,
                samplerate=self.config.features.audio_sample_rate,
                length=length,
                hop_size=self.config.features.hop_size,
                interp_uv=True
            )
        else:
            raise ValueError(f"Unknown pitch extraction method: {pe_method}")
        return f0, uv

    @dask.delayed
    def get_energy(self, waveform: numpy.ndarray, length: int, smooth_fn_name: str = None):
        energy = get_energy_librosa(
            waveform, length,
            hop_size=self.config.features.hop_size,
            win_size=self.config.features.win_size
        )
        if smooth_fn_name is not None:
            energy = self.smooth_curve(energy, smooth_fn_name=smooth_fn_name)
        return energy

    @dask.delayed(nout=2)
    def world_analyze(self, waveform: numpy.ndarray, f0: numpy.ndarray):
        sp, ap = world_analyze(
            waveform, f0,
            samplerate=self.config.features.audio_sample_rate,
            hop_size=self.config.features.hop_size,
            fft_size=self.config.features.fft_size
        )
        return sp, ap

    @dask.delayed
    def world_synthesize_aperiodic(self, f0: numpy.ndarray, sp: numpy.ndarray, ap: numpy.ndarray):
        noise = world_synthesize_aperiodic(
            f0, sp, ap,
            samplerate=self.config.features.audio_sample_rate,
            time_step=self.timestep
        )
        return noise

    @dask.delayed
    def world_synthesize_harmonics(self, f0: numpy.ndarray, sp: numpy.ndarray, ap: numpy.ndarray):
        harmonic = world_synthesize_harmonics(
            f0, sp, ap,
            samplerate=self.config.features.audio_sample_rate,
            time_step=self.timestep
        )
        return harmonic

    @dask.delayed(nout=2)
    @torch.no_grad()
    def run_vr_separation(self, waveform: numpy.ndarray):
        if self.hn_sep_model is None:
            from modules.vr import load_sep_model
            self.hn_sep_model = load_sep_model(
                model_path=self.config.extractors.harmonic_noise_separation.model_path,
                device=self.device
            )
        x = torch.from_numpy(waveform).to(self.device).reshape(1, 1, -1)
        if not self.hn_sep_model.is_mono:
            x = x.repeat(1, 2, 1)
        x = self.hn_sep_model.predict_from_audio(x)
        x = torch.mean(x, dim=1)
        harmonic = x.squeeze().cpu().numpy()
        noise = waveform - harmonic
        return harmonic, noise

    def harmonic_noise_separation(self, waveform: numpy.ndarray, f0: numpy.ndarray):
        hn_sep_method = self.config.extractors.harmonic_noise_separation.method
        if hn_sep_method == "world":
            sp, ap = self.world_analyze(waveform, f0)
            noise = self.world_synthesize_aperiodic(f0, sp, ap)
            harmonic = self.world_synthesize_harmonics(f0, sp, ap)
        elif hn_sep_method == "vr":
            harmonic, noise = self.run_vr_separation(waveform)
        else:
            raise ValueError(f"Unknown harmonic-noise separation method: {hn_sep_method}")
        return harmonic, noise

    @dask.delayed
    def get_kth_harmonic(self, harmonic: numpy.ndarray, f0: numpy.ndarray, k: int):
        kth_harmonic = get_kth_harmonic(
            harmonic, f0, k=k,
            samplerate=self.config.features.audio_sample_rate,
            hop_size=self.config.features.hop_size,
            win_size=self.config.features.win_size,
            device=self.device
        )
        return kth_harmonic

    @dask.delayed
    def get_tension(self, harmonic: numpy.ndarray, base_harmonic: numpy.ndarray, length: numpy.ndarray):
        tension = get_tension(
            harmonic, base_harmonic, length,
            hop_size=self.config.features.hop_size,
            win_size=self.config.features.win_size
        )
        tension = self.smooth_curve(tension, smooth_fn_name="tension")
        return tension

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
    def resample_align_curve(
            self, curve: numpy.ndarray,
            original_timestep: float, target_timestep: float, align_length: int
    ):
        resample_aligned_curve = resample_align_curve(
            curve, original_timestep, target_timestep, align_length
        )
        return resample_aligned_curve
