from pathlib import Path
import csv
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from utils.binarizer_utils import get_pitch_parselmouth


class PitchCorrector:
    def __init__(self, hparams=None):
        if hparams is None:
            hparams = {"hop_size": 512, "audio_sample_rate": 44100}
        self.hparams = hparams

    def calculate_note_mean_pitch(self, note_seq, note_dur, f0):
        hop_size = self.hparams["hop_size"]
        sample_rate = self.hparams["audio_sample_rate"]
        start_time = 0
        corrected_pitches = []
        pitch_deviations = []

        for pitch, duration in zip(note_seq, note_dur):
            if pitch == "rest":
                corrected_pitches.append(0)
                pitch_deviations.append(0)
                start_time += duration
                continue

            start_frame = int(start_time * sample_rate / hop_size)
            end_frame = int((start_time + duration) * sample_rate / hop_size)

            note_f0 = f0[start_frame:end_frame]
            pitch_midi = librosa.note_to_midi(pitch)
            note_midi = librosa.hz_to_midi(note_f0)
            filtered_f0 = note_midi[
                (note_midi >= pitch_midi - 0.5) & (note_midi < pitch_midi + 0.5)
            ]

            mean_pitch = np.mean(filtered_f0)

            # 如果平均值为零或无效，使用标准频率作为基准值
            # if mean_pitch == 0 or np.isnan(mean_pitch):
            #     mean_pitch = librosa.note_to_hz(pitch)

            corrected_pitches.append(mean_pitch)

            deviation = mean_pitch - pitch_midi
            pitch_deviations.append(deviation * 100)

            start_time += duration

        return corrected_pitches, pitch_deviations

    def correct_pitch(self, name, note_seq, note_dur, ph_dur):
        seconds = sum(float(x) for x in ph_dur.split())
        timestep = self.hparams["hop_size"] / self.hparams["audio_sample_rate"]
        length = round(seconds / timestep)
        wav_data, _ = librosa.load(
            wavs_dir.joinpath(f"{name}.wav"), sr=self.hparams["audio_sample_rate"]
        )
        f0, uv = get_pitch_parselmouth(wav_data, length, self.hparams, interp_uv=True)
        corrected_pitches, pitch_deviations = self.calculate_note_mean_pitch(
            note_seq, note_dur, f0
        )
        return corrected_pitches, pitch_deviations, f0


class NotePlotter:
    def __init__(self, name, note_seq, show=False, save=False):
        valid_notes = [n for n in note_seq if n != "rest"]
        min_note = min(valid_notes, key=librosa.note_to_midi)
        max_note = max(valid_notes, key=librosa.note_to_midi)

        self.bottom_key = librosa.note_to_midi(min_note) - 6
        self.top_key = librosa.note_to_midi(max_note) + 6
        self.name = name
        self.show = show
        self.save = save

    def plot_notes(self, note_seq, note_dur, f0, corrected_pitches, pitch_deviations):
        start_time = 0
        total_duration = sum(note_dur)
        fig, ax = plt.subplots(figsize=(total_duration * 4, 6))

        # 绘制原始音高曲线
        plt.plot(
            np.arange(len(f0)) * hparams["hop_size"] / hparams["audio_sample_rate"],
            librosa.hz_to_midi(f0),
            linewidth=1,
            linestyle="--",
            color="blue",
            label="Original Pitch",
        )

        show_corrected = True
        # 绘制音符块和计算后的音高曲线
        for pitch, duration, deviation in zip(note_seq, note_dur, pitch_deviations):
            if pitch == "rest":
                start_time += duration
                continue

            midi_pitch = librosa.note_to_midi(pitch)
            corrected_midi_pitch = midi_pitch + deviation / 100

            ax.add_patch(
                patches.Rectangle(
                    (start_time, midi_pitch - 0.5),
                    duration,
                    1,
                    facecolor="cornflowerblue",
                    edgecolor="black",
                )
            )

            text_x = start_time + 0.01
            text_y = midi_pitch + 0.55
            ax.text(text_x, text_y, f"{deviation:.2f}¢\n{pitch}", fontsize=6, color="black")

            plt.plot(
                [start_time, start_time + duration],
                [corrected_midi_pitch, corrected_midi_pitch],
                color="firebrick",
                linewidth=2,
                label="Corrected Pitch" if show_corrected else "",
            )

            show_corrected = False
            start_time += duration

        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        ax.set_ylim(self.bottom_key - 0.5, self.top_key + 0.5)
        ax.set_yticks(np.arange(self.bottom_key, self.top_key + 1, step=2))
        ax.set_yticklabels(
            librosa.midi_to_note(np.arange(self.bottom_key, self.top_key + 1, step=2)),
            fontweight="bold",
        )
        ax.grid(True)
        plt.legend()
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        if self.show:
            plt.show()
        if self.save:
            out = output_dir / f"{self.name}.jpg"
            plt.savefig(out, format="jpg")
        plt.close()
        return


output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
output_csv = output_dir / "transcriptions-pitch-correction.csv"
hparams = {"hop_size": 512, "audio_sample_rate": 44100}
wavs_dir = Path("wavs")
transcriptions = Path("transcriptions-ph2word.csv")
reader = csv.reader(transcriptions.read_text().splitlines())
header = next(reader)
reader = list(reader)
data_list = []
pitch_corrector = PitchCorrector(hparams)
need_print = False
show_plot = False
save_plot = False

custom_columns = [
    TextColumn("[bold blue]Processing: {task.fields[processing_name]}"),
    TextColumn("Elapsed:"),
    TimeElapsedColumn(),
    BarColumn(bar_width=None, finished_style="green"),
    TextColumn("Processed: {task.completed}/{task.total}"),
    TextColumn("ETA:"),
    TimeRemainingColumn(),
]

with Progress(*custom_columns) as progress:
    task = progress.add_task("Processing...", total=len(reader), processing_name="N/A")
    for row in reader:
        name, ph_seq, ph_dur, ph_num, note_seq, note_dur = row
        progress.update(task, processing_name=name)
        progress.update(task, advance=1)
        # 将音符序列和音符持续时间转换为列表
        note_seq_list = note_seq.split()
        note_dur_list = [float(x) for x in note_dur.split()]

        corrected_pitches, pitch_deviations, f0 = pitch_corrector.correct_pitch(
            name, note_seq_list, note_dur_list, ph_dur
        )

        for idx, (pitch, corrected_pitch, deviation) in enumerate(
            zip(note_seq_list, corrected_pitches, pitch_deviations)
        ):
            if need_print:
                if pitch == "rest":
                    progress.print(
                        f"{name} - Note: rest, Original Pitch: N/A, Corrected Pitch: N/A, Deviation (cents): N/A"
                    )
                else:
                    progress.print(
                        f"{name} - "
                        f"Note {idx + 1}: {pitch}, "
                        f"Original Pitch (Hz): {librosa.note_to_hz(pitch):.2f}, "
                        f"Corrected Pitch (Hz): {corrected_pitch:.2f}, "
                        f"Deviation (cents): {deviation:.2f}"
                    )

        if show_plot or save_plot:
            note_plotter = NotePlotter(name, note_seq_list, show=show_plot, save=save_plot)
            note_plotter.plot_notes(
                note_seq_list, note_dur_list, f0, corrected_pitches, pitch_deviations
            )

        new_note_seq_list = []
        for pitch, deviation in zip(note_seq_list, pitch_deviations):
            if pitch == "rest":
                new_note_seq_list.append("rest")
            elif deviation and not np.isnan(deviation):
                dev = round(deviation)
                new_note_seq_list.append(f"{pitch}{'+' if dev >= 0 else ''}{dev}")
            else:
                new_note_seq_list.append(pitch)

        data_list.append(
            [
                name,
                ph_seq,
                ph_dur,
                ph_num,
                " ".join(new_note_seq_list),
                note_dur,
            ]
        )

writer = csv.writer(output_csv.open("w", newline=""))
writer.writerow(header)
for data in data_list:
    writer.writerow(data)
