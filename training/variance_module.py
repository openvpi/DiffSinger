import torch
from lightning.pytorch.loggers import TensorBoardLogger

from lib.plot import pitch_note_to_figure, curve_to_figure
from modules.losses import RectifiedFlowLoss
from modules.metrics import RawCurveAccuracy, RawCurveR2Score
from modules.toplevel import DiffSingerVariance
from training.pl_module_base import BaseLightningModule


class VarianceLightningModule(BaseLightningModule):
    def build_model(self) -> DiffSingerVariance:
        return DiffSingerVariance(self.model_config)

    def register_losses_and_metrics(self) -> None:
        if self.model_config.prediction.predict_pitch:
            pitch_loss_type = self.training_config.loss.pitch_predictor.main_loss_type
            if pitch_loss_type not in ["L1", "L2"]:
                raise ValueError(f"Invalid pitch_predictor.main_loss_type: {pitch_loss_type}")
            diff_spec_loss = RectifiedFlowLoss(
                loss_type=pitch_loss_type,
                log_norm=self.training_config.loss.pitch_predictor.main_loss_log_norm
            )
            self.register_loss("diff_pitch_loss", diff_spec_loss)
            self.register_metric("pitch_accuracy", RawCurveAccuracy(tolerance=0.5))
        variance_prediction_list = self.model_config.prediction.predicted_variance_names
        if variance_prediction_list:
            variance_loss_type = self.training_config.loss.variance_predictor.main_loss_type
            if variance_loss_type not in ["L1", "L2"]:
                raise ValueError(f"Invalid variance_predictor.main_loss_type: {variance_loss_type}")
            diff_spec_loss = RectifiedFlowLoss(
                loss_type=variance_loss_type,
                log_norm=self.training_config.loss.variance_predictor.main_loss_log_norm
            )
            self.register_loss("diff_variance_loss", diff_spec_loss)
            for v_name in variance_prediction_list:
                self.register_metric(f"{v_name}_r2", RawCurveR2Score())

    def forward_model(self, sample: dict[str, torch.Tensor], infer: bool) -> dict[str, torch.Tensor]:
        tokens = sample["tokens"]
        languages = sample["languages"]
        durations = sample["ph_dur"]
        spk_id = sample["spk_id"]
        if self.model_config.prediction.predict_pitch:
            base_pitch = sample["base_pitch"]
            note_midi = sample["note_midi"]
            note_rest = sample["note_rest"]
            note_dur = sample["note_dur"]
            note_glide = sample.get("note_glide")
        else:
            base_pitch = note_midi = note_rest = note_dur = note_glide = None
        pitch = sample["pitch"]
        uv = sample["uv"]
        variances = {v_name: sample[v_name] for v_name in self.model_config.prediction.predicted_variance_names}
        pitch_out, variance_out, mask = self.model(
            tokens=tokens, durations=durations, languages=languages, spk_ids=spk_id,
            note_midi=note_midi, note_rest=note_rest, note_dur=note_dur, note_glide=note_glide,
            base_pitch=base_pitch, pitch=pitch, infer=infer, **variances
        )
        if infer:
            outputs = {}
            if pitch_out is not None:
                outputs["pitch"] = pitch_out
                self.metrics["pitch_accuracy"].update(pred=pitch_out, target=pitch, mask=mask & ~uv)
            if variance_out is not None:
                for v_name, v_out in zip(self.model_config.prediction.predicted_variance_names, variance_out):
                    outputs[v_name] = v_out
                    self.metrics[f"{v_name}_r2"].update(pred=v_out, target=sample[v_name], mask=mask)
            return outputs
        else:
            non_padding = mask.unsqueeze(-1).to(pitch)
            losses = {}
            if pitch_out is not None:
                v_pred, v_gt, t = pitch_out
                losses["diff_pitch_loss"] = self.losses["diff_pitch_loss"](v_pred, v_gt, t=t, non_padding=non_padding)
            if variance_out is not None:
                v_pred, v_gt, t = variance_out
                losses["diff_variance_loss"] = self.losses["diff_variance_loss"](
                    v_pred, v_gt, t=t, non_padding=non_padding
                )
            return losses

    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        for i in range(len(sample["indices"])):
            data_idx = sample["indices"][i].item()
            curve_len = self.valid_dataset.info["pitch"][data_idx]
            if data_idx >= self.training_config.validation.max_plots:
                continue
            spk_name = self.valid_dataset.info["spk_names"][data_idx]
            item_name = self.valid_dataset.info["item_names"][data_idx]
            title = f"{spk_name} - {item_name}"
            if (pitch_pred := outputs.get("pitch")) is not None:
                note_len = self.valid_dataset.info["note_midi"][data_idx]
                self.plot_pitch(
                    tag=f"pitch/diff_pitch_{data_idx}",
                    pitch_gt=sample["pitch"][i, :curve_len],
                    pitch_pred=pitch_pred[i, :curve_len],
                    note_midi=sample["note_midi"][i, :note_len],
                    note_dur=sample["note_dur"][i, :note_len],
                    note_rest=sample["note_rest"][i, :note_len],
                    title=title
                )
            for v_name in self.model_config.prediction.predicted_variance_names:
                self.plot_curve(
                    tag=f"{v_name}/diff_{v_name}_{data_idx}",
                    curve_gt=sample[v_name][i, :curve_len],
                    curve_pred=outputs[v_name][i, :curve_len],
                    title=title
                )

    def plot_pitch(
            self, tag: str, pitch_gt: torch.Tensor, pitch_pred: torch.Tensor,
            note_midi: torch.Tensor, note_dur: torch.Tensor, note_rest: torch.Tensor,
            title=None
    ):
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(tag, pitch_note_to_figure(
            pitch_gt, pitch_pred, note_midi, note_dur, note_rest, title
        ), self.global_step)

    def plot_curve(
            self, tag: str, curve_gt: torch.Tensor, curve_pred: torch.Tensor,
            title=None
    ):
        logger: TensorBoardLogger = self.logger
        logger.experiment.add_figure(tag, curve_to_figure(
            curve_gt, curve_pred, None, None, title
        ), self.global_step)
