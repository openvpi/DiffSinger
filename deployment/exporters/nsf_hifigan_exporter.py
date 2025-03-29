import json
from pathlib import Path
from typing import Union

import onnx
import onnxsim
import torch
import yaml
from torch import nn

from basics.base_exporter import BaseExporter
from deployment.modules.nsf_hifigan import NSFHiFiGANONNX
from utils import load_ckpt, remove_suffix
from utils.hparams import hparams


class NSFHiFiGANExporter(BaseExporter):
    def __init__(
            self,
            device: Union[str, torch.device] = 'cpu',
            cache_dir: Path = None,
            model_path: Path = None,
            model_name: str = 'nsf_hifigan'
    ):
        super().__init__(device=device, cache_dir=cache_dir)
        self.model_path = model_path
        self.model_name = model_name
        self.vocoder_pitch_controllable = False
        self.model = self.build_model()
        self.model_class_name = remove_suffix(self.model.__class__.__name__, 'ONNX')
        self.model_cache_path = (self.cache_dir / self.model_name).with_suffix('.onnx')

    def build_model(self) -> nn.Module:
        config_path = self.model_path.with_name('config.json')
        with open(config_path, 'r', encoding='utf8') as f:
            config = json.load(f)
        assert hparams.get('mel_base') == 'e', (
            "Mel base must be set to \'e\' according to 2nd stage of the migration plan. "
            "See https://github.com/openvpi/DiffSinger/releases/tag/v2.3.0 for more details."
        )
        model = NSFHiFiGANONNX(config).eval().to(self.device)
        self.vocoder_pitch_controllable = config.get("pc_aug", False)
        load_ckpt(model.generator, str(self.model_path),
                  prefix_in_ckpt=None, key_in_ckpt='generator',
                  strict=True, device=self.device)
        model.generator.remove_weight_norm()
        return model

    def export(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.export_model(path / self.model_cache_path.name)
        self.export_attachments(path)

    def export_model(self, path: Path):
        self._torch_export_model()
        model_onnx = self._optimize_model_graph(onnx.load(self.model_cache_path))
        onnx.save(model_onnx, path)
        self.model_cache_path.unlink()
        print(f'| export model => {path}')

    def export_attachments(self, path: Path):
        config_path = path / 'vocoder.yaml'
        with open(config_path, 'w', encoding='utf8') as fw:
            yaml.safe_dump({
                # basic configs
                'name': self.model_name,
                'model': self.model_cache_path.name,
                # mel specifications
                'sample_rate': hparams['audio_sample_rate'],
                'hop_size': hparams['hop_size'],
                'win_size': hparams['win_size'],
                'fft_size': hparams['fft_size'],
                'num_mel_bins': hparams['audio_num_mel_bins'],
                'mel_fmin': hparams['fmin'],
                'mel_fmax': hparams['fmax'] if hparams['fmax'] is not None else hparams['audio_sample_rate'] / 2,
                'mel_base': 'e',
                'mel_scale': 'slaney',
                'pitch_controllable': self.vocoder_pitch_controllable,
                # Some old vocoder versions may have severe performance issues on CUDA;
                # the issues were fixed in newer versions, and this flag is to distinguish them
                'force_on_cpu': False,
            }, fw, sort_keys=False)
        print(f'| export configs => {config_path} **PLEASE EDIT BEFORE USE**')

    @torch.no_grad()
    def _torch_export_model(self):
        # Prepare inputs for NSFHiFiGAN
        n_frames = 10
        mel = torch.randn((1, n_frames, hparams['audio_num_mel_bins']), dtype=torch.float32, device=self.device)
        f0 = torch.randn((1, n_frames), dtype=torch.float32, device=self.device) + 440.

        # PyTorch ONNX export for NSFHiFiGAN
        print(f'Exporting {self.model_class_name}...')
        torch.onnx.export(
            self.model,
            (
                mel,
                f0
            ),
            self.model_cache_path,
            input_names=[
                'mel',
                'f0'
            ],
            output_names=[
                'waveform'
            ],
            dynamic_axes={
                'mel': {
                    1: 'n_frames'
                },
                'f0': {
                    1: 'n_frames'
                },
                'waveform': {
                    1: 'n_samples'
                }
            },
            opset_version=15
        )

    def _optimize_model_graph(self, model: onnx.ModelProto) -> onnx.ModelProto:
        print(f'Running ONNX simplifier for {self.model_class_name}...')
        model, check = onnxsim.simplify(model, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        return model
