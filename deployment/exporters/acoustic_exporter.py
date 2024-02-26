import shutil
from pathlib import Path
from typing import List, Union, Tuple, Dict

import onnx
import onnxsim
import torch
import yaml

from basics.base_exporter import BaseExporter
from deployment.modules.toplevel import DiffSingerAcousticONNX
from modules.fastspeech.param_adaptor import VARIANCE_CHECKLIST
from utils import load_ckpt, onnx_helper, remove_suffix
from utils.hparams import hparams
from utils.phoneme_utils import locate_dictionary, build_phoneme_list
from utils.text_encoder import TokenTextEncoder


class DiffSingerAcousticExporter(BaseExporter):
    def __init__(
            self,
            device: Union[str, torch.device] = 'cpu',
            cache_dir: Path = None,
            ckpt_steps: int = None,
            freeze_gender: float = None,
            freeze_velocity: bool = False,
            export_spk: List[Tuple[str, Dict[str, float]]] = None,
            freeze_spk: Tuple[str, Dict[str, float]] = None
    ):
        super().__init__(device=device, cache_dir=cache_dir)
        # Basic attributes
        self.model_name: str = hparams['exp_name']
        self.ckpt_steps: int = ckpt_steps
        self.spk_map: dict = self.build_spk_map()
        self.vocab = TokenTextEncoder(vocab_list=build_phoneme_list())
        self.model = self.build_model()
        self.fs2_aux_cache_path = self.cache_dir / (
            'fs2_aux.onnx' if self.model.use_shallow_diffusion else 'fs2.onnx'
        )
        self.diffusion_cache_path = self.cache_dir / 'diffusion.onnx'

        # Attributes for logging
        self.model_class_name = remove_suffix(self.model.__class__.__name__, 'ONNX')
        fs2_aux_cls_logging = [remove_suffix(self.model.fs2.__class__.__name__, 'ONNX')]
        if self.model.use_shallow_diffusion:
            fs2_aux_cls_logging.append(remove_suffix(
                self.model.aux_decoder.decoder.__class__.__name__, 'ONNX'
            ))
        self.fs2_aux_class_name = ', '.join(fs2_aux_cls_logging)
        self.aux_decoder_class_name = remove_suffix(
            self.model.aux_decoder.decoder.__class__.__name__, 'ONNX'
        ) if self.model.use_shallow_diffusion else None
        self.denoiser_class_name = remove_suffix(self.model.diffusion.denoise_fn.__class__.__name__, 'ONNX')
        self.diffusion_class_name = remove_suffix(self.model.diffusion.__class__.__name__, 'ONNX')

        # Attributes for exporting
        self.expose_gender = freeze_gender is None
        self.expose_velocity = not freeze_velocity
        self.freeze_spk: Tuple[str, Dict[str, float]] = freeze_spk \
            if hparams['use_spk_id'] else None
        self.export_spk: List[Tuple[str, Dict[str, float]]] = export_spk \
            if hparams['use_spk_id'] and export_spk is not None else []
        if hparams['use_key_shift_embed'] and not self.expose_gender:
            shift_min, shift_max = hparams['augmentation_args']['random_pitch_shifting']['range']
            key_shift = freeze_gender * shift_max if freeze_gender >= 0. else freeze_gender * abs(shift_min)
            key_shift = max(min(key_shift, shift_max), shift_min)  # clip key shift
            self.model.fs2.register_buffer('frozen_key_shift', torch.FloatTensor([key_shift]).to(self.device))
        if hparams['use_spk_id']:
            if not self.export_spk and self.freeze_spk is None:
                # In case the user did not specify any speaker settings:
                if len(self.spk_map) == 1:
                    # If there is only one speaker, freeze him/her.
                    first_spk = next(iter(self.spk_map.keys()))
                    self.freeze_spk = (first_spk, {first_spk: 1.0})
                else:
                    # If there are multiple speakers, export them all.
                    self.export_spk = [(name, {name: 1.0}) for name in self.spk_map.keys()]
            if self.freeze_spk is not None:
                self.model.fs2.register_buffer('frozen_spk_embed', self._perform_spk_mix(self.freeze_spk[1]))

    def build_model(self) -> DiffSingerAcousticONNX:
        model = DiffSingerAcousticONNX(
            vocab_size=len(self.vocab),
            out_dims=hparams['audio_num_mel_bins']
        ).eval().to(self.device)
        load_ckpt(model, hparams['work_dir'], ckpt_steps=self.ckpt_steps,
                  prefix_in_ckpt='model', strict=True, device=self.device)
        return model

    def export(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        model_name = self.model_name
        if self.freeze_spk is not None:
            model_name += '.' + self.freeze_spk[0]
        self.export_model(path / f'{model_name}.onnx')
        self.export_attachments(path)

    def export_model(self, path: Path):
        self._torch_export_model()
        fs2_aux_onnx = self._optimize_fs2_aux_graph(onnx.load(self.fs2_aux_cache_path))
        diffusion_onnx = self._optimize_diffusion_graph(onnx.load(self.diffusion_cache_path))
        model_onnx = self._merge_fs2_aux_diffusion_graphs(fs2_aux_onnx, diffusion_onnx)
        onnx.save(model_onnx, path)
        self.fs2_aux_cache_path.unlink()
        self.diffusion_cache_path.unlink()
        print(f'| export model => {path}')

    def export_attachments(self, path: Path):
        for spk in self.export_spk:
            self._export_spk_embed(
                path / f'{self.model_name}.{spk[0]}.emb',
                self._perform_spk_mix(spk[1])
            )
        self._export_dictionary(path / 'dictionary.txt')
        self._export_phonemes(path / f'{self.model_name}.phonemes.txt')

        model_name = self.model_name
        if self.freeze_spk is not None:
            model_name += '.' + self.freeze_spk[0]
        dsconfig = {
            # basic configs
            'phonemes': f'{self.model_name}.phonemes.txt',
            'acoustic': f'{model_name}.onnx',
            'vocoder': 'nsf_hifigan',
        }
        # multi-speaker
        if len(self.export_spk) > 0:
            dsconfig['speakers'] = [f'{self.model_name}.{spk[0]}' for spk in self.export_spk]
        # parameters
        if self.expose_gender:
            dsconfig['augmentation_args'] = {
                'random_pitch_shifting': {
                    'range': hparams['augmentation_args']['random_pitch_shifting']['range']
                }
            }
        dsconfig['use_key_shift_embed'] = self.expose_gender
        dsconfig['use_speed_embed'] = self.expose_velocity
        for variance in VARIANCE_CHECKLIST:
            dsconfig[f'use_{variance}_embed'] = variance in self.model.fs2.variance_embed_list
        # shallow diffusion
        dsconfig['use_shallow_diffusion'] = self.model.use_shallow_diffusion
        dsconfig['max_depth'] = self.model.diffusion.k_step
        # mel specification
        dsconfig['sample_rate'] = hparams['audio_sample_rate']
        dsconfig['hop_size'] = hparams['hop_size']
        dsconfig['num_mel_bins'] = hparams['audio_num_mel_bins']
        dsconfig['mel_fmin'] = hparams['fmin']
        dsconfig['mel_fmax'] = hparams['fmax'] if hparams['fmax'] is not None else hparams['audio_sample_rate'] / 2
        dsconfig['mel_base'] = str(hparams.get('mel_base', '10'))
        dsconfig['mel_scale'] = 'slaney'
        config_path = path / 'dsconfig.yaml'
        with open(config_path, 'w', encoding='utf8') as fw:
            yaml.safe_dump(dsconfig, fw, sort_keys=False)
        print(f'| export configs => {config_path} **PLEASE EDIT BEFORE USE**')

    @torch.no_grad()
    def _torch_export_model(self):
        # Prepare inputs for FastSpeech2 and aux decoder tracing
        n_frames = 10
        tokens = torch.LongTensor([[1]]).to(self.device)
        durations = torch.LongTensor([[n_frames]]).to(self.device)
        f0 = torch.FloatTensor([[440.] * n_frames]).to(self.device)
        variances = {
            v_name: torch.zeros(1, n_frames, dtype=torch.float32, device=self.device)
            for v_name in self.model.fs2.variance_embed_list
        }
        kwargs: Dict[str, torch.Tensor] = {}
        arguments = (tokens, durations, f0, variances, kwargs)
        input_names = ['tokens', 'durations', 'f0'] + self.model.fs2.variance_embed_list
        dynamix_axes = {
            'tokens': {
                1: 'n_tokens'
            },
            'durations': {
                1: 'n_tokens'
            },
            'f0': {
                1: 'n_frames'
            },
            **{
                v_name: {
                    1: 'n_frames'
                }
                for v_name in self.model.fs2.variance_embed_list
            }
        }
        if hparams['use_key_shift_embed']:
            if self.expose_gender:
                kwargs['gender'] = torch.rand((1, n_frames), dtype=torch.float32, device=self.device)
                input_names.append('gender')
                dynamix_axes['gender'] = {
                    1: 'n_frames'
                }
        if hparams['use_speed_embed']:
            if self.expose_velocity:
                kwargs['velocity'] = torch.rand((1, n_frames), dtype=torch.float32, device=self.device)
                input_names.append('velocity')
                dynamix_axes['velocity'] = {
                    1: 'n_frames'
                }
        if hparams['use_spk_id'] and not self.freeze_spk:
            kwargs['spk_embed'] = torch.rand(
                (1, n_frames, hparams['hidden_size']),
                dtype=torch.float32, device=self.device
            )
            input_names.append('spk_embed')
            dynamix_axes['spk_embed'] = {
                1: 'n_frames'
            }
        dynamix_axes['condition'] = {
            1: 'n_frames'
        }

        # PyTorch ONNX export for FastSpeech2 and aux decoder
        output_names = ['condition']
        if self.model.use_shallow_diffusion:
            output_names.append('aux_mel')
            dynamix_axes['aux_mel'] = {
                1: 'n_frames'
            }
        print(f'Exporting {self.fs2_aux_class_name}...')
        torch.onnx.export(
            self.model.view_as_fs2_aux(),
            arguments,
            self.fs2_aux_cache_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamix_axes,
            opset_version=15
        )

        condition = torch.rand((1, n_frames, hparams['hidden_size']), device=self.device)

        # Prepare inputs for denoiser tracing and GaussianDiffusion scripting
        shape = (1, 1, hparams['audio_num_mel_bins'], n_frames)
        noise = torch.randn(shape, device=self.device)
        x_start = torch.randn((1, n_frames, hparams['audio_num_mel_bins']),device=self.device)
        step = (torch.rand((1,), device=self.device) * hparams['K_step']).long()

        print(f'Tracing {self.denoiser_class_name} denoiser...')
        diffusion = self.model.view_as_diffusion()
        diffusion.diffusion.denoise_fn = torch.jit.trace(
            diffusion.diffusion.denoise_fn,
            (
                noise,
                step,
                condition.transpose(1, 2)
            )
        )

        print(f'Scripting {self.diffusion_class_name}...')
        diffusion_inputs = [
            condition,
            *([x_start, 100] if self.model.use_shallow_diffusion else [])
        ]
        diffusion = torch.jit.script(
            diffusion,
            example_inputs=[
                (
                    *diffusion_inputs,
                    1  # p_sample branch
                ),
                (
                    *diffusion_inputs,
                    200  # p_sample_plms branch
                )
            ]
        )

        # PyTorch ONNX export for GaussianDiffusion
        print(f'Exporting {self.diffusion_class_name}...')
        torch.onnx.export(
            diffusion,
            (
                *diffusion_inputs,
                200
            ),
            self.diffusion_cache_path,
            input_names=[
                'condition',
                *(['x_start', 'depth'] if self.model.use_shallow_diffusion else []),
                'speedup'
            ],
            output_names=[
                'mel'
            ],
            dynamic_axes={
                'condition': {
                    1: 'n_frames'
                },
                **({'x_start': {1: 'n_frames'}} if self.model.use_shallow_diffusion else {}),
                'mel': {
                    1: 'n_frames'
                }
            },
            opset_version=15
        )

    @torch.no_grad()
    def _perform_spk_mix(self, spk_mix: Dict[str, float]):
        spk_mix_ids = []
        spk_mix_values = []
        for name, value in spk_mix.items():
            spk_mix_ids.append(self.spk_map[name])
            assert value >= 0., f'Speaker mix checks failed.\n' \
                                f'Proportion of speaker \'{name}\' is negative.'
            spk_mix_values.append(value)
        spk_mix_id_N = torch.LongTensor(spk_mix_ids).to(self.device)[None]  # => [1, N]
        spk_mix_value_N = torch.FloatTensor(spk_mix_values).to(self.device)[None]  # => [1, N]
        spk_mix_value_sum = spk_mix_value_N.sum()
        assert spk_mix_value_sum > 0., f'Speaker mix checks failed.\n' \
                                       f'Proportions of speaker mix sum to zero.'
        spk_mix_value_N /= spk_mix_value_sum  # normalize
        spk_mix_embed = torch.sum(
            self.model.fs2.spk_embed(spk_mix_id_N) * spk_mix_value_N.unsqueeze(2),  # => [1, N, H]
            dim=1, keepdim=False
        )  # => [1, H]
        return spk_mix_embed

    def _optimize_fs2_aux_graph(self, fs2: onnx.ModelProto) -> onnx.ModelProto:
        print(f'Running ONNX Simplifier on {self.fs2_aux_class_name}...')
        fs2, check = onnxsim.simplify(fs2, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        print(f'| optimize graph: {self.fs2_aux_class_name}')
        return fs2

    def _optimize_diffusion_graph(self, diffusion: onnx.ModelProto) -> onnx.ModelProto:
        onnx_helper.model_override_io_shapes(diffusion, output_shapes={
            'mel': (1, 'n_frames', hparams['audio_num_mel_bins'])
        })
        print(f'Running ONNX Simplifier #1 on {self.diffusion_class_name}...')
        diffusion, check = onnxsim.simplify(diffusion, include_subgraph=True)
        assert check, 'Simplified ONNX model could not be validated'
        onnx_helper.graph_fold_back_to_squeeze(diffusion.graph)
        onnx_helper.graph_extract_conditioner_projections(
            graph=diffusion.graph, op_type='Conv',
            weight_pattern=r'diffusion\.denoise_fn\.residual_layers\.\d+\.conditioner_projection\.weight',
            alias_prefix='/diffusion/denoise_fn/cache'
        )
        onnx_helper.graph_remove_unused_values(diffusion.graph)
        print(f'Running ONNX Simplifier #2 on {self.diffusion_class_name}...')
        diffusion, check = onnxsim.simplify(
            diffusion,
            include_subgraph=True
        )
        assert check, 'Simplified ONNX model could not be validated'
        print(f'| optimize graph: {self.diffusion_class_name}')
        return diffusion

    def _merge_fs2_aux_diffusion_graphs(self, fs2: onnx.ModelProto, diffusion: onnx.ModelProto) -> onnx.ModelProto:
        onnx_helper.model_add_prefixes(
            fs2, dim_prefix=('fs2aux.' if self.model.use_shallow_diffusion else 'fs2.'),
            ignored_pattern=r'(n_tokens)|(n_frames)'
        )
        onnx_helper.model_add_prefixes(diffusion, dim_prefix='diffusion.', ignored_pattern='n_frames')
        print(f'Merging {self.fs2_aux_class_name} and {self.diffusion_class_name} '
              f'back into {self.model_class_name}...')
        merged = onnx.compose.merge_models(
            fs2, diffusion, io_map=[
                ('condition', 'condition'),
                *([('aux_mel', 'x_start')] if self.model.use_shallow_diffusion else []),
            ],
            prefix1='', prefix2='', doc_string='',
            producer_name=fs2.producer_name, producer_version=fs2.producer_version,
            domain=fs2.domain, model_version=fs2.model_version
        )
        merged.graph.name = fs2.graph.name

        print(f'Running ONNX Simplifier on {self.model_class_name}...')
        merged, check = onnxsim.simplify(
            merged,
            include_subgraph=True
        )
        assert check, 'Simplified ONNX model could not be validated'
        print(f'| optimize graph: {self.model_class_name}')

        return merged

    # noinspection PyMethodMayBeStatic
    def _export_spk_embed(self, path: Path, spk_embed: torch.Tensor):
        with open(path, 'wb') as f:
            f.write(spk_embed.cpu().numpy().tobytes())
        print(f'| export spk embed => {path}')

    # noinspection PyMethodMayBeStatic
    def _export_dictionary(self, path: Path):
        print(f'| export dictionary => {path}')
        shutil.copy(locate_dictionary(), path)

    def _export_phonemes(self, path: Path):
        self.vocab.store_to_file(path)
        print(f'| export phonemes => {path}')
