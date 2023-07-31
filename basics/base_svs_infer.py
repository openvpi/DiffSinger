# coding=utf8
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Dict

from utils.hparams import hparams
from utils.infer_utils import resample_align_curve


class BaseSVSInfer:
    """
        Base class for SVS inference models.
        Subclasses should define:
        1. *build_model*:
            how to build the model;
        2. *run_model*:
            how to run the model (typically, generate a mel-spectrogram and
            pass it to the pre-built vocoder);
        3. *preprocess_input*:
            how to preprocess user input.
        4. *infer_once*
            infer from raw inputs to the final outputs
    """

    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.timestep = hparams['hop_size'] / hparams['audio_sample_rate']
        self.spk_map = {}
        self.model: torch.nn.Module = None

    def build_model(self, ckpt_steps=None) -> torch.nn.Module:
        raise NotImplementedError()

    def load_speaker_mix(self, param_src: dict, summary_dst: dict,
                         mix_mode: str = 'frame', mix_length: int = None) -> Tuple[Tensor, Tensor]:
        """

        :param param_src: param dict
        :param summary_dst: summary dict
        :param mix_mode: 'token' or 'frame'
        :param mix_length: total tokens or frames to mix
        :return: spk_mix_id [B=1, 1, N], spk_mix_value [B=1, T, N]
        """
        assert mix_mode == 'token' or mix_mode == 'frame'
        param_key = 'spk_mix' if mix_mode == 'frame' else 'ph_spk_mix'
        summary_solo_key = 'spk' if mix_mode == 'frame' else 'ph_spk'
        spk_mix_map = param_src.get(param_key)  # { spk_name: value } or { spk_name: "value value value ..." }
        dynamic = False
        if spk_mix_map is None:
            # Get the first speaker
            for name in self.spk_map.keys():
                spk_mix_map = {name: 1.0}
                break
        else:
            for name in spk_mix_map:
                assert name in self.spk_map, f'Speaker \'{name}\' not found.'
        if len(spk_mix_map) == 1:
            summary_dst[summary_solo_key] = list(spk_mix_map.keys())[0]
        elif any([isinstance(val, str) for val in spk_mix_map.values()]):
            print_mix = '|'.join(spk_mix_map.keys())
            summary_dst[param_key] = f'dynamic({print_mix})'
            dynamic = True
        else:
            print_mix = '|'.join([f'{n}:{"%.3f" % spk_mix_map[n]}' for n in spk_mix_map])
            summary_dst[param_key] = f'static({print_mix})'
        spk_mix_id_list = []
        spk_mix_value_list = []
        if dynamic:
            for name, values in spk_mix_map.items():
                spk_mix_id_list.append(self.spk_map[name])
                if isinstance(values, str):
                    # this speaker has a variable proportion
                    if mix_mode == 'token':
                        cur_spk_mix_value = values.split()
                        assert len(cur_spk_mix_value) == mix_length, \
                            'Speaker mix checks failed. In dynamic token-level mix, ' \
                            'number of proportion values must equal number of tokens.'
                        cur_spk_mix_value = torch.from_numpy(
                            np.array(cur_spk_mix_value, 'float32')
                        ).to(self.device)[None]  # => [B=1, T]
                    else:
                        cur_spk_mix_value = torch.from_numpy(resample_align_curve(
                            np.array(values.split(), 'float32'),
                            original_timestep=float(param_src['spk_mix_timestep']),
                            target_timestep=self.timestep,
                            align_length=mix_length
                        )).to(self.device)[None]  # => [B=1, T]
                    assert torch.all(cur_spk_mix_value >= 0.), \
                        f'Speaker mix checks failed.\n' \
                        f'Proportions of speaker \'{name}\' on some {mix_mode}s are negative.'
                else:
                    # this speaker has a constant proportion
                    assert values >= 0., f'Speaker mix checks failed.\n' \
                                         f'Proportion of speaker \'{name}\' is negative.'
                    cur_spk_mix_value = torch.full(
                        (1, mix_length), fill_value=values,
                        dtype=torch.float32, device=self.device
                    )
                spk_mix_value_list.append(cur_spk_mix_value)
            spk_mix_id = torch.LongTensor(spk_mix_id_list).to(self.device)[None, None]  # => [B=1, 1, N]
            spk_mix_value = torch.stack(spk_mix_value_list, dim=2)  # [B=1, T] => [B=1, T, N]
            spk_mix_value_sum = torch.sum(spk_mix_value, dim=2, keepdim=True)  # => [B=1, T, 1]
            assert torch.all(spk_mix_value_sum > 0.), \
                f'Speaker mix checks failed.\n' \
                f'Proportions of speaker mix on some frames sum to zero.'
            spk_mix_value /= spk_mix_value_sum  # normalize
        else:
            for name, value in spk_mix_map.items():
                spk_mix_id_list.append(self.spk_map[name])
                assert value >= 0., f'Speaker mix checks failed.\n' \
                                    f'Proportion of speaker \'{name}\' is negative.'
                spk_mix_value_list.append(value)
            spk_mix_id = torch.LongTensor(spk_mix_id_list).to(self.device)[None, None]  # => [B=1, 1, N]
            spk_mix_value = torch.FloatTensor(spk_mix_value_list).to(self.device)[None, None]  # => [B=1, 1, N]
            spk_mix_value_sum = spk_mix_value.sum()
            assert spk_mix_value_sum > 0., f'Speaker mix checks failed.\n' \
                                           f'Proportions of speaker mix sum to zero.'
            spk_mix_value /= spk_mix_value_sum  # normalize
        return spk_mix_id, spk_mix_value

    def preprocess_input(self, param: dict, idx=0) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def forward_model(self, sample: Dict[str, torch.Tensor]):
        raise NotImplementedError()

    def run_inference(self, params, **kwargs):
        raise NotImplementedError()
