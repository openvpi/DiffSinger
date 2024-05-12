import torch
import numpy as np
import yaml
import os
from modules.hnsep.pw import DecomposedWaveformPyWorld
from .nets import CascadedNet

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__


def load_sep_model(model_path, device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    model = CascadedNet(
                args.n_fft, 
                args.hop_length, 
                args.n_out, 
                args.n_out_lstm, 
                True, 
                is_mono=args.is_mono)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
    
  
class DecomposedWaveformVocalRemover(DecomposedWaveformPyWorld):
    def __init__(
            self, waveform, samplerate, f0,  # basic parameters
            model_path,
            hop_size=None, fft_size=None, win_size=None, base_harmonic_radius=3.5,  # analysis parameters
            device=None  # computation parameters
    ):
        super().__init__(waveform, samplerate, f0, hop_size=hop_size, fft_size=fft_size, 
                            win_size=win_size, base_harmonic_radius=base_harmonic_radius, device=device)
        self.sep_model = load_sep_model(model_path, self._device)
    
    def _infer(self):
        with torch.no_grad():
            x = torch.from_numpy(self._waveform).to(self._device).reshape(1, 1, -1)
            if not self.sep_model.is_mono:
                x = x.repeat(1, 2, 1)
            x = self.sep_model.predict_fromaudio(x)
            x = torch.mean(x, dim=1)
            self._harmonic_part = x.squeeze().cpu().numpy()
            self._aperiodic_part = self._waveform - self._harmonic_part
        
    def harmonic(self, k: int = None) -> np.ndarray:
        """
        Extract the full harmonic part, or the Kth harmonic if `k` is not None, from the waveform.
        :param k: an integer representing the harmonic index, starting from 0
        :return: full_harmonics float32[T] or kth_harmonic float32[T]
        """
        if k is not None:
            return self._kth_harmonic(k)
        if self._harmonic_part is not None:
            return self._harmonic_part
        self._infer()
        return self._harmonic_part

    def aperiodic(self) -> np.ndarray:
        """
        Extract the aperiodic part from the waveform.
        :return: aperiodic_part float32[T]
        """
        if self._aperiodic_part is not None:
            return self._aperiodic_part
        self._infer()
        return self._aperiodic_part        