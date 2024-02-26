import torch

from modules.nsf_hifigan.env import AttrDict
from modules.nsf_hifigan.models import Generator


# noinspection SpellCheckingInspection
class NSFHiFiGANONNX(torch.nn.Module):
    def __init__(self, attrs: dict, mel_base='e'):
        super().__init__()
        self.mel_base = str(mel_base)
        assert self.mel_base in ['e', '10'], "mel_base must be 'e', '10' or 10."
        self.generator = Generator(AttrDict(attrs))

    def forward(self, mel: torch.Tensor, f0: torch.Tensor):
        mel = mel.transpose(1, 2)
        if self.mel_base != 'e':
            # log10 to log mel
            mel = mel * 2.30259
        wav = self.generator(mel, f0)
        return wav.squeeze(1)
