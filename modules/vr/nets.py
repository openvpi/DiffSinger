import torch
from torch import nn
import torch.nn.functional as F

from . import layers


class BaseNet(nn.Module):

    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super(BaseNet, self).__init__()
        self.enc1 = layers.Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = layers.Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = layers.Encoder(nout * 6, nout * 8, 3, 2, 1)

        self.aspp = layers.ASPPModule(nout * 8, nout * 8, dilations, dropout=True)

        self.dec4 = layers.Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = layers.LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = layers.Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        h = self.aspp(e5)

        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)

        return h


class CascadedNet(nn.Module):

    def __init__(self, n_fft, hop_length, nout=32, nout_lstm=128, is_complex=False, is_mono=False):
        super(CascadedNet, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.is_complex = is_complex
        self.is_mono = is_mono
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        nin = 4 if is_complex else 2
        if is_mono:
            nin = nin // 2

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(nin, nout // 2, self.nin_lstm // 2, nout_lstm),
            layers.Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0)
        )
        self.stg1_high_band_net = BaseNet(
            nin, nout // 4, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + nin, nout, self.nin_lstm // 2, nout_lstm),
            layers.Conv2DBNActiv(nout, nout // 2, 1, 1, 0)
        )
        self.stg2_high_band_net = BaseNet(
            nout // 4 + nin, nout // 2, self.nin_lstm // 2, nout_lstm // 2
        )

        self.stg3_full_band_net = BaseNet(
            3 * nout // 4 + nin, nout, self.nin_lstm, nout_lstm
        )

        self.out = nn.Conv2d(nout, nin, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, nin, 1, bias=False)

    def forward(self, x):
        if self.is_complex:
            x = torch.cat([x.real, x.imag], dim=1)

        x = x[:, :, :self.max_bin]

        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)

        if self.is_complex:
            mask = self.out(f3)
            if self.is_mono:
                mask = torch.complex(mask[:, :1], mask[:, 1:])
            else:
                mask = torch.complex(mask[:, :2], mask[:, 2:])
            mask = self.bounded_mask(mask)
        else:
            mask = torch.sigmoid(self.out(f3))

        mask = F.pad(
            input=mask,
            pad=(0, 0, 0, self.output_bin - mask.size()[2]),
            mode='replicate'
        )

        return mask

    def bounded_mask(self, mask, eps=1e-8):
        mask_mag = torch.abs(mask)
        mask = torch.tanh(mask_mag) * mask / (mask_mag + eps)
        return mask

    def predict_mask(self, x):
        mask = self.forward(x)

        if self.offset > 0:
            mask = mask[:, :, :, self.offset:-self.offset]
            assert mask.size()[3] > 0

        return mask

    def predict(self, x):
        mask = self.forward(x)
        pred = x * mask

        if self.offset > 0:
            pred = pred[:, :, :, self.offset:-self.offset]
            assert pred.size()[3] > 0

        return pred

    def audio2spec(self, x, use_pad=False):
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        if use_pad:
            n_frames = T // self.hop_length + 1
            T_pad = (32 * ((n_frames - 1) // 32 + 1) - 1) * self.hop_length - T
            nl_pad = T_pad // 2 // self.hop_length
            Tl_pad = nl_pad * self.hop_length
            x = F.pad(x, (Tl_pad, T_pad - Tl_pad))
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=self.window,
            pad_mode='constant')
        spec = spec.reshape(B, C, spec.shape[-2], spec.shape[-1])
        return spec

    def spec2audio(self, x):
        B, C, N, T = x.shape
        x = x.reshape(-1, N, T)
        x = torch.istft(x, self.n_fft, self.hop_length, window=self.window)
        x = x.reshape(B, C, -1)
        return x

    def predict_from_audio(self, x):
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        n_frames = T // self.hop_length + 1
        T_pad = (32 * (n_frames // 32 + 1) - 1) * self.hop_length - T
        nl_pad = T_pad // 2 // self.hop_length
        Tl_pad = nl_pad * self.hop_length
        x = F.pad(x, (Tl_pad, T_pad - Tl_pad))
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=self.window,
            pad_mode='constant')
        spec = spec.reshape(B, C, spec.shape[-2], spec.shape[-1])
        mask = self.forward(spec)
        spec_pred = spec * mask
        spec_pred = spec_pred.reshape(B * C, spec.shape[-2], spec.shape[-1])
        x_pred = torch.istft(spec_pred, self.n_fft, self.hop_length, window=self.window)
        x_pred = x_pred[:, Tl_pad: Tl_pad + T]
        x_pred = x_pred.reshape(B, C, T)
        return x_pred
