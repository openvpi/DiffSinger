class BasePE:
    def get_pitch(
            self, waveform, samplerate, length,
            *, hop_size, f0_min=65, f0_max=1100,
            speed=1, interp_uv=False
    ):
        raise NotImplementedError()
