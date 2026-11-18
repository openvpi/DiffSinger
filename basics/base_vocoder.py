class BaseVocoder:
    def to_device(self, device):
        """
        
        :param device: torch.device or str
        """
        raise NotImplementedError()

    def get_device(self):
        """
        
        :return: device: torch.device or str
        """
        raise NotImplementedError()
    
    def spec2wav(self, mel, **kwargs):
        """

        :param mel: [T, num_mels]
        :return: wav: [T']
        """

        raise NotImplementedError()
