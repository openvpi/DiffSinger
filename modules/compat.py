def get_backbone_type(root_config: dict, nested_config: dict = None):
    if nested_config is None:
        nested_config = root_config
    return nested_config.get(
        'backbone_type',
        root_config.get(
            'backbone_type',
            root_config.get('diff_decoder_type', 'wavenet')
        )
    )


def get_backbone_args(config: dict, backbone_type: str):
    args = config.get('backbone_args')
    if args is not None:
        return args
    elif backbone_type == 'wavenet':
        return {
            'num_layers': config.get('residual_layers'),
            'num_channels': config.get('residual_channels'),
            'dilation_cycle_length': config.get('dilation_cycle_length'),
            'hidden_size': config.get('hidden_size'),
        }
    else:
        return None
