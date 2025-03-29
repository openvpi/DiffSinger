import importlib


VOCODERS = {}


def register_vocoder(cls):
    VOCODERS[cls.__name__.lower()] = cls
    VOCODERS[cls.__name__] = cls
    return cls


def get_vocoder_cls(config):
    if config['vocoder'] in VOCODERS:
        return VOCODERS[config['vocoder']]
    else:
        vocoder_cls = config['vocoder']
        pkg = ".".join(vocoder_cls.split(".")[:-1])
        cls_name = vocoder_cls.split(".")[-1]
        vocoder_cls = getattr(importlib.import_module(pkg), cls_name)
        return vocoder_cls
