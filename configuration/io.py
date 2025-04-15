import pathlib
from typing import List

import omegaconf


def load_raw_config(config_path: pathlib.Path, overrides: List[str] = None) -> dict:
    def _load(path: pathlib.Path) -> omegaconf.DictConfig:
        cfg = omegaconf.OmegaConf.load(path)

        if "bases" in cfg:
            if isinstance(cfg.bases, str):
                cfg.bases = [cfg.bases]
            elif not isinstance(cfg.bases, omegaconf.ListConfig):
                raise TypeError(f"Invalid type <{type(cfg.bases)}> for bases in {path}")
            for base in cfg.bases:
                base_path = pathlib.Path(base).resolve()
                base_cfg = _load(base_path)
                cfg = omegaconf.OmegaConf.merge(base_cfg, cfg)
            del cfg["bases"]

        return cfg

    config = _load(config_path)
    if overrides:
        override_config = omegaconf.OmegaConf.from_dotlist(overrides)
        config = omegaconf.OmegaConf.merge(config, override_config)
    config = omegaconf.OmegaConf.to_container(config, resolve=True)

    return config


def save_raw_config(config: dict, save_path: str):
    omegaconf.OmegaConf.save(config=config, f=save_path, resolve=True)
