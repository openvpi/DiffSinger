from typing import Any, List, Optional, Tuple

from pydantic import BaseModel

from lib.conf.io import load_raw_config
from lib.conf.schema import ConfigurationScope, RootConfig


class ModelFormatter:
    def __init__(self, line_width: int = 80, indent: int = 4, connector: str = ": ", separator: str = ", "):
        self.line_width = line_width
        self.indent = indent
        self.connector = connector
        self.separator = separator
        self.lines = []
        self.current_level = 0
        self.current_line = []
        self.current_width = 0
        self.width_cache = {}

    def reset(self):
        self.lines = []
        self.current_level = 0
        self.current_line = []
        self.current_width = 0
        self.width_cache.clear()

    def current_max_width(self):
        return self.line_width - self.current_level * self.indent

    def current_remaining_width(self):
        return self.current_max_width() - self.current_width

    def get_width(self, key: Optional[str], value: Any) -> int:
        cache_key = id(value)
        if cache_key in self.width_cache:
            return self.width_cache[cache_key]

        key_width = 0 if key is None else len(key) + len(self.connector)
        if isinstance(value, BaseModel):
            width = float('inf')
        elif isinstance(value, (tuple, list)):
            item_count = len(value)
            items_width = sum(self.get_width(None, item) for item in value)
            separators_width = (item_count - 1) * len(self.separator)
            brackets_width = 2
            width = items_width + key_width + separators_width + brackets_width
        elif isinstance(value, dict):
            item_count = len(value)
            items_width = sum(self.get_width(k, v) for k, v in value.items())
            separators_width = (item_count - 1) * len(self.separator)
            braces_width = 2
            width = items_width + key_width + separators_width + braces_width
        else:
            width = len(str(value)) + key_width

        self.width_cache[cache_key] = width
        return width

    def new_line(self):
        if self.current_line:
            self.lines.append(" " * (self.current_level * self.indent) + "".join(self.current_line))
            self.current_line = []
            self.current_width = 0

    def add_entries(self, entries: List[Tuple[Optional[str], Any]]):
        for idx, (key, value) in enumerate(entries):
            width = self.get_width(key, value)
            if width > self.current_remaining_width():
                self.new_line()
                self.add_entry(key, value)
                if idx < len(entries) - 1:
                    self.current_line.append(self.separator)
                    self.current_width += len(self.separator)
                if width > self.current_max_width():
                    self.new_line()
            else:
                self.add_entry(key, value)
                if idx < len(entries) - 1:
                    self.current_line.append(self.separator)
                    self.current_width += len(self.separator)

    def add_entry(self, key: Optional[str], value: Any):
        width = self.get_width(key, value)
        if width > self.current_remaining_width():
            self.new_line()
        if key is None:
            prefix = ""
        else:
            prefix = f"{key}{self.connector}"

        def _process(_elements: List[Tuple[Optional[str], Any]], _prefix: str, _suffix: str):
            self.current_line.append(_prefix)
            if width > self.current_max_width():
                self.new_line()
                self.current_level += 1
            self.add_entries(_elements)
            if width > self.current_max_width():
                self.new_line()
                self.current_level -= 1
            self.current_line.append(_suffix)

        if isinstance(value, tuple):
            _process([(None, item) for item in value], prefix + "(", ")")
        elif isinstance(value, list):
            _process([(None, item) for item in value], prefix + "[", "]")
        elif isinstance(value, dict):
            _process(value.items(), prefix + "{", "}")
        elif isinstance(value, BaseModel):
            _process(
                [
                    (name, getattr(value, name))
                    for name, field in type(value).model_fields.items()
                    if hasattr(value, name) and not field.exclude
                ],
                prefix + f"{value.__class__.__name__}(", ")",
            )
        else:
            self.current_line.append(f"{prefix}{value}")
            self.current_width += width

    def format(self, model: BaseModel) -> str:
        self.add_entry(None, model)
        self.new_line()
        result = "\n".join(ln.rstrip() for ln in self.lines)
        self.reset()
        return result


if __name__ == '__main__':
    # Example usage
    config = load_raw_config("configs/acoustic_v3.yaml")
    config = RootConfig.model_validate(config, scope=ConfigurationScope.ACOUSTIC)
    config.resolve()
    config.check()
    formatter = ModelFormatter(line_width=120, indent=4)
    formatted_str = formatter.format(config)
    print(formatted_str)
