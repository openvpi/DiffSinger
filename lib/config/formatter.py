import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel

__all__ = [
    "format_model",
]

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub('', s)


@dataclass
class _Fmt:
    line_width: int
    indent: int
    connector: str
    separator: str
    lines: list[str] = field(default_factory=list)
    level: int = 0
    cur_line: list[str] = field(default_factory=list)
    cur_width: int = 0
    width_cache: dict = field(default_factory=dict)

    def max_width(self) -> int:
        return self.line_width - self.level * self.indent

    def remaining_width(self) -> int:
        return self.max_width() - self.cur_width

    def flush_line(self):
        if self.cur_line:
            self.lines.append(" " * (self.level * self.indent) + "".join(self.cur_line))
            self.cur_line = []
            self.cur_width = 0


def _get_width(fmt: _Fmt, key: Optional[str], value: Any) -> int:
    cache_key = (key, id(value))
    if cache_key in fmt.width_cache:
        return fmt.width_cache[cache_key]

    key_width = 0 if key is None else len(key) + len(fmt.connector)
    if isinstance(value, BaseModel):
        if any(
            not field.exclude and hasattr(value, name) and getattr(value, name) is not None
            for name, field in type(value).model_fields.items()
        ):
            width = float('inf')  # force new line
        else:
            width = key_width + 2  # wrap empty object
    elif isinstance(value, (tuple, list)):
        item_count = len(value)
        items_width = sum(_get_width(fmt, None, item) for item in value)
        separators_width = (item_count - 1) * len(fmt.separator)
        brackets_width = 2
        width = items_width + key_width + separators_width + brackets_width
    elif isinstance(value, dict):
        item_count = len(value)
        items_width = sum(_get_width(fmt, k, v) for k, v in value.items())
        separators_width = (item_count - 1) * len(fmt.separator)
        braces_width = 2
        width = items_width + key_width + separators_width + braces_width
    else:
        width = len(str(value)) + key_width

    fmt.width_cache[cache_key] = width
    return width


def _entries_width(fmt: _Fmt, entries: list) -> int:
    items_width = sum(_get_width(fmt, k, v) for k, v in entries)
    separators_width = (len(entries) - 1) * len(fmt.separator)
    return items_width + separators_width + 2  # brackets


def _process(fmt: _Fmt, elements: list, prefix: str, suffix: str):
    fmt.cur_line.append(prefix)
    fmt.cur_width += len(_strip_ansi(prefix))
    if _entries_width(fmt, elements) > fmt.max_width():
        fmt.flush_line()
        fmt.level += 1
    _add_entries(fmt, elements)
    if _entries_width(fmt, elements) > fmt.max_width():
        fmt.flush_line()
        fmt.level -= 1
    fmt.cur_line.append(suffix)
    fmt.cur_width += len(_strip_ansi(suffix))


def _add_entries(fmt: _Fmt, entries: list):
    for idx, (key, value) in enumerate(entries):
        width = _get_width(fmt, key, value)
        if width > fmt.remaining_width():
            fmt.flush_line()
            _add_entry(fmt, key, value)
            if idx < len(entries) - 1:
                fmt.cur_line.append(fmt.separator)
                fmt.cur_width += len(fmt.separator)
            if width > fmt.max_width():
                fmt.flush_line()
        else:
            _add_entry(fmt, key, value)
            if idx < len(entries) - 1:
                fmt.cur_line.append(fmt.separator)
                fmt.cur_width += len(fmt.separator)


def _add_entry(fmt: _Fmt, key: Optional[str], value: Any):
    width = _get_width(fmt, key, value)
    if width > fmt.remaining_width():
        fmt.flush_line()
    prefix = "" if key is None else f"\033[0;33m{key}\033[0m{fmt.connector}"

    if isinstance(value, tuple):
        _process(fmt, [(None, item) for item in value], prefix + "(", ")")
    elif isinstance(value, list):
        _process(fmt, [(None, item) for item in value], prefix + "[", "]")
    elif isinstance(value, dict):
        _process(fmt, list(value.items()), prefix + "{", "}")
    elif isinstance(value, BaseModel):
        _process(
            fmt,
            [
                (name, getattr(value, name))
                for name, field in type(value).model_fields.items()
                if hasattr(value, name) and not field.exclude
            ],
            prefix + f"{value.__class__.__name__}(",
            ")",
        )
    else:
        fmt.cur_line.append(f"{prefix}{value}")
        fmt.cur_width += width


def format_model(
    model: BaseModel, line_width: int = 80, indent: int = 4,
    connector: str = ": ", separator: str = ", ",
) -> str:
    """Format a Pydantic model as a readable string."""
    fmt = _Fmt(line_width=line_width, indent=indent, connector=connector, separator=separator)
    _add_entry(fmt, None, model)
    fmt.flush_line()
    return "\n".join(ln.rstrip() for ln in fmt.lines)
