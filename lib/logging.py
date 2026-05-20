import contextvars
import inspect
import sys

from loguru import logger

LOGGER_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level:<8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
_PRIVATE_LOGGER_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level:<8}</level> | "
    "<cyan>{extra[name]}</cyan>:<cyan>{extra[function]}</cyan>:<cyan>{extra[line]}</cyan> - <level>{message}</level>"
)
_log_callback: contextvars.ContextVar = contextvars.ContextVar("_log_callback", default=None)

logger.level("DEBUG", color="")
logger.level("INFO", color="<green>")
logger.remove()


def _private_sink(msg):
    """Permanent sink that routes formatted output based on context variable."""
    cb = _log_callback.get()
    if cb is not None:
        cb(msg.rstrip())
    else:
        print(msg.rstrip())


logger.add(sys.stdout, colorize=True, format=LOGGER_FORMAT,
           filter=lambda record: not record["extra"].get("_routed", False))
logger.add(_private_sink, colorize=True, format=_PRIVATE_LOGGER_FORMAT,
           filter=lambda record: record["extra"].get("_routed", False))


def _get_bind(last=1):
    """Walk the stack to get the real caller's module, function, and line."""
    frame = inspect.currentframe()
    try:
        # Get the caller's frame
        for _ in range(last + 1):
            frame = frame.f_back
        name = frame.f_globals["__name__"]
        function = frame.f_code.co_name
        line = frame.f_lineno
        return {"name": name, "function": function, "line": line}
    finally:
        del frame


def _log(level: str, sink_callback, bind: dict, message: str):
    token = _log_callback.set(sink_callback)
    try:
        logger.bind(_routed=True, **bind).log(level, message)
    finally:
        _log_callback.reset(token)


def trace(message: str, callback=None):
    _log("TRACE", callback, _get_bind(), message)


def debug(message: str, callback=None):
    _log("DEBUG", callback, _get_bind(), message)


def info(message: str, callback=None):
    _log("INFO", callback, _get_bind(), message)


def success(message: str, callback=None):
    _log("SUCCESS", callback, _get_bind(), message)


def warning(message: str, callback=None):
    _log("WARNING", callback, _get_bind(), message)


def error(message: str, callback=None):
    _log("ERROR", callback, _get_bind(), message)


def critical(message: str, callback=None):
    _log("CRITICAL", callback, _get_bind(), message)
