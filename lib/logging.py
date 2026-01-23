import inspect
import io
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
logger.level("DEBUG", color="")
logger.level("INFO", color="<green>")
logger.remove()
logger.add(sys.stdout, colorize=True, format=LOGGER_FORMAT)


def _get_bind(last=1):
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


def _log(logger_callback, sink_callback, message: str):
    with io.StringIO() as string:
        logger.remove()
        logger.add(string, colorize=True, format=_PRIVATE_LOGGER_FORMAT)
        logger_callback(message)
        logger.remove()
        logger.add(sys.stdout, colorize=True, format=LOGGER_FORMAT)
        formatted_massage = string.getvalue().rstrip()
    if sink_callback is None:
        print(formatted_massage)
    else:
        sink_callback(formatted_massage)


def trace(message: str, callback=None):
    _log(logger.bind(**_get_bind()).trace, callback, message)


def debug(message: str, callback=None):
    _log(logger.bind(**_get_bind()).debug, callback, message)


def info(message: str, callback=None):
    _log(logger.bind(**_get_bind()).info, callback, message)


def success(message: str, callback=None):
    _log(logger.bind(**_get_bind()).success, callback, message)


def warning(message: str, callback=None):
    _log(logger.bind(**_get_bind()).warning, callback, message)


def error(message: str, callback=None):
    _log(logger.bind(**_get_bind()).error, callback, message)


def critical(message: str, callback=None):
    _log(logger.bind(**_get_bind()).critical, callback, message)
