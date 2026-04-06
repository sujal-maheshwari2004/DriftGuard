import logging
import os


_PACKAGE_LOGGER_NAME = "driftguard"
_DEFAULT_LEVEL = "INFO"
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level

    level_name = (level or os.getenv("DRIFTGUARD_LOG_LEVEL", _DEFAULT_LEVEL)).upper()
    return getattr(logging, level_name, logging.INFO)


def configure_logging(level: str | int | None = None) -> logging.Logger:
    """
    Configure package-local logging for DriftGuard.

    Safe to call multiple times. The first call installs a stream handler,
    and later calls only adjust the log level.
    """

    logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
    resolved_level = _resolve_level(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(resolved_level)
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a package-scoped logger.
    """

    if name == _PACKAGE_LOGGER_NAME:
        return logging.getLogger(_PACKAGE_LOGGER_NAME)

    suffix = name.split(_PACKAGE_LOGGER_NAME, maxsplit=1)[-1].lstrip(".")
    return logging.getLogger(
        f"{_PACKAGE_LOGGER_NAME}.{suffix}" if suffix else _PACKAGE_LOGGER_NAME
    )
