"""
A small cross-process lock for the file-backed persistence engines.

SQLite and Postgres bring their own transactions; the JSON backend has nothing,
so a read-modify-write save needs exclusion of its own. Built on os.mkdir,
which is atomic on both POSIX and Windows and needs no extra dependency.
"""

import errno
import os
import time
from pathlib import Path

from driftguard.errors import DriftGuardError
from driftguard.logging_config import get_logger


logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_STALE_AFTER_SECONDS = 60.0
_POLL_SECONDS = 0.02


class GraphLockTimeout(DriftGuardError):
    """
    Raised when another process held the graph lock for too long.
    """


class FileLock:
    """
    Exclusive lock keyed on a path, held for the duration of a `with` block.

    A lock older than `stale_after` is broken rather than waited on, so a
    process that dies mid-save cannot wedge the store permanently.
    """

    def __init__(
        self,
        target: Path,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        stale_after: float = DEFAULT_STALE_AFTER_SECONDS,
    ):
        self.path = Path(f"{target}.lock")
        self.timeout = timeout
        self.stale_after = stale_after

    def __enter__(self) -> "FileLock":
        deadline = time.monotonic() + self.timeout

        while True:
            try:
                self.path.mkdir(parents=True)
                return self
            except FileExistsError:
                pass
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

            if self._break_if_stale():
                continue

            if time.monotonic() >= deadline:
                raise GraphLockTimeout(
                    f"Timed out after {self.timeout:g}s waiting for the graph "
                    f"lock at {self.path}. Another DriftGuard process is "
                    f"holding it; remove the directory if that process is gone."
                )

            time.sleep(_POLL_SECONDS)

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.path.rmdir()
        except OSError:
            logger.warning("Graph lock at %s was already released", self.path)

    def _break_if_stale(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            # Released between the failed mkdir and the stat — retry normally.
            return True

        if age < self.stale_after:
            return False

        logger.warning(
            "Breaking graph lock at %s after %.0fs; the holding process "
            "appears to have died",
            self.path,
            age,
        )
        try:
            self.path.rmdir()
        except OSError:
            return False

        return True
