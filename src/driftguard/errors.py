class DriftGuardError(RuntimeError):
    """
    Base class for DriftGuard runtime errors.
    """


class DriftGuardDependencyError(DriftGuardError):
    """
    Raised when an optional runtime dependency is unavailable or misconfigured.
    """


class EmbeddingDependencyError(DriftGuardDependencyError):
    """
    Raised when the embedding backend cannot be loaded or used.
    """


class NormalizationDependencyError(DriftGuardDependencyError):
    """
    Raised when the text normalization backend cannot be loaded or used.
    """
