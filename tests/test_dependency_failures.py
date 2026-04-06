import pytest

from driftguard.embedding import embedding_engine
from driftguard.errors import EmbeddingDependencyError, NormalizationDependencyError
from driftguard.utils import normalization


def test_embedding_engine_raises_friendly_error_when_backend_import_fails(monkeypatch):
    """EmbeddingEngine should raise a DriftGuard-specific error for missing backend imports."""

    def fail_import(name: str):
        raise ImportError("missing dependency")

    monkeypatch.setattr(embedding_engine, "import_module", fail_import)

    with pytest.raises(EmbeddingDependencyError, match="sentence-transformers"):
        embedding_engine.EmbeddingEngine()


def test_embedding_engine_wraps_encode_failures(monkeypatch):
    """EmbeddingEngine should wrap encode-time failures with a DriftGuard-specific error."""

    class BrokenModel:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, *args, **kwargs):
            raise RuntimeError("encode boom")

    class FakeModule:
        SentenceTransformer = BrokenModel

    monkeypatch.setattr(embedding_engine, "import_module", lambda name: FakeModule())

    engine = embedding_engine.EmbeddingEngine()

    with pytest.raises(EmbeddingDependencyError, match="single text input"):
        engine.embed("hello world")


def test_normalization_raises_friendly_error_when_spacy_import_fails(monkeypatch):
    """Normalization should raise a DriftGuard-specific error for missing spaCy imports."""

    monkeypatch.setattr(normalization, "_nlp", None)

    def fail_import(name: str):
        raise ImportError("missing spacy")

    monkeypatch.setattr(normalization, "import_module", fail_import)

    with pytest.raises(NormalizationDependencyError, match="import spaCy"):
        normalization.normalize_text("hello world")


def test_normalization_raises_friendly_error_when_model_load_fails(monkeypatch):
    """Normalization should raise a DriftGuard-specific error for missing spaCy models."""

    monkeypatch.setattr(normalization, "_nlp", None)

    class FakeSpacyModule:
        @staticmethod
        def load(name: str):
            raise OSError("model missing")

    monkeypatch.setattr(
        normalization,
        "import_module",
        lambda name: FakeSpacyModule(),
    )

    with pytest.raises(NormalizationDependencyError, match="en_core_web_sm"):
        normalization.normalize_text("hello world")
