from importlib import import_module

from driftguard.errors import EmbeddingDependencyError
from driftguard.logging_config import get_logger


logger = get_logger(__name__)


class EmbeddingEngine:
    """
    Central embedding interface for DriftGuard.

    Responsibilities:
    - Provide sentence embeddings
    - Allow easy backend swapping later
    - Support batch embedding

    Future upgrade targets:
    - OpenAI embeddings
    - Azure embeddings
    - Instructor models
    - Local GGUF encoders
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None,
    ):
        self._model_name = model_name
        logger.info(
            "Loading embedding model model_name=%s device=%s",
            model_name,
            device,
        )

        try:
            sentence_transformers = import_module("sentence_transformers")
            self.model = sentence_transformers.SentenceTransformer(
                model_name,
                device=device,
            )
        except Exception as exc:
            logger.exception(
                "Failed to initialize embedding model model_name=%s",
                model_name,
            )
            raise EmbeddingDependencyError(
                "DriftGuard could not initialize the sentence-transformers embedding "
                f"model {model_name!r}. Install the 'sentence-transformers' package "
                "and ensure the model is available locally or downloadable."
            ) from exc
        logger.info("Embedding model ready model_name=%s", model_name)

    # =====================================================
    # SINGLE EMBEDDING
    # =====================================================

    def embed(self, text: str):
        """
        Generate embedding for a single string.
        """

        logger.debug("Embedding single text length=%d", len(text))
        try:
            return self.model.encode(
                text,
                normalize_embeddings=True,
            )
        except Exception as exc:
            logger.exception("Embedding failed for single text")
            raise EmbeddingDependencyError(
                "DriftGuard failed while generating an embedding for a single text input."
            ) from exc

    # =====================================================
    # BATCH EMBEDDING
    # =====================================================

    def embed_batch(self, texts):
        """
        Generate embeddings for a list of strings.
        """

        logger.debug("Embedding batch size=%d", len(texts))
        try:
            return self.model.encode(
                texts,
                normalize_embeddings=True,
            )
        except Exception as exc:
            logger.exception("Embedding failed for batch size=%d", len(texts))
            raise EmbeddingDependencyError(
                "DriftGuard failed while generating embeddings for a batch of texts."
            ) from exc

    # =====================================================
    # MODEL INFO
    # =====================================================

    def model_name(self) -> str:
        """
        Return active embedding model name.
        Stored at init — avoids accessing private internals.
        """

        return self._model_name
