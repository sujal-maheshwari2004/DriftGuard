from sentence_transformers import SentenceTransformer


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

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    # =====================================================
    # SINGLE EMBEDDING
    # =====================================================

    def embed(self, text: str):
        """
        Generate embedding for a single string.
        """

        return self.model.encode(
            text,
            normalize_embeddings=True,
        )

    # =====================================================
    # BATCH EMBEDDING
    # =====================================================

    def embed_batch(self, texts):
        """
        Generate embeddings for a list of strings.
        """

        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

    # =====================================================
    # MODEL INFO
    # =====================================================

    def model_name(self) -> str:
        """
        Return active embedding model name.
        Stored at init — avoids accessing private internals.
        """

        return self._model_name