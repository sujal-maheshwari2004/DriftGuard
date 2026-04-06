from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """
    Central embedding interface for DriftGuard.

    Responsibilities:
    - provide sentence embeddings
    - allow easy backend swapping later
    - support batch embedding

    Future upgrade targets:
    - OpenAI embeddings
    - Azure embeddings
    - Instructor models
    - local GGUF encoders
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
    ):
        """
        model_name:
            SentenceTransformer model identifier

        device:
            "cpu", "cuda", or None (auto-detect)
        """

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
        Generate embeddings for multiple strings.
        """

        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

    # =====================================================
    # MODEL INFO (optional helper)
    # =====================================================

    def model_name(self):
        """
        Return active embedding model name.
        """

        return self.model._first_module().auto_model.config._name_or_path