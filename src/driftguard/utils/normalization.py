from importlib import import_module

from driftguard.errors import NormalizationDependencyError
from driftguard.logging_config import get_logger

_nlp = None
logger = get_logger(__name__)


def _get_nlp():
    """
    Lazy-load spaCy model on first use.
    Avoids slow import-time load and makes testing faster.
    """

    global _nlp

    if _nlp is None:
        logger.info("Loading spaCy normalization model en_core_web_sm")
        try:
            spacy = import_module("spacy")
        except Exception as exc:
            logger.exception("Failed to import spaCy")
            raise NormalizationDependencyError(
                "DriftGuard could not import spaCy. Install the 'spacy' package "
                "to enable text normalization."
            ) from exc

        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception as exc:
            logger.exception("Failed to load spaCy model en_core_web_sm")
            raise NormalizationDependencyError(
                "DriftGuard could not load the spaCy model 'en_core_web_sm'. "
                "Install it with: python -m spacy download en_core_web_sm"
            ) from exc

    return _nlp


def normalize_text(text: str) -> str:
    """
    Lowercase, lemmatize, and strip stopwords and punctuation.
    """

    nlp = _get_nlp()

    doc = nlp(text.lower().strip())

    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    normalized = " ".join(lemmas)
    logger.debug("Normalized text %r -> %r", text, normalized)
    return normalized
