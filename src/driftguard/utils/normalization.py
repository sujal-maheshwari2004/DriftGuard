import spacy

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
        _nlp = spacy.load("en_core_web_sm")

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
