from importlib import import_module

from driftguard.errors import NormalizationDependencyError
from driftguard.logging_config import get_logger

_nlp = None
logger = get_logger(__name__)

# spaCy classifies these as stopwords, but dropping them makes an instruction
# and its opposite normalize to the same string: "do not deploy on friday" and
# "do deploy on friday" both became "deploy friday", so the two merged into one
# node and a warning recorded for one was surfaced for the other.
POLARITY_WORDS = frozenset(
    {
        "no",
        "not",
        "n't",
        "never",
        "none",
        "neither",
        "nor",
        "nothing",
        "nowhere",
        "cannot",
        "without",
        "always",
    }
)


def _is_polarity(token) -> bool:
    """
    True for tokens that flip or fix the meaning of the rest of the phrase.

    Checks the dependency label as well as the word list, so contractions and
    negations spaCy tags but the list misses are still kept.
    """

    return (
        token.dep_ == "neg"
        or token.lower_ in POLARITY_WORDS
        or token.lemma_.lower() in POLARITY_WORDS
    )


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


def ensure_available() -> None:
    """
    Load the spaCy model now so a missing one is reported at startup.

    en_core_web_sm cannot be declared as a dependency — PyPI rejects direct
    URL requirements, and the model is not on the index — so it has to be
    installed by hand. Without this check the failure surfaced at the first
    record() call, which is the worst possible moment: the agent is mid-run
    and trying to write down a mistake.
    """

    _get_nlp()


def normalize_text(text: str) -> str:
    """
    Lowercase, lemmatize, and strip stopwords and punctuation.

    Polarity words survive the stopword filter — see _is_polarity().
    """

    nlp = _get_nlp()

    doc = nlp(text.lower().strip())

    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_punct and (not token.is_stop or _is_polarity(token))
    ]

    normalized = " ".join(lemmas)
    logger.debug("Normalized text %r -> %r", text, normalized)
    return normalized
