"""
Regression tests for negation surviving normalization and blocking a merge.

Two halves have to hold for "do not deploy on friday" to stay distinct from
"do deploy on friday": normalization must keep the negation, and the merge
guard must treat it as discriminative. Keeping it in the string alone is not
enough — the encoder still scores the pair at ~0.93, well above the 0.72
action threshold.

spaCy's en_core_web_sm is not installed in CI, so normalization is exercised
against a fake pipeline that mimics the token attributes it relies on.
"""

import pytest

from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine, discriminative_tokens
from driftguard.graph.prune_engine import PruneEngine
from driftguard.models.event import Event
from driftguard.utils import normalization


STOPWORDS = {
    "a",
    "always",
    "do",
    "never",
    "not",
    "on",
    "the",
    "with",
    "without",
}


class FakeToken:
    def __init__(self, text: str):
        self.lower_ = text
        self.lemma_ = text
        self.is_stop = text in STOPWORDS
        self.is_punct = not any(character.isalnum() for character in text)
        self.dep_ = "neg" if text in {"not", "n't"} else "dep"


@pytest.fixture
def fake_nlp(monkeypatch):
    monkeypatch.setattr(
        normalization,
        "_get_nlp",
        lambda: (lambda text: [FakeToken(part) for part in text.split()]),
    )


class IdenticalEmbeddingEngine:
    """Stands in for an encoder that cannot tell negation apart."""

    def embed(self, text: str):
        return [1.0, 0.0]

    def model_name(self) -> str:
        return "identical-stub"


@pytest.fixture
def graph_store(monkeypatch):
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.normalize_text",
        lambda text: text.lower().strip(),
    )
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.EmbeddingEngine",
        lambda model_name=None, device=None: IdenticalEmbeddingEngine(),
    )
    return GraphStore(
        merge_engine=MergeEngine(),
        prune_engine=PruneEngine(),
        persistence_engine=type(
            "DummyPersistence",
            (),
            {"save_graph": lambda self, graph: None, "load_graph": lambda self: None},
        )(),
    )


# =====================================================
# NORMALIZATION KEEPS POLARITY
# =====================================================

def test_negation_survives_normalization(fake_nlp):
    assert normalization.normalize_text("do not deploy on friday") == "not deploy friday"
    assert normalization.normalize_text("do deploy on friday") == "deploy friday"


def test_opposite_frequency_adverbs_stay_distinct(fake_nlp):
    assert normalization.normalize_text("never delete the backup") == "never delete backup"
    assert normalization.normalize_text("always delete the backup") == "always delete backup"


def test_ordinary_stopwords_are_still_dropped(fake_nlp):
    assert normalization.normalize_text("deploy the app on a friday") == "deploy app friday"


def test_punctuation_is_still_dropped(fake_nlp):
    assert normalization.normalize_text("deploy , the app !") == "deploy app"


# =====================================================
# MERGE GUARD TREATS POLARITY AS DISCRIMINATIVE
# =====================================================

def test_discriminative_tokens_include_polarity_and_identifiers():
    assert discriminative_tokens("not deploy friday") == frozenset({"not"})
    assert discriminative_tokens("never delete backup") == frozenset({"never"})
    assert discriminative_tokens("deploy without run test") == frozenset({"without"})
    assert discriminative_tokens("run migration 23") == frozenset({"23"})
    assert discriminative_tokens("increase salt") == frozenset()


def test_negated_action_does_not_merge_into_its_opposite(graph_store):
    graph_store.add_event(
        Event(action="deploy friday", feedback="site down", outcome="rollback")
    )
    graph_store.add_event(
        Event(action="not deploy friday", feedback="quiet weekend", outcome="no pages")
    )

    actions = sorted(
        node
        for node, data in graph_store.graph.nodes(data=True)
        if "action" in data["type"]
    )
    assert actions == ["deploy friday", "not deploy friday"]


def test_opposite_adverbs_do_not_merge(graph_store):
    graph_store.add_event(
        Event(action="never delete backup", feedback="safe", outcome="recovered")
    )
    graph_store.add_event(
        Event(action="always delete backup", feedback="unsafe", outcome="data lost")
    )

    actions = sorted(
        node
        for node, data in graph_store.graph.nodes(data=True)
        if "action" in data["type"]
    )
    assert actions == ["always delete backup", "never delete backup"]


def test_paraphrases_without_polarity_still_merge(graph_store):
    graph_store.add_event(
        Event(action="increase salt", feedback="too salty", outcome="dish ruined")
    )
    graph_store.add_event(
        Event(action="add more salt", feedback="over-seasoned", outcome="dish tossed")
    )

    actions = [
        node
        for node, data in graph_store.graph.nodes(data=True)
        if "action" in data["type"]
    ]
    assert actions == ["increase salt"]
