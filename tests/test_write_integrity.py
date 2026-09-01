"""
Regression tests for partial writes and unvalidated input.

record(action=None, ...) used to raise an AttributeError from inside
str.lower(), and only after the action node had already been added to the
graph. The orphan then survived in memory and was persisted by the next
successful write.
"""

import pytest

from driftguard.errors import DriftGuardError, EventValidationError
from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine
from driftguard.graph.prune_engine import PruneEngine
from driftguard.models.event import Event


class DummyPersistence:
    def save_graph(self, graph):
        pass

    def load_graph(self):
        return None


class OrthogonalEmbeddingEngine:
    def __init__(self):
        self._seen: dict[str, int] = {}

    def embed(self, text: str):
        index = self._seen.setdefault(text, len(self._seen))
        vector = [0.0] * 16
        vector[index % 16] = 1.0
        return vector

    def model_name(self) -> str:
        return "orthogonal-stub"


@pytest.fixture
def graph_store(monkeypatch):
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.normalize_text",
        lambda text: text.lower().strip(),
    )
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.EmbeddingEngine",
        lambda model_name=None, device=None: OrthogonalEmbeddingEngine(),
    )
    return GraphStore(
        merge_engine=MergeEngine(),
        prune_engine=PruneEngine(),
        persistence_engine=DummyPersistence(),
    )


# =====================================================
# VALIDATION
# =====================================================

@pytest.mark.parametrize("value", [None, 123, 4.5, [], {}])
def test_non_string_fields_are_rejected(value):
    with pytest.raises(EventValidationError):
        Event(action=value, feedback="x", outcome="y")


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_blank_fields_are_rejected(value):
    with pytest.raises(EventValidationError):
        Event(action="a", feedback=value, outcome="y")


def test_validation_error_names_the_offending_field():
    with pytest.raises(EventValidationError, match="outcome"):
        Event(action="a", feedback="b", outcome=None)


def test_validation_error_is_catchable_as_a_driftguard_error():
    with pytest.raises(DriftGuardError):
        Event(action=None, feedback="b", outcome="c")


def test_valid_events_are_unaffected():
    event = Event(action="deploy", feedback="broke", outcome="rollback")
    assert event.action == "deploy"
    assert event.confidence == 1.0


# =====================================================
# ATOMICITY
# =====================================================

def test_a_rejected_event_never_reaches_the_graph(graph_store):
    with pytest.raises(EventValidationError):
        graph_store.add_event(
            Event(action="valid action here", feedback=None, outcome="x")
        )

    assert graph_store.stats() == {"nodes": 0, "edges": 0}


def test_a_failure_during_normalization_leaves_no_orphan(graph_store, monkeypatch):
    def explode_on_feedback(text: str) -> str:
        if text == "server crashed":
            raise RuntimeError("normalization backend is down")
        return text.lower().strip()

    monkeypatch.setattr(graph_store.merge_engine, "normalize", explode_on_feedback)

    with pytest.raises(RuntimeError):
        graph_store.add_event(
            Event(
                action="restart the server",
                feedback="server crashed",
                outcome="service restored",
            )
        )

    assert graph_store.stats() == {"nodes": 0, "edges": 0}
    assert list(graph_store.graph.nodes) == []


def test_a_failure_during_embedding_leaves_no_orphan(graph_store, monkeypatch):
    def explode_on_outcome(text: str):
        if text == "service restored":
            raise RuntimeError("embedding backend is down")
        return [1.0] + [0.0] * 15

    monkeypatch.setattr(graph_store.merge_engine, "embed", explode_on_outcome)

    with pytest.raises(RuntimeError):
        graph_store.add_event(
            Event(
                action="restart the server",
                feedback="server crashed",
                outcome="service restored",
            )
        )

    assert graph_store.stats() == {"nodes": 0, "edges": 0}


def test_a_failed_write_does_not_corrupt_an_existing_graph(graph_store, monkeypatch):
    graph_store.add_event(
        Event(action="increase salt", feedback="too salty", outcome="dish ruined")
    )
    before = graph_store.stats()

    monkeypatch.setattr(
        graph_store.merge_engine,
        "embed",
        lambda text: (_ for _ in ()).throw(RuntimeError("embedding backend is down")),
    )

    with pytest.raises(RuntimeError):
        graph_store.add_event(
            Event(action="raise pan heat", feedback="burned", outcome="unusable")
        )

    assert graph_store.stats() == before


def test_each_field_is_embedded_once_per_event(graph_store):
    calls = []
    original = graph_store.merge_engine.embed
    graph_store.merge_engine.embed = lambda text: (calls.append(text), original(text))[1]

    graph_store.add_event(
        Event(action="increase salt", feedback="too salty", outcome="dish ruined")
    )

    assert sorted(calls) == ["dish ruined", "increase salt", "too salty"]
