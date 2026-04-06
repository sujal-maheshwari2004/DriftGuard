import pytest

from DriftGuard.models.event import Event
from DriftGuard.graph.graph_store import GraphStore
from DriftGuard.graph.merge_engine import MergeEngine
from DriftGuard.graph.prune_engine import PruneEngine
from DriftGuard.retrieval.retrieval_engine import RetrievalEngine


# =====================================================
# DUMMY PERSISTENCE — no disk I/O during tests
# =====================================================

class DummyPersistence:

    def save_graph(self, graph):
        pass

    def load_graph(self):
        return None


# =====================================================
# SHARED FIXTURES
# =====================================================

@pytest.fixture(scope="module")
def graph_store():
    store = GraphStore(
        merge_engine=MergeEngine(),
        prune_engine=PruneEngine(),
        persistence_engine=DummyPersistence(),
    )
    return store


@pytest.fixture(scope="module")
def retriever(graph_store):
    return RetrievalEngine(graph_store)


# =====================================================
# TESTS
# =====================================================

def test_event_timestamp_is_unique():
    """Each Event should get a fresh timestamp, not a shared one."""

    e1 = Event(action="a", feedback="b", outcome="c")
    e2 = Event(action="a", feedback="b", outcome="c")

    assert e1.timestamp != e2.timestamp


def test_add_event_creates_nodes(graph_store):
    """Adding an event should create nodes in the graph."""

    before = graph_store.stats()["nodes"]

    graph_store.add_event(
        Event(
            action="increase salt",
            feedback="too salty",
            outcome="dish ruined",
        )
    )

    assert graph_store.stats()["nodes"] > before


def test_semantic_merge(graph_store):
    """
    Semantically similar events should merge into existing nodes
    rather than creating duplicates.
    """

    before = graph_store.stats()["nodes"]

    graph_store.add_event(
        Event(
            action="add more salt",
            feedback="over-seasoned",
            outcome="dish ruined",
        )
    )

    # Node count should stay the same or grow minimally — not double
    after = graph_store.stats()["nodes"]
    assert after - before <= 1


def test_retrieval_returns_warnings(graph_store, retriever):
    """Querying a known action should return at least one warning."""

    result = retriever.query("increase salt")

    assert result.query == "increase salt"
    assert len(result.warnings) > 0


def test_warnings_sorted_by_confidence(graph_store, retriever):
    """Warnings should be ordered highest confidence first."""

    result = retriever.query("increase salt")

    confidences = [w.confidence for w in result.warnings]
    assert confidences == sorted(confidences, reverse=True)


def test_no_duplicate_warnings(graph_store, retriever):
    """The same (trigger, risk) pair should not appear twice."""

    result = retriever.query("increase salt")

    pairs = [(w.trigger, w.risk) for w in result.warnings]
    assert len(pairs) == len(set(pairs))


def test_cosine_similarity_zero_vector():
    """Cosine similarity should return 0.0 for zero vectors, not crash."""

    import numpy as np
    from DriftGuard.utils.similarity import cosine_similarity

    a = np.zeros(384)
    b = np.ones(384)

    assert cosine_similarity(a, b) == 0.0


def test_persistence_save_load(tmp_path):
    """Save and load cycle should preserve graph structure."""

    import networkx as nx
    from datetime import datetime, UTC
    from DriftGuard.storage.persistence import Persistence
    import numpy as np

    p = Persistence(filepath=str(tmp_path / "test.json"))

    g = nx.DiGraph()
    now = datetime.now(UTC)
    g.add_node(
        "test_node",
        type="action",
        embedding=np.random.rand(384).astype(np.float32),
        frequency=3,
        first_seen=now,
        last_seen=now,
    )

    p.save_graph(g)
    loaded = p.load_graph()

    assert "test_node" in loaded.nodes
    assert loaded.nodes["test_node"]["frequency"] == 3
    assert isinstance(loaded.nodes["test_node"]["embedding"], np.ndarray)