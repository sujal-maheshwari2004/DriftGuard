import pytest
from pathlib import Path
from uuid import uuid4

from driftguard.models.event import Event
from driftguard.models.response import RetrievalResponse, Warning
from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine
from driftguard.graph.prune_engine import PruneEngine
from driftguard.retrieval.retrieval_engine import RetrievalEngine


# =====================================================
# DUMMY PERSISTENCE — no disk I/O during tests
# =====================================================

class DummyPersistence:

    def save_graph(self, graph):
        pass

    def load_graph(self):
        return None


class StubEmbeddingEngine:
    def __init__(self, embeddings: dict[str, list[float]]):
        self.embeddings = embeddings

    def embed(self, text: str):
        return self.embeddings[text]

    def model_name(self) -> str:
        return "stub"


# =====================================================
# SHARED FIXTURES
# =====================================================

@pytest.fixture
def graph_store(monkeypatch):
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.normalize_text",
        lambda text: text.lower().strip(),
    )
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.EmbeddingEngine",
        lambda model_name=None, device=None: StubEmbeddingEngine(
            {
                "increase salt": [1.0, 0.0, 0.0],
                "add more salt": [0.96, 0.04, 0.0],
                "too salty": [0.0, 1.0, 0.0],
                "over-seasoned": [0.0, 0.97, 0.03],
                "dish ruined": [0.0, 0.0, 1.0],
            }
        ),
    )
    store = GraphStore(
        merge_engine=MergeEngine(),
        prune_engine=PruneEngine(),
        persistence_engine=DummyPersistence(),
    )
    return store


@pytest.fixture
def retriever(graph_store):
    return RetrievalEngine(graph_store)


@pytest.fixture
def writable_filepath():
    filepath = Path.cwd() / f"pipeline-test-{uuid4().hex}.json"
    temp_path = filepath.with_suffix(f"{filepath.suffix}.tmp")

    yield filepath

    for path in (filepath, temp_path):
        if path.exists():
            path.unlink()


def _seed_known_event(graph_store):
    graph_store.add_event(
        Event(
            action="increase salt",
            feedback="too salty",
            outcome="dish ruined",
        )
    )


# =====================================================
# TESTS
# =====================================================

def test_event_timestamp_is_unique():
    """Each Event should get a fresh timestamp, not a shared one."""

    e1 = Event(action="a", feedback="b", outcome="c")
    e2 = Event(action="a", feedback="b", outcome="c")

    assert e1.timestamp != e2.timestamp


def test_retrieval_response_timestamp_is_unique():
    """Each RetrievalResponse should get a fresh timestamp."""

    r1 = RetrievalResponse(
        query="q",
        warnings=[Warning("a", "b", 1, 0.6)],
        chains=[["a", "b"]],
        confidence=0.6,
    )
    r2 = RetrievalResponse(
        query="q",
        warnings=[Warning("a", "b", 1, 0.6)],
        chains=[["a", "b"]],
        confidence=0.6,
    )

    assert r1.timestamp != r2.timestamp


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

    _seed_known_event(graph_store)
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

    _seed_known_event(graph_store)
    result = retriever.query("increase salt")

    assert result.query == "increase salt"
    assert len(result.warnings) > 0


def test_warnings_sorted_by_confidence(graph_store, retriever):
    """Warnings should be ordered highest confidence first."""

    _seed_known_event(graph_store)
    result = retriever.query("increase salt")

    confidences = [w.confidence for w in result.warnings]
    assert confidences == sorted(confidences, reverse=True)


def test_no_duplicate_warnings(graph_store, retriever):
    """The same (trigger, risk) pair should not appear twice."""

    _seed_known_event(graph_store)
    result = retriever.query("increase salt")

    pairs = [(w.trigger, w.risk) for w in result.warnings]
    assert len(pairs) == len(set(pairs))


def test_cosine_similarity_zero_vector():
    """Cosine similarity should return 0.0 for zero vectors, not crash."""

    import numpy as np
    from driftguard.utils.similarity import cosine_similarity

    a = np.zeros(384)
    b = np.ones(384)

    assert cosine_similarity(a, b) == 0.0


def test_persistence_save_load(writable_filepath):
    """Save and load cycle should preserve graph structure."""

    import networkx as nx
    from datetime import datetime, UTC
    from driftguard.storage.persistence import Persistence
    import numpy as np

    p = Persistence(filepath=str(writable_filepath))

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


def test_related_chains_avoid_cycles_and_limit_growth():
    """Graph traversal should avoid revisiting nodes and respect path caps."""

    class StaticMergeEngine:
        def normalize(self, text: str) -> str:
            return text

        def find_top_k_similar(self, *args, **kwargs):
            return []

        def find_similar_node(self, *args, **kwargs):
            return None

        def embed(self, text: str):
            return [1.0, 0.0]

    store = GraphStore(
        merge_engine=StaticMergeEngine(),
        prune_engine=PruneEngine(),
        persistence_engine=DummyPersistence(),
        traversal_max_depth=4,
        traversal_max_branching=2,
        traversal_max_paths=2,
    )

    store.graph.add_edge("a", "b", frequency=5)
    store.graph.add_edge("b", "a", frequency=4)
    store.graph.add_edge("b", "c", frequency=3)
    store.graph.add_edge("a", "d", frequency=2)
    store.graph.add_edge("a", "e", frequency=1)

    chains = store.get_related_chains("a")

    assert chains == [["a", "b", "c"], ["a", "d"]]
