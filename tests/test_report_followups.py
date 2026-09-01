"""
Regression tests for the remaining findings from the end-to-end report.

One file rather than seven, because each fix is small and they share fixtures.
"""

import json
from datetime import datetime, UTC

import networkx as nx
import numpy as np
import pytest

from driftguard.config import DriftGuardSettings
from driftguard.errors import EventValidationError, PersistenceError
from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine
from driftguard.graph.prune_engine import PruneEngine
from driftguard.guard import DriftGuard
from driftguard.models.event import Event
from driftguard.models.response import Reinforcement, Warning
from driftguard.retrieval.retrieval_engine import RetrievalEngine
from driftguard.storage import embedding_codec
from driftguard.storage.persistence import (
    PERSISTENCE_FORMAT_NAME,
    PERSISTENCE_FORMAT_VERSION,
    Persistence,
)
from driftguard.storage.sqlite_persistence import SQLitePersistence


class DummyPersistence:
    def save_graph(self, graph, *, merge=True):
        pass

    def load_graph(self):
        return None


class OrthogonalEmbeddingEngine:
    def __init__(self):
        self._seen: dict[str, int] = {}

    def embed(self, text: str):
        index = self._seen.setdefault(text, len(self._seen))
        vector = np.zeros(16, dtype=np.float32)
        vector[index % 16] = 1.0
        return vector

    def model_name(self) -> str:
        return "orthogonal-stub"


@pytest.fixture
def graph_store(monkeypatch):
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.normalize_text",
        lambda text: " ".join(
            part for part in text.lower().split() if part not in {"the", "a", "of"}
        ),
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
# Empty after normalization
# =====================================================

@pytest.mark.parametrize(
    "fields",
    [
        {"action": "the a of", "feedback": "ok", "outcome": "fine"},
        {"action": "ok", "feedback": "the", "outcome": "fine"},
        {"action": "ok", "feedback": "fine", "outcome": "of the"},
    ],
)
def test_a_field_that_normalizes_away_keeps_its_raw_text(graph_store, fields):
    graph_store.add_event(Event(**fields))

    assert "" not in graph_store.graph
    assert graph_store.stats() == {"nodes": 3, "edges": 2}


def test_degenerate_events_no_longer_collapse_into_one_node(graph_store):
    graph_store.add_event(Event(action="the a of", feedback="the", outcome="of"))
    graph_store.add_event(Event(action="!!!", feedback="???", outcome="..."))

    assert "" not in graph_store.graph
    assert set(graph_store.graph.nodes) == {
        "the a of",
        "the",
        "of",
        "!!!",
        "???",
        "...",
    }
    assert graph_store.graph.number_of_edges() == 4


def test_distinct_degenerate_fields_do_not_self_loop(graph_store):
    graph_store.add_event(Event(action="the", feedback="a", outcome="of"))

    assert not any(src == dst for src, dst in graph_store.graph.edges)
    assert graph_store.stats() == {"nodes": 3, "edges": 2}


def test_a_blank_field_is_still_rejected_outright():
    with pytest.raises(EventValidationError):
        Event(action="   ", feedback="ok", outcome="fine")


# =====================================================
# Settings from the environment
# =====================================================

def test_settings_default_when_nothing_is_set():
    assert DriftGuardSettings.from_env({}) == DriftGuardSettings()


def test_settings_read_strings_ints_and_floats():
    settings = DriftGuardSettings.from_env(
        {
            "DRIFTGUARD_STORAGE_BACKEND": "sqlite",
            "DRIFTGUARD_SQLITE_FILEPATH": "/data/graph.sqlite3",
            "DRIFTGUARD_RETRIEVAL_TOP_K": "9",
            "DRIFTGUARD_RETRIEVAL_MIN_SIMILARITY": "0.8",
            "DRIFTGUARD_GUARD_POLICY": "block",
        }
    )

    assert settings.storage_backend == "sqlite"
    assert settings.sqlite_filepath == "/data/graph.sqlite3"
    assert settings.retrieval_top_k == 9
    assert settings.retrieval_min_similarity == 0.8
    assert settings.guard_policy == "block"


def test_settings_ignore_unrelated_variables():
    settings = DriftGuardSettings.from_env({"PATH": "/usr/bin", "HOME": "/root"})

    assert settings == DriftGuardSettings()


def test_settings_reject_an_unparseable_value():
    with pytest.raises(ValueError, match="DRIFTGUARD_RETRIEVAL_TOP_K"):
        DriftGuardSettings.from_env({"DRIFTGUARD_RETRIEVAL_TOP_K": "many"})


# =====================================================
# Persistence errors
# =====================================================

def test_unparseable_file_raises_a_persistence_error(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text("not json at all", encoding="utf-8")

    with pytest.raises(PersistenceError, match="graph.json"):
        Persistence(filepath=str(path)).load_graph()


def test_the_persistence_error_says_how_to_recover(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(PersistenceError, match="Move or delete the file"):
        Persistence(filepath=str(path)).load_graph()


# =====================================================
# Version
# =====================================================

def test_version_matches_installed_metadata():
    from importlib.metadata import PackageNotFoundError, version

    import driftguard

    for distribution in ("driftguard-ai", "driftguard"):
        try:
            assert driftguard.__version__ == version(distribution)
            return
        except PackageNotFoundError:
            continue

    # Nothing is installed — a source checkout on sys.path.
    assert driftguard.__version__ == "0.0.0+unknown"


def test_version_is_not_hardcoded():
    source = (
        __import__("pathlib")
        .Path(__import__("driftguard").__file__)
        .read_text(encoding="utf-8")
    )

    assert '__version__ = "0.' not in source


# =====================================================
# record_only is not overridden by raise_on_match
# =====================================================

class StubRuntime:
    def __init__(self, response):
        self._response = response
        self.settings = DriftGuardSettings(guard_policy="record_only")
        self.metrics = type(
            "Metrics", (), {"record_review": lambda self, **kwargs: None}
        )()

    def query_memory(self, context):
        return self._response


def _response_with_a_warning():
    from driftguard.models.response import RetrievalResponse

    return RetrievalResponse(
        query="deploy",
        warnings=[Warning(trigger="deploy", risk="broke", frequency=9, confidence=0.99)],
        chains=[["deploy", "broke", "outage"]],
        confidence=0.99,
    )


def test_record_only_survives_raise_on_match():
    guard = DriftGuard(
        runtime=StubRuntime(_response_with_a_warning()),
        settings=DriftGuardSettings(guard_policy="record_only"),
    )

    review = guard.before_step("deploy", raise_on_match=True)

    assert review.warnings == []
    assert review.confidence == 0.0


def test_raise_on_match_still_blocks_under_the_warn_default():
    from driftguard.guard import GuardrailTriggered

    guard = DriftGuard(
        runtime=StubRuntime(_response_with_a_warning()),
        settings=DriftGuardSettings(guard_policy="warn"),
    )

    with pytest.raises(GuardrailTriggered):
        guard.before_step("deploy", raise_on_match=True)


# =====================================================
# The outcome reaches the warning
# =====================================================

def test_warnings_and_reinforcements_carry_the_outcome(graph_store):
    graph_store.add_event(
        Event(action="increase salt", feedback="too salty", outcome="dish ruined")
    )
    engine = RetrievalEngine(graph_store, min_similarity=0.0)

    response = engine.query("increase salt")

    assert response.warnings
    assert response.warnings[0].risk == "too salty"
    assert response.warnings[0].outcome == "dish ruined"


def test_outcome_defaults_to_none_so_existing_callers_are_unaffected():
    warning = Warning(trigger="a", risk="b", frequency=1, confidence=0.5)
    reinforcement = Reinforcement(
        trigger="a", recommendation="b", frequency=1, confidence=0.5
    )

    assert warning.outcome is None
    assert reinforcement.outcome is None


# =====================================================
# node_link edges key
# =====================================================

def test_saved_graphs_pin_the_edges_key(tmp_path):
    path = tmp_path / "graph.json"
    graph = nx.DiGraph()
    graph.add_edge("a", "b", frequency=1, weight=1.0, created_at=datetime.now(UTC))
    Persistence(filepath=str(path)).save_graph(graph)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "edges" in payload["graph"]


def test_a_file_written_with_the_old_links_key_still_loads(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "format": PERSISTENCE_FORMAT_NAME,
                "format_version": 1,
                "graph": {
                    "directed": True,
                    "multigraph": False,
                    "graph": {},
                    "nodes": [{"id": "a", "type": "action"}, {"id": "b", "type": "feedback"}],
                    "links": [{"source": "a", "target": "b", "frequency": 2}],
                },
            }
        ),
        encoding="utf-8",
    )

    graph = Persistence(filepath=str(path)).load_graph()

    assert graph.has_edge("a", "b")
    assert graph["a"]["b"]["frequency"] == 2


# =====================================================
# Embedding codec
# =====================================================

def test_embedding_round_trips_through_the_packed_form():
    vector = np.array([0.5, -0.25, 1.0], dtype=np.float32)

    restored = embedding_codec.decode(embedding_codec.encode(vector))

    assert np.array_equal(restored, vector)


def test_the_packed_form_is_much_smaller_than_json_floats():
    vector = np.random.default_rng(0).random(384, dtype=np.float32)

    packed = len(embedding_codec.encode(vector))
    as_json = len(json.dumps(vector.tolist()))

    assert packed < as_json / 3


def test_a_list_of_floats_still_decodes():
    restored = embedding_codec.decode([0.5, -0.25, 1.0])

    assert np.array_equal(restored, np.array([0.5, -0.25, 1.0], dtype=np.float32))


def test_encode_and_decode_pass_none_through():
    assert embedding_codec.encode(None) is None
    assert embedding_codec.decode(None) is None


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_graphs_written_with_packed_embeddings_round_trip(tmp_path, backend):
    if backend == "json":
        persistence = Persistence(filepath=str(tmp_path / "graph.json"))
    else:
        persistence = SQLitePersistence(filepath=str(tmp_path / "graph.sqlite3"))

    vector = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    graph = nx.DiGraph()
    stamp = datetime.now(UTC)
    graph.add_node(
        "a",
        type=("action",),
        embedding=vector,
        frequency=1,
        first_seen=stamp,
        last_seen=stamp,
    )
    persistence.save_graph(graph)

    stored = persistence.load_graph()

    assert np.array_equal(stored.nodes["a"]["embedding"], vector)


def test_a_version_1_file_with_list_embeddings_still_loads(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "format": PERSISTENCE_FORMAT_NAME,
                "format_version": 1,
                "graph": {
                    "directed": True,
                    "multigraph": False,
                    "graph": {},
                    "nodes": [
                        {"id": "a", "type": "action", "embedding": [0.5, -0.25, 1.0]}
                    ],
                    "edges": [],
                },
            }
        ),
        encoding="utf-8",
    )

    graph = Persistence(filepath=str(path)).load_graph()

    assert np.array_equal(
        graph.nodes["a"]["embedding"], np.array([0.5, -0.25, 1.0], dtype=np.float32)
    )


def test_current_saves_use_the_new_format_version(tmp_path):
    path = tmp_path / "graph.json"
    Persistence(filepath=str(path)).save_graph(nx.DiGraph())

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["format_version"] == PERSISTENCE_FORMAT_VERSION == 2
