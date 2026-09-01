"""
Regression tests for the two memory-destroying defects:

- merging distinct events that differ only by an identifier
- deep_prune deleting the whole graph on its first run
"""

from datetime import datetime, timedelta, UTC

import networkx as nx
import pytest

from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine, literal_tokens
from driftguard.utils.node_roles import has_role
from driftguard.graph.prune_engine import PruneEngine
from driftguard.models.event import Event


class DummyPersistence:
    def save_graph(self, graph):
        pass

    def load_graph(self):
        return None


class NearIdenticalEmbeddingEngine:
    """
    Stands in for MiniLM, which scores short strings differing only by a
    digit at ~0.93 — far above the 0.72 action merge threshold.
    """

    def embed(self, text: str):
        return [1.0, 0.0]

    def model_name(self) -> str:
        return "near-identical-stub"


@pytest.fixture
def graph_store(monkeypatch):
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.normalize_text",
        lambda text: text.lower().strip(),
    )
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.EmbeddingEngine",
        lambda model_name=None, device=None: NearIdenticalEmbeddingEngine(),
    )
    return GraphStore(
        merge_engine=MergeEngine(),
        prune_engine=PruneEngine(),
        persistence_engine=DummyPersistence(),
    )


# =====================================================
# MERGE: identifiers must not collapse
# =====================================================

def test_literal_tokens_picks_up_identifiers():
    assert literal_tokens("delete user 1") == frozenset({"1"})
    assert literal_tokens("run migration 23") == frozenset({"23"})
    assert literal_tokens("restart worker-7 on port 8080") == frozenset(
        {"worker-7", "8080"}
    )
    assert literal_tokens("increase salt") == frozenset()


def test_events_differing_only_by_identifier_stay_separate(graph_store):
    for index in (1, 2, 3):
        graph_store.add_event(
            Event(
                action=f"delete user {index}",
                feedback=f"user {index} gone",
                outcome=f"ticket {index}",
            )
        )

    actions = [
        node
        for node, data in graph_store.graph.nodes(data=True)
        if has_role(data["type"], "action")
    ]
    assert sorted(actions) == ["delete user 1", "delete user 2", "delete user 3"]
    assert graph_store.graph.nodes["delete user 1"]["frequency"] == 1


def test_identifier_presence_blocks_merge(graph_store):
    graph_store.add_event(
        Event(action="delete user 1", feedback="user 1 gone", outcome="ticket 1")
    )
    graph_store.add_event(
        Event(
            action="delete all users",
            feedback="everyone gone",
            outcome="full outage",
        )
    )

    actions = {
        node
        for node, data in graph_store.graph.nodes(data=True)
        if has_role(data["type"], "action")
    }
    assert actions == {"delete user 1", "delete all users"}


def test_paraphrases_without_identifiers_still_merge(graph_store):
    graph_store.add_event(
        Event(action="increase salt", feedback="too salty", outcome="dish ruined")
    )
    graph_store.add_event(
        Event(action="add more salt", feedback="over-seasoned", outcome="dish tossed")
    )

    actions = [
        node
        for node, data in graph_store.graph.nodes(data=True)
        if has_role(data["type"], "action")
    ]
    assert actions == ["increase salt"]
    assert graph_store.graph.nodes["increase salt"]["frequency"] == 2


# =====================================================
# PRUNE: fresh memory must survive
# =====================================================

def _graph_with_one_chain(age_days: int) -> nx.DiGraph:
    stamp = datetime.now(UTC) - timedelta(days=age_days)
    graph = nx.DiGraph()

    for text, node_type in (
        ("increase salt", "action"),
        ("too salty", "feedback"),
        ("dish ruined", "outcome"),
    ):
        graph.add_node(text, type=node_type, frequency=1, first_seen=stamp, last_seen=stamp)

    graph.add_edge("increase salt", "too salty", frequency=1, weight=1.0, created_at=stamp)
    graph.add_edge("too salty", "dish ruined", frequency=1, weight=1.0, created_at=stamp)
    return graph


def test_deep_prune_keeps_a_memory_recorded_once_today():
    graph = _graph_with_one_chain(age_days=0)
    summary = PruneEngine().deep_prune(graph)

    assert summary["after"] == {"nodes": 3, "edges": 2}
    assert summary["removed_weak_edges"] == 0
    assert summary["removed_isolated_nodes"] == 0


def test_deep_prune_still_removes_stale_unreinforced_memory():
    graph = _graph_with_one_chain(age_days=90)
    summary = PruneEngine(node_stale_days=60).deep_prune(graph)

    assert summary["after"] == {"nodes": 0, "edges": 0}
    assert summary["removed_weak_edges"] == 2


def test_deep_prune_keeps_a_reinforced_stale_chain_intact():
    graph = _graph_with_one_chain(age_days=90)
    for src, dst in graph.edges:
        graph[src][dst]["frequency"] = 5
    for node in graph.nodes:
        graph.nodes[node]["last_seen"] = datetime.now(UTC)

    summary = PruneEngine(node_stale_days=60).deep_prune(graph)

    assert summary["after"] == {"nodes": 3, "edges": 2}


def test_deep_prune_keeps_nodes_and_edges_without_timestamps():
    graph = nx.DiGraph()
    graph.add_node("legacy action", type="action", frequency=1)
    graph.add_node("legacy feedback", type="feedback", frequency=1)
    graph.add_edge("legacy action", "legacy feedback", frequency=1, weight=1.0)

    summary = PruneEngine().deep_prune(graph)

    assert summary["after"] == {"nodes": 2, "edges": 1}
