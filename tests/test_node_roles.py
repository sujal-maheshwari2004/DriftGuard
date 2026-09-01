"""
Regression tests for text that holds more than one role.

Nodes are keyed by normalized text, so "restart the server" can arrive as an
action in one event and as an outcome in another. Before roles were a set, the
second write replaced the first node's role and frequency, and the action
became unreachable.
"""

import pytest

from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine
from driftguard.graph.prune_engine import PruneEngine
from driftguard.models.event import Event
from driftguard.utils.node_roles import (
    add_role,
    has_role,
    parse_roles,
    serialize_roles,
)


class DummyPersistence:
    def save_graph(self, graph):
        pass

    def load_graph(self):
        return None


class OrthogonalEmbeddingEngine:
    """Every text gets its own axis, so nothing merges on similarity."""

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
# ROLE HELPERS
# =====================================================

def test_parse_roles_accepts_legacy_string():
    assert parse_roles("action") == ("action",)


def test_parse_roles_accepts_serialized_and_iterable_forms():
    assert parse_roles("action,outcome") == ("action", "outcome")
    assert parse_roles(["action", "outcome"]) == ("action", "outcome")
    assert parse_roles(("action", "action")) == ("action",)


def test_parse_roles_tolerates_missing_and_empty_values():
    assert parse_roles(None) == ()
    assert parse_roles("") == ()
    assert parse_roles([]) == ()


def test_add_role_is_idempotent_and_order_preserving():
    roles = add_role("action", "outcome")
    assert roles == ("action", "outcome")
    assert add_role(roles, "action") == ("action", "outcome")


def test_serialize_roles_round_trips():
    assert parse_roles(serialize_roles(("action", "outcome"))) == (
        "action",
        "outcome",
    )


# =====================================================
# GRAPH BEHAVIOUR
# =====================================================

def test_action_survives_when_it_is_also_an_outcome(graph_store):
    graph_store.add_event(
        Event(
            action="restart the server",
            feedback="server crashed",
            outcome="restart the server",
        )
    )

    roles = graph_store.graph.nodes["restart the server"]["type"]
    assert has_role(roles, "action")
    assert has_role(roles, "outcome")

    actions = graph_store.find_similar_nodes("restart the server", node_type="action")
    assert actions == ["restart the server"]


def test_collision_does_not_reset_frequency(graph_store):
    graph_store.add_event(
        Event(action="retry the job", feedback="still failing", outcome="gave up")
    )
    graph_store.add_event(
        Event(action="escalate", feedback="asked on-call", outcome="retry the job")
    )

    node = graph_store.graph.nodes["retry the job"]
    assert node["frequency"] == 2
    assert parse_roles(node["type"]) == ("action", "outcome")


def test_collision_keeps_the_original_first_seen(graph_store):
    graph_store.add_event(
        Event(action="roll back", feedback="site down", outcome="recovered")
    )
    first_seen = graph_store.graph.nodes["roll back"]["first_seen"]

    graph_store.add_event(
        Event(action="page on-call", feedback="no response", outcome="roll back")
    )

    assert graph_store.graph.nodes["roll back"]["first_seen"] == first_seen


def test_chains_are_still_retrievable_after_a_collision(graph_store):
    graph_store.add_event(
        Event(
            action="restart the server",
            feedback="server crashed",
            outcome="restart the server",
        )
    )

    chains = graph_store.get_related_chains("restart the server")
    assert chains
    assert all(chain[0] == "restart the server" for chain in chains)


def test_unrelated_roles_are_not_conflated(graph_store):
    graph_store.add_event(
        Event(action="increase salt", feedback="too salty", outcome="dish ruined")
    )

    assert parse_roles(graph_store.graph.nodes["increase salt"]["type"]) == ("action",)
    assert parse_roles(graph_store.graph.nodes["too salty"]["type"]) == ("feedback",)
    assert parse_roles(graph_store.graph.nodes["dish ruined"]["type"]) == ("outcome",)
