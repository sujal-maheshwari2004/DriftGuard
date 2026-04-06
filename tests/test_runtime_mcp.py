from dataclasses import dataclass

import pytest

from driftguard.config import DriftGuardSettings
from driftguard.models.response import RetrievalResponse, Warning
from driftguard import mcp as mcp_module
from driftguard import runtime as runtime_module


class FakeGraphStore:
    def __init__(self, merge_engine, prune_engine, persistence_engine):
        self.merge_engine = merge_engine
        self.prune_engine = prune_engine
        self.persistence_engine = persistence_engine
        self.load_calls = 0
        self.save_calls = 0
        self.added_events = []
        self.graph = {"graph": "value"}

    def load(self):
        self.load_calls += 1

    def save(self):
        self.save_calls += 1

    def add_event(self, event):
        self.added_events.append(event)

    def stats(self):
        return {"nodes": 4, "edges": 3}


class FakeMergeEngine:
    def __init__(self, *, settings=None):
        self.settings = settings


class FakePruneEngine:
    def __init__(self, node_stale_days: int, edge_min_frequency: int):
        self.node_stale_days = node_stale_days
        self.edge_min_frequency = edge_min_frequency
        self.deep_prune_calls = []

    def deep_prune(self, graph):
        self.deep_prune_calls.append(graph)


class FakePersistence:
    def __init__(self, filepath: str):
        self.filepath = filepath


class FakeRetrievalEngine:
    def __init__(self, graph_store, top_k: int, min_similarity: float):
        self.graph_store = graph_store
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.queries = []

    def query(self, context: str):
        self.queries.append(context)
        return RetrievalResponse(
            query=context,
            warnings=[Warning("increase salt", "too salty", 2, 0.8)],
            chains=[["increase salt", "too salty", "dish ruined"]],
            confidence=0.8,
        )


def test_build_runtime_threads_settings_into_components(monkeypatch):
    settings = DriftGuardSettings(
        graph_filepath="custom-graph.json",
        retrieval_top_k=7,
        retrieval_min_similarity=0.77,
        prune_node_stale_days=12,
        prune_edge_min_frequency=4,
    )

    monkeypatch.setattr(runtime_module, "MergeEngine", FakeMergeEngine)
    monkeypatch.setattr(runtime_module, "PruneEngine", FakePruneEngine)
    monkeypatch.setattr(runtime_module, "Persistence", FakePersistence)
    monkeypatch.setattr(runtime_module, "GraphStore", FakeGraphStore)
    monkeypatch.setattr(runtime_module, "RetrievalEngine", FakeRetrievalEngine)

    runtime = runtime_module.build_runtime(settings=settings)

    assert runtime.settings is settings
    assert runtime.merge_engine.settings is settings
    assert runtime.prune_engine.node_stale_days == 12
    assert runtime.prune_engine.edge_min_frequency == 4
    assert runtime.persistence.filepath == "custom-graph.json"
    assert runtime.retrieval_engine.top_k == 7
    assert runtime.retrieval_engine.min_similarity == 0.77
    assert runtime.graph_store.load_calls == 1


def test_build_runtime_can_skip_auto_load(monkeypatch):
    settings = DriftGuardSettings()

    monkeypatch.setattr(runtime_module, "MergeEngine", FakeMergeEngine)
    monkeypatch.setattr(runtime_module, "PruneEngine", FakePruneEngine)
    monkeypatch.setattr(runtime_module, "Persistence", FakePersistence)
    monkeypatch.setattr(runtime_module, "GraphStore", FakeGraphStore)
    monkeypatch.setattr(runtime_module, "RetrievalEngine", FakeRetrievalEngine)

    runtime = runtime_module.build_runtime(
        settings=settings,
        auto_load=False,
    )

    assert runtime.graph_store.load_calls == 0


def test_runtime_register_query_prune_and_stats_delegate_to_components():
    runtime = runtime_module.DriftGuardRuntime(
        settings=DriftGuardSettings(),
        merge_engine=object(),
        prune_engine=FakePruneEngine(node_stale_days=1, edge_min_frequency=1),
        persistence=FakePersistence(filepath="graph.json"),
        graph_store=FakeGraphStore(None, None, None),
        retrieval_engine=FakeRetrievalEngine(
            graph_store=None,
            top_k=5,
            min_similarity=0.6,
        ),
    )
    runtime.retrieval_engine.graph_store = runtime.graph_store

    record = runtime.register_mistake("increase salt", "too salty", "dish ruined")
    query = runtime.query_memory("increase salt")
    prune = runtime.deep_prune()
    stats = runtime.graph_stats()

    assert record["status"] == "stored"
    assert runtime.graph_store.save_calls == 2
    assert runtime.graph_store.added_events[0].action == "increase salt"
    assert query.query == "increase salt"
    assert runtime.prune_engine.deep_prune_calls == [runtime.graph_store.graph]
    assert prune == {"status": "pruned", "before": {"nodes": 4, "edges": 3}, "after": {"nodes": 4, "edges": 3}}
    assert stats == {"nodes": 4, "edges": 3}


@dataclass
class FakeRuntime:
    query_response: RetrievalResponse

    def __post_init__(self):
        self.calls = []

    def register_mistake(self, action: str, feedback: str, outcome: str):
        self.calls.append(("register_mistake", action, feedback, outcome))
        return {"status": "stored", "action": action, "feedback": feedback, "outcome": outcome}

    def query_memory(self, context: str):
        self.calls.append(("query_memory", context))
        return self.query_response

    def deep_prune(self):
        self.calls.append(("deep_prune",))
        return {"status": "pruned"}

    def graph_stats(self):
        self.calls.append(("graph_stats",))
        return {"nodes": 2, "edges": 1}


@pytest.mark.anyio
async def test_create_mcp_server_registers_tools_and_calls_runtime():
    runtime = FakeRuntime(
        query_response=RetrievalResponse(
            query="increase salt",
            warnings=[Warning("increase salt", "too salty", 2, 0.8)],
            chains=[["increase salt", "too salty", "dish ruined"]],
            confidence=0.8,
        )
    )

    server = mcp_module.create_mcp_server(runtime=runtime)

    tools = await server.list_tools()
    tool_names = sorted(tool.name for tool in tools)

    assert tool_names == [
        "deep_prune",
        "graph_stats",
        "query_memory",
        "register_mistake",
    ]

    register_result = await server.call_tool(
        "register_mistake",
        {"action": "increase salt", "feedback": "too salty", "outcome": "dish ruined"},
    )
    query_result = await server.call_tool("query_memory", {"context": "increase salt"})
    prune_result = await server.call_tool("deep_prune", {})
    stats_result = await server.call_tool("graph_stats", {})

    assert register_result.structured_content == {
        "status": "stored",
        "action": "increase salt",
        "feedback": "too salty",
        "outcome": "dish ruined",
    }
    assert query_result.structured_content["query"] == "increase salt"
    assert query_result.structured_content["warnings"][0]["risk"] == "too salty"
    assert prune_result.structured_content == {"status": "pruned"}
    assert stats_result.structured_content == {"nodes": 2, "edges": 1}
    assert runtime.calls == [
        ("register_mistake", "increase salt", "too salty", "dish ruined"),
        ("query_memory", "increase salt"),
        ("deep_prune",),
        ("graph_stats",),
    ]


def test_create_mcp_server_builds_runtime_from_settings(monkeypatch):
    settings = DriftGuardSettings(retrieval_top_k=9)
    fake_runtime = FakeRuntime(
        query_response=RetrievalResponse(
            query="q",
            warnings=[],
            chains=[],
            confidence=0.0,
        )
    )
    calls = []

    def fake_build_runtime(*, settings=None):
        calls.append(settings)
        return fake_runtime

    monkeypatch.setattr(mcp_module, "build_runtime", fake_build_runtime)

    server = mcp_module.create_mcp_server(settings=settings)

    assert server is not None
    assert calls == [settings]
