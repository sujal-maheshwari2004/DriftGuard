"""
Regression tests for concurrent writers clobbering each other.

Every backend used to write by replacing the stored graph with the caller's
whole in-memory graph, so two runtimes pointed at the same file silently
destroyed each other's memories.
"""

from datetime import datetime, timedelta, UTC
import threading

import networkx as nx
import pytest

from driftguard.storage.file_lock import FileLock, GraphLockTimeout
from driftguard.storage.graph_merge import merge_graphs
from driftguard.storage.persistence import Persistence
from driftguard.storage.sqlite_persistence import SQLitePersistence


def chain(action: str, feedback: str, *, frequency: int = 1, stamp=None) -> nx.DiGraph:
    stamp = stamp or datetime.now(UTC)
    graph = nx.DiGraph()
    graph.add_node(
        action,
        type=("action",),
        embedding=None,
        frequency=frequency,
        first_seen=stamp,
        last_seen=stamp,
    )
    graph.add_node(
        feedback,
        type=("feedback",),
        embedding=None,
        frequency=frequency,
        first_seen=stamp,
        last_seen=stamp,
    )
    graph.add_edge(action, feedback, frequency=frequency, weight=1.0, created_at=stamp)
    return graph


# =====================================================
# MERGE SEMANTICS
# =====================================================

def test_merge_keeps_both_sets_of_nodes():
    merged = merge_graphs(chain("scale up", "cost spike"), chain("disable retries", "dropped"))

    assert set(merged.nodes) == {"scale up", "cost spike", "disable retries", "dropped"}
    assert merged.number_of_edges() == 2


def test_merge_takes_the_larger_frequency_rather_than_the_sum():
    merged = merge_graphs(
        chain("scale up", "cost spike", frequency=3),
        chain("scale up", "cost spike", frequency=5),
    )

    assert merged.nodes["scale up"]["frequency"] == 5
    assert merged["scale up"]["cost spike"]["frequency"] == 5


def test_merge_keeps_the_earliest_first_seen_and_latest_last_seen():
    old = datetime.now(UTC) - timedelta(days=10)
    new = datetime.now(UTC)
    merged = merge_graphs(
        chain("scale up", "cost spike", stamp=old),
        chain("scale up", "cost spike", stamp=new),
    )

    assert merged.nodes["scale up"]["first_seen"] == old
    assert merged.nodes["scale up"]["last_seen"] == new


def test_merge_unions_roles():
    base = chain("restart", "crashed")
    incoming = chain("restart", "crashed")
    incoming.nodes["restart"]["type"] = ("outcome",)

    merged = merge_graphs(base, incoming)

    assert set(merged.nodes["restart"]["type"]) == {"action", "outcome"}


def test_merge_does_not_let_a_missing_embedding_win():
    base = chain("restart", "crashed")
    incoming = chain("restart", "crashed")
    incoming.nodes["restart"]["embedding"] = [1.0, 0.0]

    merged = merge_graphs(base, incoming)

    assert merged.nodes["restart"]["embedding"] == [1.0, 0.0]


# =====================================================
# BACKENDS
# =====================================================

@pytest.fixture(params=["json", "sqlite"])
def persistence(request, tmp_path):
    if request.param == "json":
        return Persistence(filepath=str(tmp_path / "graph.json"))
    return SQLitePersistence(filepath=str(tmp_path / "graph.sqlite3"))


def test_a_second_writer_does_not_erase_the_first(persistence):
    persistence.save_graph(chain("scale up workers", "cost spike"))
    persistence.save_graph(chain("disable retries", "requests dropped"))

    stored = persistence.load_graph()

    assert set(stored.nodes) == {
        "scale up workers",
        "cost spike",
        "disable retries",
        "requests dropped",
    }


def test_merge_false_replaces_the_stored_graph(persistence):
    persistence.save_graph(chain("scale up workers", "cost spike"))
    persistence.save_graph(chain("disable retries", "requests dropped"), merge=False)

    stored = persistence.load_graph()

    assert set(stored.nodes) == {"disable retries", "requests dropped"}


def test_pruned_nodes_do_not_come_back(persistence):
    persistence.save_graph(chain("scale up workers", "cost spike"))
    persistence.save_graph(nx.DiGraph(), merge=False)

    assert persistence.load_graph().number_of_nodes() == 0


def test_concurrent_writers_all_survive(persistence):
    def write(index: int):
        persistence.save_graph(chain(f"action {index}", f"feedback {index}"))

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stored = persistence.load_graph()

    for index in range(8):
        assert f"action {index}" in stored
        assert stored.has_edge(f"action {index}", f"feedback {index}")


# =====================================================
# LOCK
# =====================================================

def test_lock_is_exclusive_and_released(tmp_path):
    target = tmp_path / "graph.json"

    with FileLock(target) as lock:
        assert lock.path.exists()
        with pytest.raises(GraphLockTimeout):
            with FileLock(target, timeout=0.05):
                pass

    assert not lock.path.exists()


def test_lock_timeout_is_a_driftguard_error():
    from driftguard.errors import DriftGuardError

    assert issubclass(GraphLockTimeout, DriftGuardError)


def test_a_stale_lock_is_broken_rather_than_waited_on(tmp_path):
    target = tmp_path / "graph.json"
    abandoned = FileLock(target)
    abandoned.path.mkdir()

    with FileLock(target, timeout=0.5, stale_after=0.0):
        assert True

    assert not abandoned.path.exists()


def test_lock_is_released_when_the_save_raises(tmp_path, monkeypatch):
    persistence = Persistence(filepath=str(tmp_path / "graph.json"))
    monkeypatch.setattr(
        persistence,
        "_write_locked",
        lambda graph, merge: (_ for _ in ()).throw(RuntimeError("disk full")),
    )

    with pytest.raises(RuntimeError):
        persistence.save_graph(chain("a", "b"))

    assert not FileLock(persistence.filepath).path.exists()
