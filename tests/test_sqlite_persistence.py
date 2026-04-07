from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
import sqlite3

import networkx as nx
import numpy as np
import pytest

from driftguard.storage.sqlite_persistence import (
    SQLITE_SCHEMA_NAME,
    SQLITE_SCHEMA_VERSION,
    SQLitePersistence,
)


def _build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    now = datetime.now(UTC)
    graph.add_node(
        "increase salt",
        type="action",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        frequency=4,
        first_seen=now,
        last_seen=now,
    )
    graph.add_node(
        "too salty",
        type="feedback",
        embedding=np.array([0.5, 0.5], dtype=np.float32),
        frequency=3,
        first_seen=now,
        last_seen=now,
    )
    graph.add_edge(
        "increase salt",
        "too salty",
        frequency=3,
        weight=1.0,
        created_at=now,
    )
    return graph


@pytest.fixture
def sqlite_filepath():
    filepath = Path.cwd() / f"persistence-test-{uuid4().hex}.sqlite3"
    yield filepath
    if filepath.exists():
        filepath.unlink()


def test_sqlite_persistence_round_trips_graph(sqlite_filepath):
    """SQLite persistence should restore nodes, edges, embeddings, and timestamps."""

    persistence = SQLitePersistence(filepath=str(sqlite_filepath))
    persistence.save_graph(_build_graph())

    loaded = persistence.load_graph()

    assert loaded is not None
    assert loaded.number_of_nodes() == 2
    assert loaded.number_of_edges() == 1
    assert loaded.nodes["increase salt"]["frequency"] == 4
    assert isinstance(loaded.nodes["increase salt"]["embedding"], np.ndarray)
    assert loaded["increase salt"]["too salty"]["frequency"] == 3


def test_sqlite_persistence_creates_expected_meta(sqlite_filepath):
    """SQLite persistence should initialize schema metadata for future migrations."""

    persistence = SQLitePersistence(filepath=str(sqlite_filepath))
    persistence.save_graph(_build_graph())

    connection = sqlite3.connect(sqlite_filepath)
    try:
        schema_name = connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_name'"
        ).fetchone()[0]
        schema_version = connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
    finally:
        connection.close()

    assert schema_name == SQLITE_SCHEMA_NAME
    assert schema_version == str(SQLITE_SCHEMA_VERSION)


def test_sqlite_persistence_rejects_unknown_schema_version(sqlite_filepath):
    """SQLite persistence should fail clearly for unsupported future schemas."""

    persistence = SQLitePersistence(filepath=str(sqlite_filepath))
    persistence.save_graph(_build_graph())

    connection = sqlite3.connect(sqlite_filepath)
    try:
        connection.execute(
            "UPDATE meta SET value = '999' WHERE key = 'schema_version'"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="Unsupported SQLite schema version"):
        persistence.load_graph()


def test_sqlite_persistence_returns_none_when_file_is_missing(sqlite_filepath):
    """SQLite persistence should behave like JSON persistence when nothing exists yet."""

    persistence = SQLitePersistence(filepath=str(sqlite_filepath))

    assert persistence.load_graph() is None
