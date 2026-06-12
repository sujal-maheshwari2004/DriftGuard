from datetime import UTC, datetime

import networkx as nx
import numpy as np
import pytest

sa = pytest.importorskip("sqlalchemy")
from sqlalchemy.pool import StaticPool

from driftguard.storage.postgres_persistence import (
    POSTGRES_SCHEMA_NAME,
    POSTGRES_SCHEMA_VERSION,
    PostgresPersistence,
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
def engine():
    return sa.create_engine("sqlite://", poolclass=StaticPool)


@pytest.fixture
def persistence(engine):
    return PostgresPersistence(engine=engine)


def test_postgres_persistence_round_trips_graph(persistence):
    """Postgres persistence should restore nodes, edges, embeddings, and timestamps."""

    persistence.save_graph(_build_graph())

    loaded = persistence.load_graph()

    assert loaded is not None
    assert loaded.number_of_nodes() == 2
    assert loaded.number_of_edges() == 1
    assert loaded.nodes["increase salt"]["frequency"] == 4
    assert isinstance(loaded.nodes["increase salt"]["embedding"], np.ndarray)
    assert loaded["increase salt"]["too salty"]["frequency"] == 3


def test_postgres_persistence_overwrites_previous_graph(persistence):
    """Saving a new graph should replace the previously persisted contents."""

    persistence.save_graph(_build_graph())

    smaller_graph = nx.DiGraph()
    smaller_graph.add_node(
        "lower heat",
        type="action",
        embedding=np.array([0.1, 0.2], dtype=np.float32),
        frequency=1,
        first_seen=datetime.now(UTC),
        last_seen=datetime.now(UTC),
    )
    persistence.save_graph(smaller_graph)

    loaded = persistence.load_graph()

    assert loaded.number_of_nodes() == 1
    assert loaded.number_of_edges() == 0
    assert "lower heat" in loaded.nodes


def test_postgres_persistence_handles_null_embeddings(persistence):
    """Postgres persistence should round-trip nodes that have no embedding."""

    graph = nx.DiGraph()
    graph.add_node(
        "no embedding yet",
        type="action",
        embedding=None,
        frequency=1,
        first_seen=None,
        last_seen=None,
    )
    persistence.save_graph(graph)

    loaded = persistence.load_graph()

    assert loaded.nodes["no embedding yet"]["embedding"] is None
    assert loaded.nodes["no embedding yet"]["first_seen"] is None
    assert loaded.nodes["no embedding yet"]["last_seen"] is None


def test_postgres_persistence_creates_expected_meta(persistence, engine):
    """Postgres persistence should initialize schema metadata for future migrations."""

    persistence.save_graph(_build_graph())

    with engine.connect() as connection:
        schema_name = connection.execute(
            sa.text("SELECT value FROM driftguard_meta WHERE key = 'schema_name'")
        ).scalar_one()
        schema_version = connection.execute(
            sa.text("SELECT value FROM driftguard_meta WHERE key = 'schema_version'")
        ).scalar_one()

    assert schema_name == POSTGRES_SCHEMA_NAME
    assert schema_version == str(POSTGRES_SCHEMA_VERSION)


def test_postgres_persistence_rejects_unknown_schema_version(persistence, engine):
    """Postgres persistence should fail clearly for unsupported future schemas."""

    persistence.save_graph(_build_graph())

    with engine.begin() as connection:
        connection.execute(
            sa.text("UPDATE driftguard_meta SET value = '999' WHERE key = 'schema_version'")
        )

    with pytest.raises(ValueError, match="Unsupported Postgres schema version"):
        persistence.load_graph()


def test_postgres_persistence_returns_none_when_no_tables_exist(persistence):
    """Postgres persistence should behave like other backends when nothing exists yet."""

    assert persistence.load_graph() is None


def test_postgres_persistence_redacts_password_in_dsn(persistence):
    """Postgres persistence should never log a raw DSN password."""

    redacted = persistence._redact_dsn(
        "postgresql+psycopg://user:supersecret@localhost:5432/driftguard"
    )

    assert "supersecret" not in redacted
    assert "user" in redacted
