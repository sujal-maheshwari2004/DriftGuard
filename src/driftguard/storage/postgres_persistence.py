from __future__ import annotations

from datetime import datetime
from importlib import import_module
from typing import Any

import networkx as nx
import numpy as np

from driftguard.errors import DriftGuardDependencyError
from driftguard.logging_config import get_logger
from driftguard.utils.node_roles import parse_roles, serialize_roles


logger = get_logger(__name__)
POSTGRES_SCHEMA_NAME = "driftguard_postgres"
POSTGRES_SCHEMA_VERSION = 1


def _load_sqlalchemy():
    try:
        sa = import_module("sqlalchemy")
        postgresql_dialect = import_module("sqlalchemy.dialects.postgresql")
    except ImportError as exc:
        raise DriftGuardDependencyError(
            "DriftGuard could not import SQLAlchemy, which is required for the "
            "Postgres persistence backend. Install it with "
            'pip install "driftguard[postgres]".'
        ) from exc
    return sa, postgresql_dialect


class PostgresPersistence:
    """
    Postgres-backed graph persistence for DriftGuard.

    Structurally mirrors SQLitePersistence: the in-memory graph model stays
    the same, only the storage backend changes. Built on SQLAlchemy core, so
    importing this module never requires SQLAlchemy or a Postgres driver to
    be installed -- only constructing an instance does.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        engine: Any | None = None,
        table_prefix: str = "",
    ):
        sa, postgresql_dialect = _load_sqlalchemy()
        self._sa = sa
        self._table_prefix = table_prefix

        if engine is not None:
            self._engine = engine
        elif dsn:
            self._engine = sa.create_engine(dsn)
        else:
            raise DriftGuardDependencyError(
                "PostgresPersistence requires either a 'dsn' or an injected 'engine'."
            )

        embedding_type = sa.JSON().with_variant(
            postgresql_dialect.JSONB(), "postgresql"
        )

        self._metadata = sa.MetaData()
        self._meta_table = sa.Table(
            f"{table_prefix}driftguard_meta",
            self._metadata,
            sa.Column("key", sa.Text, primary_key=True),
            sa.Column("value", sa.Text, nullable=False),
        )
        self._nodes_table = sa.Table(
            f"{table_prefix}driftguard_nodes",
            self._metadata,
            sa.Column("text", sa.Text, primary_key=True),
            sa.Column("type", sa.Text, nullable=False),
            sa.Column("embedding", embedding_type, nullable=True),
            sa.Column("frequency", sa.Integer, nullable=False),
            sa.Column("first_seen", sa.Text, nullable=True),
            sa.Column("last_seen", sa.Text, nullable=True),
        )
        self._edges_table = sa.Table(
            f"{table_prefix}driftguard_edges",
            self._metadata,
            sa.Column(
                "src",
                sa.Text,
                sa.ForeignKey(f"{table_prefix}driftguard_nodes.text", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "dst",
                sa.Text,
                sa.ForeignKey(f"{table_prefix}driftguard_nodes.text", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("frequency", sa.Integer, nullable=False),
            sa.Column("weight", sa.Float, nullable=False),
            sa.Column("created_at", sa.Text, nullable=True),
        )

        logger.info(
            "Postgres persistence configured with dsn=%s table_prefix=%r",
            self._redact_dsn(dsn) if dsn else "<injected engine>",
            table_prefix,
        )

    def save_graph(self, graph: nx.DiGraph) -> None:
        sa = self._sa
        with self._engine.begin() as connection:
            self._ensure_schema(connection)
            connection.execute(sa.delete(self._edges_table))
            connection.execute(sa.delete(self._nodes_table))

            node_rows = [
                {
                    "text": node_text,
                    "type": serialize_roles(node_data.get("type")),
                    "embedding": self._serialize_embedding(node_data.get("embedding")),
                    "frequency": int(node_data.get("frequency", 1)),
                    "first_seen": self._serialize_datetime(node_data.get("first_seen")),
                    "last_seen": self._serialize_datetime(node_data.get("last_seen")),
                }
                for node_text, node_data in graph.nodes(data=True)
            ]
            if node_rows:
                connection.execute(sa.insert(self._nodes_table), node_rows)

            edge_rows = [
                {
                    "src": src,
                    "dst": dst,
                    "frequency": int(edge_data.get("frequency", 1)),
                    "weight": float(edge_data.get("weight", 1.0)),
                    "created_at": self._serialize_datetime(edge_data.get("created_at")),
                }
                for src, dst, edge_data in graph.edges(data=True)
            ]
            if edge_rows:
                connection.execute(sa.insert(self._edges_table), edge_rows)

        logger.info(
            "Saved graph to Postgres nodes=%d edges=%d",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    def load_graph(self) -> nx.DiGraph | None:
        sa = self._sa

        if not sa.inspect(self._engine).has_table(self._nodes_table.name):
            logger.info("Postgres persistence tables do not exist yet")
            return None

        graph = nx.DiGraph()

        with self._engine.begin() as connection:
            self._ensure_schema(connection)

            for row in connection.execute(sa.select(self._nodes_table)):
                graph.add_node(
                    row.text,
                    type=parse_roles(row.type),
                    embedding=self._deserialize_embedding(row.embedding),
                    frequency=row.frequency,
                    first_seen=self._deserialize_datetime(row.first_seen),
                    last_seen=self._deserialize_datetime(row.last_seen),
                )

            for row in connection.execute(sa.select(self._edges_table)):
                graph.add_edge(
                    row.src,
                    row.dst,
                    frequency=row.frequency,
                    weight=row.weight,
                    created_at=self._deserialize_datetime(row.created_at),
                )

        logger.info(
            "Loaded graph from Postgres nodes=%d edges=%d",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    def _ensure_schema(self, connection) -> None:
        sa = self._sa
        self._metadata.create_all(connection, checkfirst=True)

        schema_name = self._get_meta_value(connection, "schema_name")
        schema_version = self._get_meta_value(connection, "schema_version")

        if schema_name is None and schema_version is None:
            connection.execute(
                sa.insert(self._meta_table),
                [
                    {"key": "schema_name", "value": POSTGRES_SCHEMA_NAME},
                    {"key": "schema_version", "value": str(POSTGRES_SCHEMA_VERSION)},
                ],
            )
            return

        if schema_name != POSTGRES_SCHEMA_NAME:
            raise ValueError(f"Unsupported Postgres schema name: {schema_name!r}")

        if schema_version != str(POSTGRES_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported Postgres schema version: {schema_version!r}"
            )

    def _get_meta_value(self, connection, key: str) -> str | None:
        sa = self._sa
        row = connection.execute(
            sa.select(self._meta_table.c.value).where(self._meta_table.c.key == key)
        ).fetchone()
        return None if row is None else str(row[0])

    def _serialize_embedding(self, embedding):
        if embedding is None:
            return None

        if isinstance(embedding, np.ndarray):
            return embedding.tolist()

        return list(embedding)

    def _deserialize_embedding(self, embedding):
        if embedding is None:
            return None

        return np.array(embedding, dtype=np.float32)

    def _serialize_datetime(self, value: datetime | None) -> str | None:
        return None if value is None else value.isoformat()

    def _deserialize_datetime(self, value: str | None) -> datetime | None:
        return None if value is None else datetime.fromisoformat(value)

    def _redact_dsn(self, dsn: str) -> str:
        try:
            return self._sa.engine.make_url(dsn).render_as_string(hide_password=True)
        except Exception:
            return "<redacted>"
