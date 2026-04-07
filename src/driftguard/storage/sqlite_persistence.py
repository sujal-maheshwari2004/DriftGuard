from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import sqlite3

import networkx as nx
import numpy as np

from driftguard.logging_config import get_logger


logger = get_logger(__name__)
SQLITE_SCHEMA_NAME = "driftguard_sqlite"
SQLITE_SCHEMA_VERSION = 1


class SQLitePersistence:
    """
    SQLite-backed graph persistence for DriftGuard.

    The in-memory graph model stays the same; this class only replaces the
    on-disk storage format.
    """

    def __init__(self, filepath: str = "driftguard_graph.sqlite3"):
        self.filepath = Path(filepath)
        logger.info("SQLite persistence configured with filepath=%s", self.filepath)

    def save_graph(self, graph: nx.DiGraph) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(self.filepath)
        try:
            self._ensure_schema(connection)
            connection.execute("BEGIN")
            connection.execute("DELETE FROM edges")
            connection.execute("DELETE FROM nodes")

            for node_text, node_data in graph.nodes(data=True):
                connection.execute(
                    """
                    INSERT INTO nodes (
                        text,
                        type,
                        embedding,
                        frequency,
                        first_seen,
                        last_seen
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_text,
                        node_data.get("type"),
                        self._serialize_embedding(node_data.get("embedding")),
                        int(node_data.get("frequency", 1)),
                        self._serialize_datetime(node_data.get("first_seen")),
                        self._serialize_datetime(node_data.get("last_seen")),
                    ),
                )

            for src, dst, edge_data in graph.edges(data=True):
                connection.execute(
                    """
                    INSERT INTO edges (
                        src,
                        dst,
                        frequency,
                        weight,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        src,
                        dst,
                        int(edge_data.get("frequency", 1)),
                        float(edge_data.get("weight", 1.0)),
                        self._serialize_datetime(edge_data.get("created_at")),
                    ),
                )

            connection.commit()
        finally:
            connection.close()

        logger.info(
            "Saved graph to SQLite %s nodes=%d edges=%d",
            self.filepath,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    def load_graph(self) -> nx.DiGraph | None:
        if not self.filepath.exists():
            logger.info("SQLite persistence file does not exist at %s", self.filepath)
            return None

        connection = sqlite3.connect(self.filepath)
        try:
            self._ensure_schema(connection)

            graph = nx.DiGraph()

            for row in connection.execute(
                """
                SELECT
                    text,
                    type,
                    embedding,
                    frequency,
                    first_seen,
                    last_seen
                FROM nodes
                """
            ):
                graph.add_node(
                    row[0],
                    type=row[1],
                    embedding=self._deserialize_embedding(row[2]),
                    frequency=row[3],
                    first_seen=self._deserialize_datetime(row[4]),
                    last_seen=self._deserialize_datetime(row[5]),
                )

            for row in connection.execute(
                """
                SELECT
                    src,
                    dst,
                    frequency,
                    weight,
                    created_at
                FROM edges
                """
            ):
                graph.add_edge(
                    row[0],
                    row[1],
                    frequency=row[2],
                    weight=row[3],
                    created_at=self._deserialize_datetime(row[4]),
                )

            logger.info(
                "Loaded graph from SQLite %s nodes=%d edges=%d",
                self.filepath,
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )
            return graph
        finally:
            connection.close()

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                text TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                embedding TEXT,
                frequency INTEGER NOT NULL,
                first_seen TEXT,
                last_seen TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                weight REAL NOT NULL,
                created_at TEXT,
                PRIMARY KEY (src, dst),
                FOREIGN KEY (src) REFERENCES nodes(text) ON DELETE CASCADE,
                FOREIGN KEY (dst) REFERENCES nodes(text) ON DELETE CASCADE
            )
            """
        )

        schema_name = self._get_meta_value(connection, "schema_name")
        schema_version = self._get_meta_value(connection, "schema_version")

        if schema_name is None and schema_version is None:
            connection.executemany(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (
                    ("schema_name", SQLITE_SCHEMA_NAME),
                    ("schema_version", str(SQLITE_SCHEMA_VERSION)),
                ),
            )
            connection.commit()
            return

        if schema_name != SQLITE_SCHEMA_NAME:
            raise ValueError(f"Unsupported SQLite schema name: {schema_name!r}")

        if schema_version != str(SQLITE_SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported SQLite schema version: {schema_version!r}"
            )

    def _get_meta_value(
        self,
        connection: sqlite3.Connection,
        key: str,
    ) -> str | None:
        row = connection.execute(
            "SELECT value FROM meta WHERE key = ?",
            (key,),
        ).fetchone()
        return None if row is None else str(row[0])

    def _serialize_embedding(self, embedding) -> str | None:
        if embedding is None:
            return None

        if isinstance(embedding, np.ndarray):
            return json.dumps(embedding.tolist())

        return json.dumps(list(embedding))

    def _deserialize_embedding(self, embedding: str | None):
        if embedding is None:
            return None

        return np.array(json.loads(embedding), dtype=np.float32)

    def _serialize_datetime(self, value: datetime | None) -> str | None:
        return None if value is None else value.isoformat()

    def _deserialize_datetime(self, value: str | None) -> datetime | None:
        return None if value is None else datetime.fromisoformat(value)
