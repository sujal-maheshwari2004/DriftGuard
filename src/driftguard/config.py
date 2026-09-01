import os

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class DriftGuardSettings:
    graph_filepath: str = "driftguard_graph.json"
    storage_backend: str = "json"
    sqlite_filepath: str = "driftguard_graph.sqlite3"
    postgres_dsn: str | None = None
    success_graph_filepath: str = "driftguard_success_graph.json"
    success_sqlite_filepath: str = "driftguard_success_graph.sqlite3"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str | None = None
    retrieval_top_k: int = 5
    retrieval_min_similarity: float = 0.60
    retrieval_recency_weight: float = 0.15
    traversal_max_depth: int = 3
    traversal_max_branching: int = 10
    traversal_max_paths: int = 100
    similarity_threshold_action: float = 0.72
    similarity_threshold_feedback: float = 0.70
    similarity_threshold_outcome: float = 0.88
    guard_policy: str = "warn"
    guard_min_confidence: float = 0.0
    prune_node_stale_days: int = 60
    prune_edge_min_frequency: int = 2
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, environ=None) -> "DriftGuardSettings":
        """
        Build settings from DRIFTGUARD_* environment variables.

        Every field maps to DRIFTGUARD_<FIELD_NAME> uppercased, so
        DRIFTGUARD_STORAGE_BACKEND sets storage_backend. This is how the MCP
        server gets configured: `driftguard-mcp` takes no arguments, and an
        MCP client config can only pass environment variables, so without this
        the graph landed in whatever directory the client happened to use and
        the SQLite and Postgres backends were unreachable.

        Unset variables keep their default. A value that will not parse raises
        rather than being silently ignored.
        """

        environ = os.environ if environ is None else environ
        overrides = {}

        for field in fields(cls):
            raw = environ.get(f"DRIFTGUARD_{field.name.upper()}")

            if raw is None:
                continue

            overrides[field.name] = _coerce(field.name, field.type, raw)

        return cls(**overrides)

    def threshold_for(self, node_type: str) -> float:
        return {
            "action": self.similarity_threshold_action,
            "feedback": self.similarity_threshold_feedback,
            "outcome": self.similarity_threshold_outcome,
        }.get(node_type, 0.85)


def _coerce(name: str, annotation, raw: str):
    """
    Turn an environment string into the type the field is annotated with.

    Annotations arrive as strings under `from __future__ import annotations`
    and as real types otherwise, so matching on the text covers both.
    """

    text = annotation if isinstance(annotation, str) else getattr(
        annotation, "__name__", str(annotation)
    )

    if "int" in text:
        converter = int
    elif "float" in text:
        converter = float
    elif "bool" in text:
        converter = _parse_bool
    else:
        return raw

    try:
        return converter(raw)
    except ValueError as exc:
        raise ValueError(
            f"DRIFTGUARD_{name.upper()}={raw!r} is not a valid {text}"
        ) from exc


def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()

    if lowered in {"1", "true", "yes", "on"}:
        return True

    if lowered in {"0", "false", "no", "off"}:
        return False

    raise ValueError(raw)


DEFAULT_SETTINGS = DriftGuardSettings()

# Backwards-compatible module constants for callers that still import them.
SIM_THRESHOLD_ACTION = DEFAULT_SETTINGS.similarity_threshold_action
SIM_THRESHOLD_FEEDBACK = DEFAULT_SETTINGS.similarity_threshold_feedback
SIM_THRESHOLD_OUTCOME = DEFAULT_SETTINGS.similarity_threshold_outcome
