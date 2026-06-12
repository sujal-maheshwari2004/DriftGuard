from dataclasses import dataclass

from driftguard.config import DEFAULT_SETTINGS, DriftGuardSettings
from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine
from driftguard.graph.prune_engine import PruneEngine
from driftguard.logging_config import get_logger
from driftguard.metrics import DriftGuardMetrics
from driftguard.models.event import Event
from driftguard.retrieval.retrieval_engine import RetrievalEngine
from driftguard.storage.base import GraphPersistence
from driftguard.storage.persistence import Persistence
from driftguard.storage.postgres_persistence import PostgresPersistence
from driftguard.storage.sqlite_persistence import SQLitePersistence


logger = get_logger(__name__)


@dataclass
class DriftGuardRuntime:
    settings: DriftGuardSettings
    merge_engine: MergeEngine
    prune_engine: PruneEngine
    persistence: GraphPersistence
    success_persistence: GraphPersistence
    graph_store: GraphStore
    success_graph_store: GraphStore
    retrieval_engine: RetrievalEngine
    metrics: DriftGuardMetrics

    def register_mistake(
        self,
        action: str,
        feedback: str,
        outcome: str,
    ) -> dict:
        logger.info(
            "Registering mistake action=%r feedback=%r outcome=%r",
            action,
            feedback,
            outcome,
        )

        event = Event(
            action=action,
            feedback=feedback,
            outcome=outcome,
        )

        self.graph_store.add_event(event)
        self.graph_store.save()
        self.metrics.record_storage()

        response = {
            "status": "stored",
            "action": action,
            "feedback": feedback,
            "outcome": outcome,
        }
        logger.info(
            "Mistake stored successfully; stats=%s",
            self.graph_store.stats(),
        )
        return response

    def register_success(
        self,
        action: str,
        feedback: str,
        outcome: str,
    ) -> dict:
        logger.info(
            "Registering success action=%r feedback=%r outcome=%r",
            action,
            feedback,
            outcome,
        )

        event = Event(
            action=action,
            feedback=feedback,
            outcome=outcome,
        )

        self.success_graph_store.add_event(event)
        self.success_graph_store.save()
        self.metrics.record_storage()

        response = {
            "status": "stored",
            "action": action,
            "feedback": feedback,
            "outcome": outcome,
        }
        logger.info(
            "Success stored successfully; stats=%s",
            self.success_graph_store.stats(),
        )
        return response

    def query_memory(self, context: str):
        logger.info("Querying memory for context=%r", context)
        response = self.retrieval_engine.query(context)
        logger.info(
            "Query complete context=%r warnings=%d reinforcements=%d chains=%d confidence=%.2f",
            context,
            len(response.warnings),
            len(response.reinforcements),
            len(response.chains),
            response.confidence,
        )
        return response

    def deep_prune(self) -> dict:
        mistakes_before = self.graph_store.stats()
        successes_before = self.success_graph_store.stats()
        logger.info(
            "Starting deep prune with mistakes=%s successes=%s",
            mistakes_before,
            successes_before,
        )

        mistakes_summary = self.prune_engine.deep_prune(self.graph_store.graph)
        successes_summary = self.prune_engine.deep_prune(self.success_graph_store.graph)

        self.graph_store.save()
        self.success_graph_store.save()

        mistakes_after = self.graph_store.stats()
        successes_after = self.success_graph_store.stats()

        self.metrics.record_prune(
            nodes_removed=max(0, mistakes_before["nodes"] - mistakes_after["nodes"])
            + max(0, successes_before["nodes"] - successes_after["nodes"]),
            edges_removed=max(0, mistakes_before["edges"] - mistakes_after["edges"])
            + max(0, successes_before["edges"] - successes_after["edges"]),
        )

        logger.info(
            "Deep prune finished with mistakes=%s successes=%s",
            mistakes_after,
            successes_after,
        )
        return {
            "status": "pruned",
            "mistakes": {
                "before": mistakes_before,
                "after": mistakes_after,
                "details": mistakes_summary,
            },
            "successes": {
                "before": successes_before,
                "after": successes_after,
                "details": successes_summary,
            },
        }

    def graph_stats(self) -> dict:
        stats = {
            "mistakes": self.graph_store.stats(),
            "successes": self.success_graph_store.stats(),
        }
        logger.debug("Graph stats requested: %s", stats)
        return stats

    def metrics_snapshot(self) -> dict:
        snapshot = self.metrics.snapshot_dict()
        logger.debug("Metrics snapshot requested: %s", snapshot)
        return snapshot


def build_runtime(
    *,
    settings: DriftGuardSettings | None = None,
    merge_engine: MergeEngine | None = None,
    prune_engine: PruneEngine | None = None,
    persistence: GraphPersistence | None = None,
    success_persistence: GraphPersistence | None = None,
    metrics: DriftGuardMetrics | None = None,
    auto_load: bool = True,
) -> DriftGuardRuntime:
    logger.info("Building DriftGuard runtime")

    settings = settings or DEFAULT_SETTINGS

    merge_engine = merge_engine or MergeEngine(settings=settings)
    metrics = metrics or DriftGuardMetrics()
    prune_engine = prune_engine or PruneEngine(
        node_stale_days=settings.prune_node_stale_days,
        edge_min_frequency=settings.prune_edge_min_frequency,
    )
    persistence = persistence or _build_persistence(settings)
    success_persistence = success_persistence or _build_persistence(settings, success=True)

    graph_store = GraphStore(
        merge_engine=merge_engine,
        prune_engine=prune_engine,
        persistence_engine=persistence,
        metrics=metrics,
        traversal_max_depth=settings.traversal_max_depth,
        traversal_max_branching=settings.traversal_max_branching,
        traversal_max_paths=settings.traversal_max_paths,
    )
    success_graph_store = GraphStore(
        merge_engine=merge_engine,
        prune_engine=prune_engine,
        persistence_engine=success_persistence,
        metrics=metrics,
        traversal_max_depth=settings.traversal_max_depth,
        traversal_max_branching=settings.traversal_max_branching,
        traversal_max_paths=settings.traversal_max_paths,
    )
    retrieval_engine = RetrievalEngine(
        graph_store,
        success_graph_store,
        top_k=settings.retrieval_top_k,
        min_similarity=settings.retrieval_min_similarity,
        recency_weight=settings.retrieval_recency_weight,
        metrics=metrics,
    )

    runtime = DriftGuardRuntime(
        settings=settings,
        merge_engine=merge_engine,
        prune_engine=prune_engine,
        persistence=persistence,
        success_persistence=success_persistence,
        graph_store=graph_store,
        success_graph_store=success_graph_store,
        retrieval_engine=retrieval_engine,
        metrics=metrics,
    )

    if auto_load:
        runtime.graph_store.load()
        runtime.success_graph_store.load()
        logger.info(
            "Runtime ready with loaded graphs mistakes=%s successes=%s",
            runtime.graph_store.stats(),
            runtime.success_graph_store.stats(),
        )
    else:
        logger.info("Runtime ready without auto-loading persisted graphs")

    return runtime


def _build_persistence(
    settings: DriftGuardSettings, *, success: bool = False
) -> GraphPersistence:
    if settings.storage_backend == "json":
        filepath = settings.success_graph_filepath if success else settings.graph_filepath
        return Persistence(filepath=filepath)

    if settings.storage_backend == "sqlite":
        filepath = (
            settings.success_sqlite_filepath if success else settings.sqlite_filepath
        )
        return SQLitePersistence(filepath=filepath)

    if settings.storage_backend == "postgres":
        table_prefix = "success_" if success else ""
        return PostgresPersistence(dsn=settings.postgres_dsn, table_prefix=table_prefix)

    raise ValueError(
        f"Unsupported storage backend: {settings.storage_backend!r}"
    )
