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
from driftguard.storage.sqlite_persistence import SQLitePersistence


logger = get_logger(__name__)


@dataclass
class DriftGuardRuntime:
    settings: DriftGuardSettings
    merge_engine: MergeEngine
    prune_engine: PruneEngine
    persistence: GraphPersistence
    graph_store: GraphStore
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

    def query_memory(self, context: str):
        logger.info("Querying memory for context=%r", context)
        response = self.retrieval_engine.query(context)
        logger.info(
            "Query complete context=%r warnings=%d chains=%d confidence=%.2f",
            context,
            len(response.warnings),
            len(response.chains),
            response.confidence,
        )
        return response

    def deep_prune(self) -> dict:
        before = self.graph_store.stats()
        logger.info("Starting deep prune with stats=%s", before)
        prune_summary = self.prune_engine.deep_prune(self.graph_store.graph)
        self.graph_store.save()
        after = self.graph_store.stats()
        self.metrics.record_prune(
            nodes_removed=max(0, before["nodes"] - after["nodes"]),
            edges_removed=max(0, before["edges"] - after["edges"]),
        )
        logger.info("Deep prune finished with stats=%s", after)
        return {
            "status": "pruned",
            "before": before,
            "after": after,
            "details": prune_summary,
        }

    def graph_stats(self) -> dict:
        stats = self.graph_store.stats()
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

    graph_store = GraphStore(
        merge_engine=merge_engine,
        prune_engine=prune_engine,
        persistence_engine=persistence,
        metrics=metrics,
        traversal_max_depth=settings.traversal_max_depth,
        traversal_max_branching=settings.traversal_max_branching,
        traversal_max_paths=settings.traversal_max_paths,
    )
    retrieval_engine = RetrievalEngine(
        graph_store,
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
        graph_store=graph_store,
        retrieval_engine=retrieval_engine,
        metrics=metrics,
    )

    if auto_load:
        runtime.graph_store.load()
        logger.info(
            "Runtime ready with loaded graph stats=%s",
            runtime.graph_store.stats(),
        )
    else:
        logger.info("Runtime ready without auto-loading persisted graph")

    return runtime


def _build_persistence(settings: DriftGuardSettings) -> GraphPersistence:
    if settings.storage_backend == "json":
        return Persistence(filepath=settings.graph_filepath)

    if settings.storage_backend == "sqlite":
        return SQLitePersistence(filepath=settings.sqlite_filepath)

    raise ValueError(
        f"Unsupported storage backend: {settings.storage_backend!r}"
    )
