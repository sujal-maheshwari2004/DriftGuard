from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MetricsSnapshot:
    counters: Mapping[str, int]
    gauges: Mapping[str, float]


class DriftGuardMetrics:
    def __init__(self):
        self._counters: Counter[str] = Counter()
        self._gauges: dict[str, float] = {}

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = float(value)

    def record_review(
        self,
        *,
        warnings_count: int,
        confidence: float,
        blocked: bool = False,
        acknowledgement_required: bool = False,
        skipped: bool = False,
    ) -> None:
        if skipped:
            self.increment("reviews_skipped_total")
            return

        self.increment("reviews_total")
        self.increment("review_warnings_total", warnings_count)
        self.increment("review_confidence_samples_total")
        self.set_gauge("last_review_confidence", confidence)

        total_confidence = self._gauges.get("review_confidence_total", 0.0) + confidence
        self.set_gauge("review_confidence_total", total_confidence)
        self.set_gauge(
            "review_confidence_average",
            total_confidence / max(1, self._counters["review_confidence_samples_total"]),
        )

        if blocked:
            self.increment("reviews_blocked_total")

        if acknowledgement_required:
            self.increment("reviews_ack_required_total")

    def record_storage(self) -> None:
        self.increment("records_total")

    def record_node_created(self) -> None:
        self.increment("nodes_created_total")

    def record_node_merged(self) -> None:
        self.increment("nodes_merged_total")

    def record_edge_created(self) -> None:
        self.increment("edges_created_total")

    def record_edge_reused(self) -> None:
        self.increment("edges_reused_total")

    def record_prune(self, *, nodes_removed: int, edges_removed: int) -> None:
        self.increment("prune_runs_total")
        self.increment("prune_nodes_removed_total", nodes_removed)
        self.increment("prune_edges_removed_total", edges_removed)

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            counters=dict(self._counters),
            gauges=dict(self._gauges),
        )

    def snapshot_dict(self) -> dict[str, dict]:
        snapshot = self.snapshot()
        return {
            "counters": dict(snapshot.counters),
            "gauges": dict(snapshot.gauges),
        }
