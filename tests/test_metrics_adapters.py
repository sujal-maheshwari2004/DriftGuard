from dataclasses import dataclass

from driftguard.adapters.generic import review_payload
from driftguard.adapters.langgraph import make_langgraph_review_node
from driftguard.guard import DriftGuard
from driftguard.metrics import DriftGuardMetrics


@dataclass
class FakeWarning:
    trigger: str
    risk: str
    confidence: float
    frequency: int = 1


@dataclass
class FakeResponse:
    query: str
    warnings: list[FakeWarning]
    chains: list[list[str]]
    confidence: float


class FakeRuntime:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.metrics = DriftGuardMetrics()
        self.query_calls = []

    def query_memory(self, context: str):
        self.query_calls.append(context)
        return self.response


def test_metrics_snapshot_tracks_review_and_prune_counters():
    metrics = DriftGuardMetrics()

    metrics.record_review(warnings_count=2, confidence=0.75)
    metrics.record_review(warnings_count=1, confidence=0.9, blocked=True)
    metrics.record_review(
        warnings_count=1,
        confidence=0.85,
        acknowledgement_required=True,
    )
    metrics.record_review(warnings_count=0, confidence=0.0, skipped=True)
    metrics.record_storage()
    metrics.record_prune(nodes_removed=2, edges_removed=3)

    snapshot = metrics.snapshot_dict()

    assert snapshot["counters"] == {
        "reviews_total": 3,
        "review_warnings_total": 4,
        "review_confidence_samples_total": 3,
        "reviews_blocked_total": 1,
        "reviews_ack_required_total": 1,
        "reviews_skipped_total": 1,
        "records_total": 1,
        "prune_runs_total": 1,
        "prune_nodes_removed_total": 2,
        "prune_edges_removed_total": 3,
    }
    assert snapshot["gauges"]["last_review_confidence"] == 0.85
    assert snapshot["gauges"]["review_confidence_average"] > 0.8


def test_review_payload_returns_review_metadata():
    runtime = FakeRuntime(
        FakeResponse(
            query="increase salt",
            warnings=[FakeWarning("increase salt", "too salty", 0.86)],
            chains=[["increase salt", "too salty", "dish ruined"]],
            confidence=0.86,
        )
    )
    guard = DriftGuard(runtime=runtime)

    result = review_payload(
        guard,
        {"action": "increase salt", "attempt": 2},
    )

    assert result["payload"] == {"action": "increase salt", "attempt": 2}
    assert result["warnings_count"] == 1
    assert result["confidence"] == 0.86
    assert result["review"].warnings[0].risk == "too salty"
    assert runtime.query_calls == ["increase salt"]


def test_make_langgraph_review_node_maps_review_into_state():
    runtime = FakeRuntime(
        FakeResponse(
            query="increase salt",
            warnings=[FakeWarning("increase salt", "too salty", 0.91)],
            chains=[["increase salt", "too salty", "dish ruined"]],
            confidence=0.91,
        )
    )
    guard = DriftGuard(runtime=runtime)
    review_node = make_langgraph_review_node(guard)

    state_update = review_node({"candidate_action": "increase salt"})

    assert state_update["guard_warnings_count"] == 1
    assert state_update["guard_confidence"] == 0.91
    assert state_update["guard_top_warning"] == {
        "trigger": "increase salt",
        "risk": "too salty",
        "confidence": 0.91,
    }
    assert state_update["guard_review"].query == "increase salt"
    assert runtime.query_calls == ["increase salt"]
