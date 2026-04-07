from dataclasses import dataclass

import pytest

from driftguard.config import DriftGuardSettings
from driftguard.guard import (
    DriftGuard,
    GuardrailAcknowledgementRequired,
    GuardrailTriggered,
    guard_step,
)
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
        self.record_calls = []
        self.query_calls = []
        self.prune_calls = 0
        self.stats_calls = 0

    def query_memory(self, context: str):
        self.query_calls.append(context)
        return self.response

    def register_mistake(self, action: str, feedback: str, outcome: str):
        payload = {
            "status": "stored",
            "action": action,
            "feedback": feedback,
            "outcome": outcome,
        }
        self.record_calls.append(payload)
        return payload

    def deep_prune(self):
        self.prune_calls += 1
        return {"status": "pruned"}

    def graph_stats(self):
        self.stats_calls += 1
        return {"nodes": 3, "edges": 2}


@pytest.fixture
def warning_response():
    return FakeResponse(
        query="increase salt",
        warnings=[
            FakeWarning(
                trigger="increase salt",
                risk="too salty",
                confidence=0.85,
            )
        ],
        chains=[["increase salt", "too salty", "dish ruined"]],
        confidence=0.85,
    )


def test_driftguard_delegates_runtime_calls(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    review = guard.review("increase salt")
    record = guard.record("increase salt", "too salty", "dish ruined")
    prune = guard.prune()
    stats = guard.stats()

    assert review is warning_response
    assert runtime.query_calls == ["increase salt"]
    assert record["status"] == "stored"
    assert runtime.record_calls[0]["outcome"] == "dish ruined"
    assert prune == {"status": "pruned"}
    assert runtime.prune_calls == 1
    assert stats == {"nodes": 3, "edges": 2}
    assert runtime.stats_calls == 1


def test_before_step_returns_review_when_not_blocking(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    review = guard.before_step("increase salt")

    assert review is warning_response
    assert runtime.query_calls == ["increase salt"]


def test_before_step_blocks_when_threshold_is_met(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    with pytest.raises(GuardrailTriggered) as exc:
        guard.before_step(
            "increase salt",
            min_confidence=0.8,
            raise_on_match=True,
        )

    assert "increase salt" in str(exc.value)
    assert "too salty" in str(exc.value)
    assert runtime.metrics.snapshot_dict()["counters"]["reviews_blocked_total"] == 1


def test_before_step_does_not_block_below_threshold(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    review = guard.before_step(
        "increase salt",
        min_confidence=0.95,
        raise_on_match=True,
    )

    assert review is warning_response


def test_before_step_requires_acknowledgement_when_policy_demands_it(
    warning_response,
):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    with pytest.raises(GuardrailAcknowledgementRequired) as exc:
        guard.before_step(
            "increase salt",
            min_confidence=0.8,
            policy="acknowledge",
        )

    assert "increase salt" in str(exc.value)
    assert "too salty" in str(exc.value)
    assert (
        runtime.metrics.snapshot_dict()["counters"]["reviews_ack_required_total"] == 1
    )


def test_before_step_allows_acknowledged_warning_to_continue(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    review = guard.before_step(
        "increase salt",
        min_confidence=0.8,
        policy="acknowledge",
        acknowledged=True,
    )

    assert review is warning_response


def test_before_step_record_only_skips_runtime_review(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    review = guard.before_step("increase salt", policy="record_only")

    assert review.query == "increase salt"
    assert review.warnings == []
    assert review.confidence == 0.0
    assert runtime.query_calls == []
    assert runtime.metrics.snapshot_dict()["counters"]["reviews_skipped_total"] == 1


def test_before_step_uses_settings_defaults_for_policy_and_threshold(
    warning_response,
):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(
        runtime=runtime,
        settings=DriftGuardSettings(
            guard_policy="block",
            guard_min_confidence=0.8,
        ),
    )

    with pytest.raises(GuardrailTriggered):
        guard.before_step("increase salt")


def test_guard_step_uses_default_string_argument_context(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)
    seen = []

    @guard_step(guard, on_review=seen.append)
    def agent_step(task: str):
        return f"executed: {task}"

    result = agent_step("increase salt")

    assert result == "executed: increase salt"
    assert runtime.query_calls == ["increase salt"]
    assert seen == [warning_response]


def test_guard_step_supports_custom_input_getter(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    @guard_step(
        guard,
        input_getter=lambda payload: payload["next_action"],
    )
    def agent_step(payload: dict):
        return payload["next_action"]

    result = agent_step({"next_action": "increase salt"})

    assert result == "increase salt"
    assert runtime.query_calls == ["increase salt"]


def test_guard_step_supports_acknowledgement_getter(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    @guard_step(
        guard,
        input_getter=lambda payload: payload["next_action"],
        acknowledged_getter=lambda payload: payload["acknowledged"],
        policy="acknowledge",
    )
    def agent_step(payload: dict):
        return payload["next_action"]

    with pytest.raises(GuardrailAcknowledgementRequired):
        agent_step({"next_action": "increase salt", "acknowledged": False})

    result = agent_step({"next_action": "increase salt", "acknowledged": True})

    assert result == "increase salt"


def test_guard_step_raises_when_context_cannot_be_derived(warning_response):
    runtime = FakeRuntime(response=warning_response)
    guard = DriftGuard(runtime=runtime)

    @guard_step(guard)
    def agent_step(payload: dict):
        return payload

    with pytest.raises(ValueError) as exc:
        agent_step({"next_action": "increase salt"})

    assert "input_getter" in str(exc.value)
