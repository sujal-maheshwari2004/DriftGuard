from __future__ import annotations

from driftguard.guard import DriftGuard


def make_langgraph_review_node(
    guard: DriftGuard,
    *,
    action_key: str = "candidate_action",
    acknowledged_key: str | None = None,
    policy: str | None = None,
    min_confidence: float | None = None,
    review_key: str = "guard_review",
    warnings_key: str = "guard_warnings_count",
    confidence_key: str = "guard_confidence",
    top_warning_key: str = "guard_top_warning",
):
    def review_node(state: dict) -> dict:
        acknowledged = (
            bool(state.get(acknowledged_key, False))
            if acknowledged_key is not None
            else False
        )
        review = guard.before_step(
            state[action_key],
            policy=policy,
            min_confidence=min_confidence,
            acknowledged=acknowledged,
        )
        top_warning = review.warnings[0] if review.warnings else None
        return {
            review_key: review,
            warnings_key: len(review.warnings),
            confidence_key: review.confidence,
            top_warning_key: (
                {
                    "trigger": top_warning.trigger,
                    "risk": top_warning.risk,
                    "confidence": top_warning.confidence,
                }
                if top_warning is not None
                else None
            ),
        }

    return review_node
