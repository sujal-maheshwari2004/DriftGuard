from __future__ import annotations

from driftguard.guard import DriftGuard


def review_payload(
    guard: DriftGuard,
    payload: dict,
    *,
    action_key: str = "action",
    acknowledged_key: str | None = None,
    policy: str | None = None,
    min_confidence: float | None = None,
) -> dict:
    context = payload[action_key]
    acknowledged = (
        bool(payload.get(acknowledged_key, False))
        if acknowledged_key is not None
        else False
    )
    review = guard.before_step(
        context,
        policy=policy,
        min_confidence=min_confidence,
        acknowledged=acknowledged,
    )
    return {
        "payload": payload,
        "review": review,
        "warnings_count": len(review.warnings),
        "confidence": review.confidence,
    }
