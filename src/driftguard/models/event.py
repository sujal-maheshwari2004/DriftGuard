from dataclasses import dataclass, field
from datetime import datetime, UTC

from driftguard.errors import EventValidationError

_TEXT_FIELDS = ("action", "feedback", "outcome")


@dataclass
class Event:
    """
    Represents one causal learning unit.

    Example:
        action:   "increase salt"
        feedback: "too salty"
        outcome:  "dish ruined"
    """

    action: str
    feedback: str
    outcome: str
    confidence: float = 1.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self):
        """
        Reject bad input here rather than deep inside normalization.

        Without this, `record(action=None, ...)` surfaced as an AttributeError
        raised by str.lower(), which callers could not catch as a DriftGuard
        failure and which fired only after the action node had already been
        written to the graph.
        """

        for name in _TEXT_FIELDS:
            value = getattr(self, name)

            if not isinstance(value, str):
                raise EventValidationError(
                    f"Event {name} must be a string, got {type(value).__name__}"
                )

            if not value.strip():
                raise EventValidationError(f"Event {name} must not be empty")