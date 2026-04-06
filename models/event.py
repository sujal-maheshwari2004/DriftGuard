from dataclasses import dataclass, field
from datetime import datetime, UTC


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