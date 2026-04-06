from dataclasses import dataclass
from datetime import datetime, UTC


@dataclass
class Event:
    """
    Represents one causal learning unit.
    Example:
    action:
        "increase salt"
    feedback:
        "too salty"
    outcome:
        "dish ruined"
    """

    action: str
    feedback: str
    outcome: str
    timestamp: datetime = datetime.now(UTC)
    confidence: float = 1.0