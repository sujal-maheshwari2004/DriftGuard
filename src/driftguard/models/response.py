from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import List


@dataclass
class Warning:
    trigger: str
    risk: str
    frequency: int
    confidence: float
    # The end of the chain. `risk` is the feedback that followed the trigger,
    # which left the consequence an agent most needs out of the warning.
    outcome: str | None = None


@dataclass
class Reinforcement:
    trigger: str
    recommendation: str
    frequency: int
    confidence: float
    outcome: str | None = None


@dataclass
class RetrievalResponse:
    query: str
    warnings: List[Warning]
    chains: List[List[str]]
    confidence: float
    reinforcements: List[Reinforcement] = field(default_factory=list)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
