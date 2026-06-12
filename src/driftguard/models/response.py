from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import List


@dataclass
class Warning:
    trigger: str
    risk: str
    frequency: int
    confidence: float


@dataclass
class Reinforcement:
    trigger: str
    recommendation: str
    frequency: int
    confidence: float


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
