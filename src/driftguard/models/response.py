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
class RetrievalResponse:
    query: str
    warnings: List[Warning]
    chains: List[List[str]]
    confidence: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
