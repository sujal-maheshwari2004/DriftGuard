from dataclasses import dataclass
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
    timestamp: datetime = datetime.now(UTC)