from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from src.models import DocumentProfile, ExtractedDocument


@dataclass
class ExtractionResult:
    document: ExtractedDocument
    page_confidences: Dict[int, float]
    strategy_name: str
    cost_estimate_usd: float


class ExtractionStrategy(ABC):
    """Common interface for all extraction strategies."""

    name: str

    @abstractmethod
    def extract(self, profile: DocumentProfile) -> ExtractionResult:
        """Run extraction for the given document profile."""

