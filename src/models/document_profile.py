from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, computed_field


class OriginType(str, Enum):
    native_digital = "native_digital"
    scanned_image = "scanned_image"
    mixed = "mixed"
    form_fillable = "form_fillable"


class LayoutComplexity(str, Enum):
    single_column = "single_column"
    multi_column = "multi_column"
    table_heavy = "table_heavy"
    figure_heavy = "figure_heavy"
    mixed = "mixed"


class DomainHint(str, Enum):
    financial = "financial"
    legal = "legal"
    technical = "technical"
    medical = "medical"
    general = "general"


class EstimatedExtractionCost(str, Enum):
    fast_text_sufficient = "fast_text_sufficient"
    needs_layout_model = "needs_layout_model"
    needs_vision_model = "needs_vision_model"


class LanguageProfile(BaseModel):
    code: str = Field(..., description="BCP-47 language code, e.g. 'en', 'am', 'fr'.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detector confidence in [0, 1]."
    )


class DocumentProfile(BaseModel):
    """Profile that governs extraction strategy selection for a document."""

    doc_id: str = Field(
        ..., description="Stable identifier for the document (e.g., content hash)."
    )
    source_path: Path = Field(
        ..., description="Original path of the document when profiled."
    )
    num_pages: int = Field(..., ge=1, description="Total number of pages.")

    origin_type: OriginType
    layout_complexity: LayoutComplexity
    language: LanguageProfile
    domain_hint: DomainHint
    estimated_extraction_cost: EstimatedExtractionCost

    avg_char_density: float = Field(
        ...,
        description="Average characters per unit page area across pages, used for digital vs. scanned detection.",
    )
    avg_image_area_ratio: float = Field(
        ...,
        description="Average fraction of page area occupied by images (0-1).",
    )

    # Additional empirical signals expected by the rubric
    has_ocr_layer: bool = Field(
        ...,
        description="Heuristic flag indicating presence of an OCR/text layer on most pages.",
    )
    table_count_estimate: int = Field(
        ...,
        ge=0,
        description="Approximate number of tables detected across the document.",
    )

    # Triage-level confidence; kept for backward compatibility.
    triage_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in this profile in [0,1].",
    )

    notes: Optional[str] = Field(
        None, description="Optional freeform notes about unusual layout or heuristics."
    )

    class Config:
        frozen = True

    @computed_field  # type: ignore[misc]
    @property
    def confidence_score(self) -> float:
        """Alias for rubric's confidence_score field."""
        return self.triage_confidence

    @computed_field  # type: ignore[misc]
    @property
    def extraction_strategy_needed(
        self,
    ) -> Literal["fast_text", "layout_aware", "vision"]:
        """Map internal EstimatedExtractionCost enum to rubric-friendly label."""
        if self.estimated_extraction_cost is EstimatedExtractionCost.fast_text_sufficient:
            return "fast_text"
        if self.estimated_extraction_cost is EstimatedExtractionCost.needs_layout_model:
            return "layout_aware"
        return "vision"

