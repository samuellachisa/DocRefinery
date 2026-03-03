from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in PDF coordinate space."""

    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(BaseModel):
    page_number: int = Field(..., ge=1)
    bbox: BoundingBox
    text: str
    reading_order: int = Field(
        ...,
        description="Monotonic index representing reading order within the document.",
    )


class TableCell(BaseModel):
    row_index: int
    col_index: int
    bbox: Optional[BoundingBox] = None
    text: str


class Table(BaseModel):
    page_number: int = Field(..., ge=1)
    bbox: Optional[BoundingBox] = None
    headers: List[str] = Field(
        default_factory=list, description="Header labels in column order."
    )
    rows: List[List[TableCell]] = Field(
        default_factory=list, description="2D array of table cells."
    )
    title: Optional[str] = None
    caption: Optional[str] = None


class Figure(BaseModel):
    page_number: int = Field(..., ge=1)
    bbox: Optional[BoundingBox] = None
    caption: Optional[str] = None
    label: Optional[str] = None


class ExtractedDocument(BaseModel):
    """Normalized representation that all extraction strategies must produce."""

    doc_id: str
    num_pages: int = Field(..., ge=1)
    text_blocks: List[TextBlock] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    figures: List[Figure] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extraction-time metadata (e.g., strategy_used, scores).",
    )

