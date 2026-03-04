from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .extracted_document import BoundingBox


class ChunkType(str, Enum):
    paragraph = "paragraph"
    table = "table"
    table_cell_group = "table_cell_group"
    figure = "figure"
    list = "list"
    header = "header"
    section_header = "section_header"
    caption = "caption"
    footnote = "footnote"


class LDU(BaseModel):
    """Logical Document Unit – a semantically coherent, self-contained unit."""

    id: str
    doc_id: str
    chunk_type: ChunkType
    content: str

    page_refs: List[int] = Field(
        default_factory=list, description="Pages that this chunk touches."
    )
    bounding_box: Optional[BoundingBox] = Field(
        None, description="Coarse bbox covering the chunk across pages if applicable."
    )

    parent_section: Optional[str] = Field(
        None, description="Section identifier or title this chunk belongs to."
    )
    token_count: int = Field(
        ..., ge=0, description="Approximate token count used for chunking decisions."
    )
    content_hash: str = Field(
        ..., description="Stable hash over spatially anchored content."
    )

    references: List[str] = Field(
        default_factory=list,
        description="Optional cross-references (e.g., 'Table 3', 'Figure 2.1').",
    )
    relationships: List[Dict] = Field(
        default_factory=list,
        description="Optional relationship descriptors to other LDUs or entities.",
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Optional embedding vector; typically populated by the vector store.",
    )

    @field_validator("page_refs")
    @classmethod
    def _validate_page_refs(cls, v: List[int]) -> List[int]:  # type: ignore[override]
        if not v:
            raise ValueError("page_refs must contain at least one page number")
        if any(p < 1 for p in v):
            raise ValueError("page_refs must contain only positive page numbers (>= 1)")
        return v

