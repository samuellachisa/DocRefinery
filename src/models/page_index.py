from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SectionNode(BaseModel):
    """Single node in a PageIndex-style section tree."""

    id: str
    title: str

    page_start: int = Field(..., ge=1)
    page_end: int = Field(..., ge=1)

    summary: Optional[str] = Field(
        None, description="Short LLM-generated summary (2–3 sentences)."
    )
    key_entities: List[str] = Field(default_factory=list)
    data_types_present: List[Literal["text", "tables", "figures", "equations"]] = (
        Field(default_factory=list)
    )

    children: List["SectionNode"] = Field(default_factory=list)


class PageIndex(BaseModel):
    """Hierarchical navigation index over a document."""

    doc_id: str
    root: SectionNode

