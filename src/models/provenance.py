from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .extracted_document import BoundingBox


class ProvenanceSpan(BaseModel):
    """Single provenance citation back to the source document."""

    doc_name: str
    doc_id: str
    page_number: int = Field(..., ge=1)
    bbox: Optional[BoundingBox] = None
    content_hash: Optional[str] = None
    snippet: Optional[str] = Field(
        None, description="Short excerpt of the supporting text/content."
    )


class ProvenanceChain(BaseModel):
    """Ordered chain of provenance spans backing an answer or claim."""

    spans: List[ProvenanceSpan] = Field(default_factory=list)

