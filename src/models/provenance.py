from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

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

    @model_validator(mode="after")
    def _validate_provenance(self) -> "ProvenanceSpan":
        # When no bounding box is available, we strongly prefer having a stable content hash.
        if self.bbox is None and not self.content_hash:
            raise ValueError(
                "ProvenanceSpan without bbox must include a non-empty content_hash"
            )
        return self


class ProvenanceChain(BaseModel):
    """Ordered chain of provenance spans backing an answer or claim."""

    spans: List[ProvenanceSpan] = Field(default_factory=list)

