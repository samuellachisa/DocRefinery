from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .extracted_document import BoundingBox


class ChunkType(str, Enum):
    paragraph = "paragraph"
    table = "table"
    table_cell_group = "table_cell_group"
    figure = "figure"
    list = "list"
    header = "header"


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

