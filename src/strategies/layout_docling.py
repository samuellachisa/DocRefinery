from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from docling.document_converter import DocumentConverter

from src.models import (
    BoundingBox,
    ExtractedDocument,
    Figure,
    Table,
    TableCell,
    TextBlock,
    DocumentProfile,
)
from .base import ExtractionResult, ExtractionStrategy


class LayoutExtractor(ExtractionStrategy):
    """Strategy B – layout-aware extraction using Docling."""

    name = "layout_docling"

    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def extract(self, profile: DocumentProfile) -> ExtractionResult:
        pdf_path = Path(profile.source_path)

        # Convert to DoclingDocument
        result = self._converter.convert(pdf_path)
        doc = result.document

        text_blocks: List[TextBlock] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        page_confidences: Dict[int, float] = {}

        # DoclingDocument has a layout tree; we use a simplified traversal.
        page_index = 1
        for page in doc.pages:
            # Text blocks
            for block in page.blocks:
                bbox = BoundingBox(
                    x0=block.bbox.x0,
                    y0=block.bbox.y0,
                    x1=block.bbox.x1,
                    y1=block.bbox.y1,
                )
                text_blocks.append(
                    TextBlock(
                        page_number=page_index,
                        bbox=bbox,
                        text=block.text,
                        reading_order=len(text_blocks),
                    )
                )

            # Tables
            for tbl in page.tables:
                headers: List[str] = [cell.text for cell in tbl.header.cells]
                rows: List[List[TableCell]] = []
                for r_idx, row in enumerate(tbl.body.rows):
                    cells_row: List[TableCell] = []
                    for c_idx, cell in enumerate(row.cells):
                        cbbox = None
                        if cell.bbox is not None:
                            cbbox = BoundingBox(
                                x0=cell.bbox.x0,
                                y0=cell.bbox.y0,
                                x1=cell.bbox.x1,
                                y1=cell.bbox.y1,
                            )
                        cells_row.append(
                            TableCell(
                                row_index=r_idx,
                                col_index=c_idx,
                                text=cell.text,
                                bbox=cbbox,
                            )
                        )
                    rows.append(cells_row)

                tbbox = None
                if tbl.bbox is not None:
                    tbbox = BoundingBox(
                        x0=tbl.bbox.x0,
                        y0=tbl.bbox.y0,
                        x1=tbl.bbox.x1,
                        y1=tbl.bbox.y1,
                    )

                tables.append(
                    Table(
                        page_number=page_index,
                        bbox=tbbox,
                        headers=headers,
                        rows=rows,
                        title=getattr(tbl, "title", None),
                        caption=getattr(tbl, "caption", None),
                    )
                )

            # Figures
            for fig in page.figures:
                fbbox = None
                if fig.bbox is not None:
                    fbbox = BoundingBox(
                        x0=fig.bbox.x0,
                        y0=fig.bbox.y0,
                        x1=fig.bbox.x1,
                        y1=fig.bbox.y1,
                    )
                figures.append(
                    Figure(
                        page_number=page_index,
                        bbox=fbbox,
                        caption=getattr(fig, "caption", None),
                        label=getattr(fig, "label", None),
                    )
                )

            # Very rough confidence: presence of both text blocks and tables/figures boosts it
            has_text = bool(page.blocks)
            has_tables = bool(page.tables)
            has_figures = bool(page.figures)
            conf = 0.3 + (0.3 if has_text else 0.0) + (0.2 if has_tables else 0.0) + (
                0.2 if has_figures else 0.0
            )
            page_confidences[page_index] = min(conf, 1.0)

            page_index += 1

        extracted = ExtractedDocument(
            doc_id=profile.doc_id,
            num_pages=profile.num_pages,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            metadata={"strategy_used": self.name},
        )

        avg_conf = (
            sum(page_confidences.values()) / len(page_confidences)
            if page_confidences
            else 0.0
        )

        # Layout models consume some compute; we still treat cost as negligible in this context.
        return ExtractionResult(
            document=extracted,
            page_confidences=page_confidences,
            strategy_name=self.name,
            cost_estimate_usd=0.0,
        )

