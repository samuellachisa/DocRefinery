from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pdfplumber
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


def _infer_page_number(item: object, fallback: int = 1) -> int:
    """
    Best-effort helper to infer a 1-based page number from a Docling item.

    Docling's DocItem types typically expose page information via a provenance
    object (e.g. `item.prov.page_no`, 1-based). We defensively check both the
    item and any `prov` attribute, and fall back to 1 if nothing is available.
    """
    def _normalize(v: object) -> int | None:
        if not isinstance(v, int):
            return None
        # Treat positive ints as already 1-based; 0 as a 0-based sentinel.
        if v >= 1:
            return v
        if v == 0:
            return 1
        return None

    # First, check direct attributes on the item.
    for attr in ("page_no", "page", "page_num", "page_index", "page_idx"):
        value = getattr(item, attr, None)
        norm = _normalize(value)
        if norm is not None:
            return norm

    # Then, check a nested provenance object if present (Docling v2 pattern).
    prov = getattr(item, "prov", None)
    if prov is not None:
        # Docling typically exposes prov as a list of ProvenanceItem.
        prov_items = prov if isinstance(prov, (list, tuple)) else [prov]
        for p in prov_items:
            for attr in ("page_no", "page", "page_num", "page_index", "page_idx"):
                value = getattr(p, attr, None)
                norm = _normalize(value)
                if norm is not None:
                    return norm

    return fallback


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

        # Docling v2 exposes top-level lists: texts, tables, pictures.
        # We rely on those rather than per-page containers to stay API-stable.

        # Texts -> TextBlock LDUs with bounding boxes and reading order.
        for idx, txt_item in enumerate(getattr(doc, "texts", []) or []):
            text = getattr(txt_item, "text", "") or ""
            if not text.strip():
                continue
            bbox_obj = getattr(txt_item, "bbox", None)
            if bbox_obj is not None:
                bbox = BoundingBox(
                    x0=float(getattr(bbox_obj, "x0", 0.0)),
                    y0=float(getattr(bbox_obj, "y0", 0.0)),
                    x1=float(getattr(bbox_obj, "x1", 0.0)),
                    y1=float(getattr(bbox_obj, "y1", 0.0)),
                )
            else:
                bbox = BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)

            page_number = _infer_page_number(txt_item)
            text_blocks.append(
                TextBlock(
                    page_number=page_number,
                    bbox=bbox,
                    text=text,
                    reading_order=idx,
                )
            )
            page_confidences[page_number] = max(page_confidences.get(page_number, 0.0), 0.6)

        # Tables -> structured Table objects.
        for tbl in getattr(doc, "tables", []) or []:
            headers: List[str] = []
            rows: List[List[TableCell]] = []

            header = getattr(tbl, "header", None)
            if header is not None and getattr(header, "cells", None):
                headers = [str(getattr(c, "text", "") or "") for c in header.cells]

            body = getattr(tbl, "body", None)
            body_rows = getattr(body, "rows", []) if body is not None else []
            for r_idx, row in enumerate(body_rows):
                cells_row: List[TableCell] = []
                for c_idx, cell in enumerate(getattr(row, "cells", []) or []):
                    cbbox_obj = getattr(cell, "bbox", None)
                    cbbox = None
                    if cbbox_obj is not None:
                        cbbox = BoundingBox(
                            x0=float(getattr(cbbox_obj, "x0", 0.0)),
                            y0=float(getattr(cbbox_obj, "y0", 0.0)),
                            x1=float(getattr(cbbox_obj, "x1", 0.0)),
                            y1=float(getattr(cbbox_obj, "y1", 0.0)),
                        )
                    cells_row.append(
                        TableCell(
                            row_index=r_idx,
                            col_index=c_idx,
                            text=str(getattr(cell, "text", "") or ""),
                            bbox=cbbox,
                        )
                    )
                rows.append(cells_row)

            tbbox_obj = getattr(tbl, "bbox", None)
            tbbox = None
            if tbbox_obj is not None:
                tbbox = BoundingBox(
                    x0=float(getattr(tbbox_obj, "x0", 0.0)),
                    y0=float(getattr(tbbox_obj, "y0", 0.0)),
                    x1=float(getattr(tbbox_obj, "x1", 0.0)),
                    y1=float(getattr(tbbox_obj, "y1", 0.0)),
                )

            page_number = _infer_page_number(tbl)
            tables.append(
                Table(
                    page_number=page_number,
                    bbox=tbbox,
                    headers=headers,
                    rows=rows,
                    title=getattr(tbl, "title", None),
                    caption=getattr(tbl, "caption", None),
                )
            )
            page_confidences[page_number] = max(page_confidences.get(page_number, 0.0), 0.7)

        # Pictures -> Figure objects, with captions when present.
        for pic in getattr(doc, "pictures", []) or []:
            fbbox_obj = getattr(pic, "bbox", None)
            fbbox = None
            if fbbox_obj is not None:
                fbbox = BoundingBox(
                    x0=float(getattr(fbbox_obj, "x0", 0.0)),
                    y0=float(getattr(fbbox_obj, "y0", 0.0)),
                    x1=float(getattr(fbbox_obj, "x1", 0.0)),
                    y1=float(getattr(fbbox_obj, "y1", 0.0)),
                )

            page_number = _infer_page_number(pic)
            caption = getattr(pic, "caption", None) or getattr(pic, "alt_text", None)
            figures.append(
                Figure(
                    page_number=page_number,
                    bbox=fbbox,
                    caption=caption,
                    label=getattr(pic, "label", None),
                )
            )
            page_confidences[page_number] = max(page_confidences.get(page_number, 0.0), 0.5)

        # Fallback: if Docling produced no usable table structure, try a pdfplumber pass.
        if tables and all((not t.headers and not t.rows) for t in tables):
            tables = []

        if not tables:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    try:
                        raw_tables = page.extract_tables() or []
                    except Exception:
                        raw_tables = []

                    for raw in raw_tables:
                        if not raw:
                            continue

                        headers: List[str] = []
                        rows_cells: List[List[TableCell]] = []

                        if raw[0]:
                            headers = [str(h) if h is not None else "" for h in raw[0]]
                            data_rows = raw[1:]
                        else:
                            data_rows = raw

                        for r_idx, row in enumerate(data_rows):
                            cells_row: List[TableCell] = []
                            for c_idx, cell_text in enumerate(row):
                                text = str(cell_text) if cell_text is not None else ""
                                cells_row.append(
                                    TableCell(
                                        row_index=r_idx,
                                        col_index=c_idx,
                                        text=text,
                                        bbox=None,
                                    )
                                )
                            rows_cells.append(cells_row)

                        tables.append(
                            Table(
                                page_number=page_idx,
                                bbox=None,
                                headers=headers,
                                rows=rows_cells,
                                title=None,
                                caption=None,
                            )
                        )
                        page_confidences[page_idx] = max(
                            page_confidences.get(page_idx, 0.0), 0.7
                        )

        extracted = ExtractedDocument(
            doc_id=profile.doc_id,
            num_pages=profile.num_pages,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            metadata={"strategy_used": self.name},
        )

        # Layout models consume some compute; we still treat cost as negligible in this context.
        return ExtractionResult(
            document=extracted,
            page_confidences=page_confidences,
            strategy_name=self.name,
            cost_estimate_usd=0.0,
        )

