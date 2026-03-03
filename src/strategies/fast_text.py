from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Dict, List

import pdfplumber

from src.models import BoundingBox, ExtractedDocument, Table, TableCell, TextBlock
from src.models import DocumentProfile
from .base import ExtractionResult, ExtractionStrategy


class FastTextExtractor(ExtractionStrategy):
    """Strategy A – fast text extraction using pdfplumber with confidence scoring."""

    name = "fast_text"

    def extract(self, profile: DocumentProfile) -> ExtractionResult:
        pdf_path = Path(profile.source_path)
        text_blocks: List[TextBlock] = []
        tables: List[Table] = []
        page_confidences: Dict[int, float] = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                width = float(page.width or 1)
                height = float(page.height or 1)
                area = width * height

                extracted_text = page.extract_text() or ""
                # simple per-line blocks with shared bbox: use page bbox for now
                if extracted_text.strip():
                    bbox = BoundingBox(x0=0.0, y0=0.0, x1=width, y1=height)
                    text_blocks.append(
                        TextBlock(
                            page_number=page_idx,
                            bbox=bbox,
                            text=extracted_text,
                            reading_order=len(text_blocks),
                        )
                    )

                # cheap table detection using pdfplumber's extract_table when available
                try:
                    raw_tables = page.extract_tables() or []
                except Exception:
                    raw_tables = []

                for t_idx, raw in enumerate(raw_tables):
                    headers: List[str] = []
                    rows: List[List[TableCell]] = []
                    if not raw:
                        continue
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
                        rows.append(cells_row)

                    tables.append(
                        Table(
                            page_number=page_idx,
                            bbox=None,
                            headers=headers,
                            rows=rows,
                            title=None,
                            caption=None,
                        )
                    )

                # Confidence: density + text presence vs. images
                chars = page.chars or []
                char_count = float(len(chars))
                char_density = char_count / area if area > 0 else 0.0

                images = page.images or []
                image_area = 0.0
                for img in images:
                    iw = float(img.get("width", 0) or 0)
                    ih = float(img.get("height", 0) or 0)
                    image_area += iw * ih
                image_ratio = image_area / area if area > 0 else 0.0

                # heuristic: more characters and fewer images -> higher confidence
                density_score = min(char_density * 500, 1.0)  # scale heuristic
                image_penalty = min(image_ratio * 1.5, 1.0)
                confidence = max(0.0, density_score * (1.0 - 0.5 * image_penalty))

                page_confidences[page_idx] = confidence

        doc = ExtractedDocument(
            doc_id=profile.doc_id,
            num_pages=profile.num_pages,
            text_blocks=text_blocks,
            tables=tables,
            figures=[],
            metadata={"strategy_used": self.name, "profile_origin_type": profile.origin_type.value},
        )

        avg_conf = mean(page_confidences.values()) if page_confidences else 0.0
        return ExtractionResult(
            document=doc,
            page_confidences=page_confidences,
            strategy_name=self.name,
            cost_estimate_usd=0.0,  # local pdfplumber – negligible marginal cost
        )

