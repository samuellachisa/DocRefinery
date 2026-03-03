from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber
import yaml

from src.llm import LLMConfig, call_vision_llm
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


@dataclass
class VisionConfig:
    max_cost_usd_per_document: float
    estimated_tokens_per_page: int
    cost_per_1k_tokens_usd: float


def load_vision_config(rules_path: Path) -> VisionConfig:
    with rules_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    vcfg = data.get("vision", {})
    return VisionConfig(
        max_cost_usd_per_document=float(vcfg["max_cost_usd_per_document"]),
        estimated_tokens_per_page=int(vcfg["estimated_tokens_per_page"]),
        cost_per_1k_tokens_usd=float(vcfg["cost_per_1k_tokens_usd"]),
    )


class VisionExtractor(ExtractionStrategy):
    """Strategy C – Vision-augmented extraction via a multimodal VLM (e.g., OpenRouter)."""

    name = "vision_vlm"

    def __init__(
        self,
        rules_path: Path | None = None,
    ) -> None:
        if rules_path is None:
            rules_path = Path("rubric/extraction_rules.yaml")
        self.vcfg = load_vision_config(rules_path)
        self.llm_cfg = LLMConfig.from_env()

    def _estimate_cost(self, num_pages: int) -> float:
        tokens = num_pages * self.vcfg.estimated_tokens_per_page
        return (tokens / 1000.0) * self.vcfg.cost_per_1k_tokens_usd

    def _page_to_png_bytes(self, page) -> bytes:
        # Rasterize a pdfplumber page to PNG in-memory
        pil_img = page.to_image(resolution=200).original
        from io import BytesIO

        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def _call_vlm(self, image_bytes: bytes) -> Optional[dict]:
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "You are a document parsing assistant. Given this page image, "
            "return a JSON object with fields: text_blocks (array of {text}), "
            "tables (array of {headers, rows}), and figures (array of {caption}). "
            "Do not include any explanation text, only JSON."
        )

        try:
            return call_vision_llm(b64, prompt, self.llm_cfg)
        except Exception:
            # In environments without a configured LLM, degrade gracefully.
            return None

    def extract(self, profile: DocumentProfile) -> ExtractionResult:
        pdf_path = Path(profile.source_path)

        est_cost = self._estimate_cost(profile.num_pages)
        if est_cost > self.vcfg.max_cost_usd_per_document:
            # Budget guard: we will still try, but mark cost capped.
            est_cost = self.vcfg.max_cost_usd_per_document

        text_blocks: List[TextBlock] = []
        tables: List[Table] = []
        figures: List[Figure] = []
        page_confidences: Dict[int, float] = {}

        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                image_bytes = self._page_to_png_bytes(page)
                result_json = self._call_vlm(image_bytes)

                if not result_json:
                    # If no VLM call possible, we degrade gracefully: empty content but low confidence.
                    page_confidences[page_idx] = 0.1
                    continue

                # Text blocks
                for tb in result_json.get("text_blocks", []):
                    text = tb.get("text", "")
                    if not text:
                        continue
                    # We do not have true coordinates; we store None bbox and rely on content_hash later.
                    text_blocks.append(
                        TextBlock(
                            page_number=page_idx,
                            bbox=BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0),
                            text=text,
                            reading_order=len(text_blocks),
                        )
                    )

                # Tables
                for tbl in result_json.get("tables", []):
                    headers = [str(h) for h in tbl.get("headers", [])]
                    rows_data = tbl.get("rows", [])
                    rows: List[List[TableCell]] = []
                    for r_idx, row in enumerate(rows_data):
                        cells_row: List[TableCell] = []
                        for c_idx, cell_value in enumerate(row):
                            cells_row.append(
                                TableCell(
                                    row_index=r_idx,
                                    col_index=c_idx,
                                    text=str(cell_value),
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

                # Figures
                for fig in result_json.get("figures", []):
                    figures.append(
                        Figure(
                            page_number=page_idx,
                            bbox=None,
                            caption=fig.get("caption"),
                            label=fig.get("label"),
                        )
                    )

                # Confidence: we assume VLM extraction is high quality when it returns anything
                has_any = bool(
                    result_json.get("text_blocks")
                    or result_json.get("tables")
                    or result_json.get("figures")
                )
                page_confidences[page_idx] = 0.9 if has_any else 0.2

        extracted = ExtractedDocument(
            doc_id=profile.doc_id,
            num_pages=profile.num_pages,
            text_blocks=text_blocks,
            tables=tables,
            figures=figures,
            metadata={
                "strategy_used": self.name,
                "backend": self.llm_cfg.backend,
                "vision_model": self.llm_cfg.vision_model,
                "cost_estimate_usd": est_cost,
            },
        )

        return ExtractionResult(
            document=extracted,
            page_confidences=page_confidences,
            strategy_name=self.name,
            cost_estimate_usd=est_cost,
        )

