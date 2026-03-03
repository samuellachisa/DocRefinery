from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.models import ExtractedDocument, Table
from src.agents.query_agent import FactTable


@dataclass
class FactExtractorConfig:
    """
    Heuristics for extracting numerical facts from tables.
    """

    min_numeric_fraction: float = 0.5


class FactExtractor:
    """
    Extracts (entity, metric, value, period) style facts from table structures
    into the FactTable SQLite backend.
    """

    def __init__(self, cfg: Optional[FactExtractorConfig] = None) -> None:
        self.cfg = cfg or FactExtractorConfig()

    def _is_numeric(self, text: str) -> bool:
        stripped = text.replace(",", "").replace(" ", "")
        try:
            float(stripped)
            return True
        except ValueError:
            return False

    def _extract_from_table(self, doc_id: str, tbl: Table, fact_table: FactTable) -> None:
        if not tbl.headers or not tbl.rows:
            return

        headers = tbl.headers
        # Assume first column is entity, remaining columns are metrics/periods.
        for row in tbl.rows:
            if not row:
                continue
            entity = row[0].text.strip()
            if not entity:
                continue

            for col_idx, cell in enumerate(row[1:], start=1):
                value_text = cell.text.strip()
                if not value_text:
                    continue
                if not self._is_numeric(value_text):
                    continue

                header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                metric = header
                period = header  # in many financial tables, header encodes period

                try:
                    value = float(value_text.replace(",", ""))
                except ValueError:
                    continue

                fact_table.insert_fact(
                    doc_id=doc_id,
                    entity=entity,
                    metric=metric,
                    value=value,
                    period=str(period),
                    page_number=tbl.page_number,
                    content_hash="",  # could be linked to LDU hash if needed
                )

    def extract(self, doc: ExtractedDocument, fact_table: FactTable) -> None:
        for tbl in doc.tables:
            self._extract_from_table(doc.doc_id, tbl, fact_table)

