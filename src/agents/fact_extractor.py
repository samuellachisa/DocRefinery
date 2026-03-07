from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.agents.query_agent import FactTable
from src.models import ExtractedDocument, Table
from src.utils.content_hash import content_hash


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

    def _parse_numeric(self, text: str) -> Optional[float]:
        """
        Extract a numeric value from a cell.

        Handles common patterns like:
        - "950.00"
        - "USD 950.00"
        - "$950.00"
        - "1,234.56"
        """
        # Find the first integer/float-like pattern in the string.
        match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text.replace("\u00a0", " "))
        if not match:
            return None
        number_str = match.group(0).replace(",", "")
        try:
            return float(number_str)
        except ValueError:
            return None

    def _table_content_for_hash(self, tbl: Table) -> str:
        """Build table content string in same format as chunker for stable content_hash."""
        header_line = " | ".join(tbl.headers) if tbl.headers else ""
        row_lines = [" | ".join(cell.text for cell in row) for row in tbl.rows]
        return "\n".join([header_line] + row_lines if header_line else row_lines).strip()

    def _extract_from_table(self, doc_id: str, tbl: Table, fact_table: FactTable) -> None:
        if not tbl.rows:
            return
        # When no headers, use first row as header for robust extraction (production-ready)
        if tbl.headers:
            headers = list(tbl.headers)
            data_rows = tbl.rows
        else:
            headers = [cell.text.strip() or f"col_{j}" for j, cell in enumerate(tbl.rows[0])]
            data_rows = tbl.rows[1:]

        table_content = self._table_content_for_hash(tbl)
        tbl_hash = content_hash(doc_id, [tbl.page_number], table_content)
        for row in data_rows:
            if not row:
                continue
            entity = row[0].text.strip()
            if not entity:
                continue

            for col_idx, cell in enumerate(row[1:], start=1):
                value_text = cell.text.strip()
                if not value_text:
                    continue
                parsed_value = self._parse_numeric(value_text)
                if parsed_value is None:
                    continue

                header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                metric = header
                period = header  # in many financial tables, header encodes period
                value = parsed_value

                fact_table.insert_fact(
                    doc_id=doc_id,
                    entity=entity,
                    metric=metric,
                    value=value,
                    period=str(period),
                    page_number=tbl.page_number,
                    content_hash=tbl_hash,
                )

    def extract(self, doc: ExtractedDocument, fact_table: FactTable) -> None:
        for tbl in doc.tables:
            self._extract_from_table(doc.doc_id, tbl, fact_table)

