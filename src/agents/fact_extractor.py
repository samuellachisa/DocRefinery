from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from src.agents.query_agent import FactTable
from src.models import ExtractedDocument, Table
from src.utils.content_hash import content_hash


# Patterns for multi-period financial tables: column headers as years or quarters
_YEAR_RE = re.compile(r"^(?:FY)?\s*['\"]?(\d{4})['\"]?\s*$", re.IGNORECASE)
_QUARTER_RE = re.compile(r"^(?:Q[1-4]|Quarter\s*[1-4])[\s\-]*(?:FY)?\s*['\"]?(\d{4})['\"]?$", re.IGNORECASE)
_METRIC_PERIOD_RE = re.compile(r"^(.+?)\s+(?:FY\s*)?['\"]?(\d{4})['\"]?$", re.IGNORECASE)


@dataclass
class FactExtractorConfig:
    """
    Heuristics for extracting numerical facts from tables.
    """

    min_numeric_fraction: float = 0.5
    """Use multi-period extraction when share of period-like headers exceeds this (0.0-1.0)."""


def _looks_like_period(header: str) -> bool:
    """True if header looks like a year or quarter (e.g. 2023, Q1 2023, FY2022)."""
    h = header.strip()
    return bool(_YEAR_RE.match(h) or _QUARTER_RE.match(h) or _METRIC_PERIOD_RE.match(h))


def _parse_period(header: str) -> str:
    """Normalize header to a period string for storage (e.g. '2023', 'Q1 2023')."""
    h = header.strip()
    m = _QUARTER_RE.match(h)
    if m:
        return f"Q{h[:2].upper()} {m.group(1)}" if "Q" in h.upper() else f"{h} {m.group(1)}"
    m = _YEAR_RE.match(h)
    if m:
        return m.group(1)
    m = _METRIC_PERIOD_RE.match(h)
    if m:
        return m.group(2)
    return h


def _parse_metric_from_header(header: str) -> str:
    """If header is 'Revenue 2022', return 'Revenue'; else return header."""
    m = _METRIC_PERIOD_RE.match(header.strip())
    return m.group(1).strip() if m else header.strip()


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

    def _is_multi_period_table(self, headers: List[str]) -> bool:
        """True if a majority of data-column headers look like periods (years/quarters)."""
        if not headers or len(headers) < 2:
            return False
        # Skip first column (entity)
        data_headers = headers[1:]
        period_like = sum(1 for h in data_headers if _looks_like_period(h))
        return period_like >= self.cfg.min_numeric_fraction * len(data_headers)

    def _entity_from_row(
        self, row: List, entity_col_count: int, headers: List[str]
    ) -> str:
        """Build entity label from one or more leading columns (multi-entity support)."""
        parts = []
        for j in range(min(entity_col_count, len(row))):
            parts.append(row[j].text.strip() or "")
        label = " | ".join(p for p in parts if p)
        return label or "Unknown"

    def _extract_multi_period_table(
        self, doc_id: str, tbl: Table, fact_table: FactTable,
        headers: List[str], data_rows: List, tbl_hash: str, base_metric: str,
        entity_col_count: int = 1,
    ) -> None:
        """
        Extract facts from multi-period layout: one fact per (entity, metric, period, value).
        Column headers are treated as periods (or "metric period"); base_metric used when header is pure period.
        """
        for row in data_rows:
            if len(row) <= entity_col_count:
                continue
            entity = self._entity_from_row(row, entity_col_count, headers)
            if not entity:
                continue
            for col_idx in range(entity_col_count, min(len(row), len(headers))):
                cell = row[col_idx]
                value_text = cell.text.strip()
                if not value_text:
                    continue
                parsed_value = self._parse_numeric(value_text)
                if parsed_value is None:
                    continue
                header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                period = _parse_period(header)
                metric = base_metric if _looks_like_period(header) and not _METRIC_PERIOD_RE.match(header.strip()) else _parse_metric_from_header(header)
                fact_table.insert_fact(
                    doc_id=doc_id,
                    entity=entity,
                    metric=metric,
                    value=parsed_value,
                    period=period,
                    page_number=tbl.page_number,
                    content_hash=tbl_hash,
                )

    def _extract_from_table(self, doc_id: str, tbl: Table, fact_table: FactTable) -> None:
        if not tbl.rows:
            return
        if tbl.headers:
            headers = list(tbl.headers)
            data_rows = tbl.rows
        else:
            headers = [cell.text.strip() or f"col_{j}" for j, cell in enumerate(tbl.rows[0])]
            data_rows = tbl.rows[1:]

        table_content = self._table_content_for_hash(tbl)
        tbl_hash = content_hash(doc_id, [tbl.page_number], table_content)
        base_metric = (tbl.title or "Value").strip()

        if len(headers) >= 2 and self._is_multi_period_table(headers):
            self._extract_multi_period_table(
                doc_id, tbl, fact_table,
                headers=headers,
                data_rows=data_rows,
                tbl_hash=tbl_hash,
                base_metric=base_metric,
                entity_col_count=1,
            )
            return

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
                metric = _parse_metric_from_header(header)
                period = _parse_period(header) if _looks_like_period(header) else header
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

