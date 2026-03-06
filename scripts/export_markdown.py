from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from src.models import BoundingBox, ExtractedDocument, TableCell


def _render_table(headers: List[str], rows: List[List[TableCell]]) -> str:
    """
    Render a Table into GitHub-flavoured Markdown.
    """
    # Determine max column count from headers/rows
    col_count = max(
        len(headers),
        max((len(r) for r in rows), default=0),
    )
    if col_count == 0:
        return ""

    # Normalise headers
    if not headers:
        headers = [f"Col {i+1}" for i in range(col_count)]
    elif len(headers) < col_count:
        headers = headers + ["" for _ in range(col_count - len(headers))]
    else:
        headers = headers[:col_count]

    # Header row + separator
    lines: List[str] = []
    lines.append("| " + " | ".join(h.strip() or " " for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in range(col_count)) + " |")

    # Data rows
    for row in rows:
        cells = [c.text for c in row]
        if len(cells) < col_count:
            cells = cells + ["" for _ in range(col_count - len(cells))]
        else:
            cells = cells[:col_count]
        lines.append("| " + " | ".join(c.strip() for c in cells) + " |")

    return "\n".join(lines)


def _render_figure_image(
    pdf_path: Path,
    page_number: int,
    bbox: Optional[BoundingBox],
    out_path: Path,
    resolution: int = 150,
) -> bool:
    """
    Render a PDF region (page, optionally cropped by bbox) to a PNG file.
    Returns True if the file was written, False on error.
    """
    try:
        import pdfplumber
    except ImportError:
        return False

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                return False
            page = pdf.pages[page_number - 1]
            pil_img = page.to_image(resolution=resolution).original
            w_pt = float(page.width)
            h_pt = float(page.height)
    except Exception:
        return False

    pw, ph = pil_img.size
    scale_x = pw / w_pt if w_pt else 1.0
    scale_y = ph / h_pt if h_pt else 1.0

    if bbox is not None and (bbox.x1 > bbox.x0 and bbox.y1 > bbox.y0):
        # PDF coordinates: assume top-left origin (y down). Clip to page.
        x0 = max(0, min(bbox.x0, w_pt))
        y0 = max(0, min(bbox.y0, h_pt))
        x1 = max(0, min(bbox.x1, w_pt))
        y1 = max(0, min(bbox.y1, h_pt))
        if x1 > x0 and y1 > y0:
            left = int(x0 * scale_x)
            top = int(y0 * scale_y)
            right = int(x1 * scale_x)
            bottom = int(y1 * scale_y)
            pil_img = pil_img.crop((left, top, right, bottom))
    # else: use full page (e.g. figure without bbox from VLM)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out_path, format="PNG")
    return True


def extracted_to_markdown(
    doc: ExtractedDocument,
    pdf_path: Optional[Path] = None,
    asset_dir: Optional[Path] = None,
) -> str:
    """
    Convert an ExtractedDocument into a human-friendly Markdown view.

    If pdf_path and asset_dir are provided, figures/charts are rendered from the PDF
    into asset_dir/images/ and embedded as ![Figure N](images/figure_p3_1.png).
    """
    lines: List[str] = []
    export_images = pdf_path is not None and asset_dir is not None
    images_dir: Optional[Path] = asset_dir / "images" if asset_dir else None

    for page in range(1, doc.num_pages + 1):
        page_blocks = [tb for tb in doc.text_blocks if tb.page_number == page]
        page_tables = [t for t in doc.tables if t.page_number == page]
        page_figs = [f for f in doc.figures if f.page_number == page]

        if not page_blocks and not page_tables and not page_figs:
            continue

        # Page heading
        lines.append(f"# Page {page}")
        lines.append("")

        # Text blocks in reading order
        for tb in sorted(page_blocks, key=lambda t: t.reading_order):
            text = (tb.text or "").strip()
            if not text:
                continue
            lines.append(text)
            lines.append("")

        # Tables
        for idx, tbl in enumerate(page_tables, start=1):
            title_bits: List[str] = []
            if tbl.title:
                title_bits.append(tbl.title.strip())
            if tbl.caption:
                title_bits.append(tbl.caption.strip())
            if title_bits:
                lines.append(f"*Table {idx}:* " + " — ".join(title_bits))
                lines.append("")

            rendered = _render_table(tbl.headers, tbl.rows)
            if rendered:
                lines.append(rendered)
                lines.append("")

        # Figures / charts: caption + optional embedded image
        for idx, fig in enumerate(page_figs, start=1):
            caption = (fig.caption or "").strip() if fig.caption else ""
            label = (fig.label or "").strip() if fig.label else ""
            title = f"Figure {idx}" + (f" ({label})" if label else "")
            if caption:
                lines.append(f"*{title}:* {caption}")
            else:
                lines.append(f"*{title}* (image/chart)")
            lines.append("")

            if export_images and pdf_path and images_dir:
                rel_name = f"figure_p{page}_{idx}.png"
                out_path = images_dir / rel_name
                if _render_figure_image(
                    pdf_path, page, fig.bbox, out_path
                ):
                    # Relative path from the markdown file (document.md lives in asset_dir)
                    lines.append(f"![{title}](images/{rel_name})")
                    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a cached ExtractedDocument (.refinery/extracted/*.json) "
            "to a human-readable Markdown file."
        )
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the input PDF. Used to locate the matching doc_id profile.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Optional output path for the Markdown file. "
            "Defaults to .refinery/markdown/<doc_id>/document.md"
        ),
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).expanduser().resolve()

    # Locate matching profile to get doc_id
    profiles_dir = Path(".refinery/profiles")
    if not profiles_dir.is_dir():
        raise SystemExit("No .refinery/profiles directory found. Run triage first.")

    doc_id: str | None = None
    for profile_path in profiles_dir.glob("*.json"):
        try:
            with profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        source = data.get("source_path")
        if not source:
            continue
        if Path(source).expanduser().resolve() == pdf_path:
            doc_id = str(data.get("doc_id"))
            break

    if not doc_id:
        raise SystemExit(
            f"Could not find a DocumentProfile for PDF: {pdf_path}. "
            "Make sure you've run triage/extraction for this file."
        )

    extracted_path = Path(".refinery/extracted") / f"{doc_id}.json"
    if not extracted_path.is_file():
        raise SystemExit(
            f"No cached extraction found at {extracted_path}. "
            "Run the extraction step first."
        )

    with extracted_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    extracted = ExtractedDocument.model_validate(data)

    if args.output:
        out_path = Path(args.output).expanduser()
        asset_dir = out_path.parent
    else:
        out_dir = Path(".refinery/markdown") / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "document.md"
        asset_dir = out_dir

    md = extracted_to_markdown(
        extracted,
        pdf_path=pdf_path,
        asset_dir=asset_dir,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote Markdown to: {out_path}", flush=True)
    if asset_dir and (asset_dir / "images").exists():
        print(f"Figure/chart images: {asset_dir / 'images'}", flush=True)


if __name__ == "__main__":
    main()

