from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.chunker import ChunkingEngine
from src.agents.fact_extractor import FactExtractor
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import VectorStore, VectorStoreConfig
from src.models import ExtractedDocument


def _find_doc_id_for_path(pdf_path: Path) -> Optional[str]:
    """
    Look up the doc_id for a given PDF path using the persisted profiles.
    """
    profiles_dir = Path(".refinery/profiles")
    if not profiles_dir.is_dir():
        return None

    pdf_path_resolved = pdf_path.expanduser().resolve()
    for profile_path in profiles_dir.glob("*.json"):
        try:
            with profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        source = data.get("source_path")
        if not source:
            continue
        if Path(source).expanduser().resolve() == pdf_path_resolved:
            return str(data.get("doc_id"))
    return None


def run_chunk_only(
    pdf_path: Path,
    show_ldu_preview: bool = True,
    ldu_preview_max_chunks: int = 5,
    ldu_preview_max_chars: int = 400,
) -> None:
    pdf_path = pdf_path.expanduser().resolve()
    print(f"\n=== Chunk-only pipeline for: {pdf_path.name} ===", flush=True)

    doc_id = _find_doc_id_for_path(pdf_path)
    if doc_id is None:
        raise SystemExit(
            "Could not find a matching DocumentProfile for this PDF. "
            "Please run triage or extraction first."
        )

    extracted_path = Path(".refinery/extracted") / f"{doc_id}.json"
    if not extracted_path.is_file():
        raise SystemExit(
            f"No cached extraction found for doc_id={doc_id}.\n"
            "Run the extraction step first (option 2 in run_pipeline.sh)."
        )

    print(f"Using cached extraction: {extracted_path}", flush=True)
    with extracted_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    extracted = ExtractedDocument.model_validate(data)

    # Chunking
    print("\n[1/3] Organizing content into chunks...", flush=True)
    chunker = ChunkingEngine()
    ldus = chunker.chunk(extracted)
    print(f"Generated {len(ldus)} LDUs", flush=True)

    # Vector store ingest
    print("\n[2/3] Ingesting chunks into vector store...", flush=True)
    vector_store = VectorStore(VectorStoreConfig())
    vector_store.ingest(ldus)
    print("Ingestion complete.", flush=True)

    # Optional preview
    if show_ldu_preview:
        print("\n[3/3] Previewing a few stored chunks...", flush=True)
        res = vector_store.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
            limit=ldu_preview_max_chunks,
        )
        docs: List[str] = list(res.get("documents") or [])
        metas: List[Dict[str, Any]] = list(res.get("metadatas") or [])

        if not docs:
            print("No chunks found in vector store for this document.", flush=True)
        else:
            for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
                print(f"\n--- Chunk {idx} ---", flush=True)
                print(f"chunk_type: {meta.get('chunk_type')}", flush=True)
                print(f"page_refs: {meta.get('page_refs')}", flush=True)
                snippet = (
                    doc[: ldu_preview_max_chars - 3] + "..."
                    if len(doc) > ldu_preview_max_chars
                    else doc
                )
                print("content:", snippet.replace("\n", " "), flush=True)
    else:
        print("\n[3/3] Skipping chunk preview.", flush=True)

    print(f"\n=== Chunk-only pipeline completed for: {pdf_path.name} ===\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run chunking and vector ingest using a cached ExtractedDocument, "
            "without re-running extraction."
        )
    )
    parser.add_argument("pdf_path", type=str, help="Path to input PDF.")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not print a preview of stored chunks.",
    )
    parser.add_argument(
        "--preview-max-chunks",
        type=int,
        default=5,
        help="Maximum number of chunks to show in the preview (default: 5).",
    )
    parser.add_argument(
        "--preview-max-chars",
        type=int,
        default=400,
        help="Maximum number of characters per chunk to show in the preview.",
    )
    args = parser.parse_args()

    run_chunk_only(
        Path(args.pdf_path),
        show_ldu_preview=not args.no_preview,
        ldu_preview_max_chunks=args.preview_max_chunks,
        ldu_preview_max_chars=args.preview_max_chars,
    )


if __name__ == "__main__":
    main()

