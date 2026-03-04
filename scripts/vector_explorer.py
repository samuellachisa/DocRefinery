from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.query_agent import VectorStore, VectorStoreConfig


def _build_vector_store(persist_directory: str | None = None) -> VectorStore:
    """
    Helper to open the Chroma-backed VectorStore used for LDU chunks.
    """
    cfg = VectorStoreConfig()
    if persist_directory is not None:
        cfg.persist_directory = persist_directory
    return VectorStore(cfg)


def cmd_summary(args: argparse.Namespace) -> None:
    """
    Print a high-level summary of the LDU collection.
    """
    vs = _build_vector_store(args.persist_dir)
    coll = vs.collection

    total = coll.count()
    print(f"Total LDU chunks in collection: {total}")
    if total == 0:
        return

    res = coll.get(include=["metadatas"], limit=total)
    metadatas: List[Dict[str, Any]] = list(res.get("metadatas") or [])

    doc_ids = [m.get("doc_id") for m in metadatas if m and m.get("doc_id")]
    doc_counter = Counter(doc_ids)
    print(f"Distinct documents: {len(doc_counter)}")

    # Show top 10 docs by chunk count
    print("\nTop documents by chunk count:")
    for doc_id, cnt in doc_counter.most_common(10):
        print(f"  {doc_id}: {cnt} chunks")


def cmd_list_docs(args: argparse.Namespace) -> None:
    """
    List all doc_ids present in the vector store, with their chunk counts.
    """
    vs = _build_vector_store(args.persist_dir)
    coll = vs.collection

    total = coll.count()
    if total == 0:
        print("No chunks found in collection.")
        return

    res = coll.get(include=["metadatas"], limit=total)
    metadatas: List[Dict[str, Any]] = list(res.get("metadatas") or [])

    doc_counter = Counter(
        m.get("doc_id") for m in metadatas if m and m.get("doc_id")
    )

    for doc_id, cnt in doc_counter.most_common():
        print(f"{doc_id}: {cnt} chunks")


def _print_chunk(idx: int, content: str, meta: Dict[str, Any], max_chars: int) -> None:
    print(f"\n=== Chunk #{idx} ===")
    print(f"id: {meta.get('content_hash') or meta.get('id')}")
    print(f"doc_id: {meta.get('doc_id')}")
    print(f"chunk_type: {meta.get('chunk_type')}")
    print(f"page_refs: {meta.get('page_refs')}")
    print("content:")
    snippet = (content[: max_chars - 3] + "...") if len(content) > max_chars else content
    print(snippet.replace("\n", " "))


def cmd_show_doc(args: argparse.Namespace) -> None:
    """
    Show chunks for a single document by doc_id.
    """
    vs = _build_vector_store(args.persist_dir)
    coll = vs.collection

    where: Dict[str, Any] = {"doc_id": args.doc_id}
    res = coll.get(where=where, include=["documents", "metadatas"])
    docs: List[str] = list(res.get("documents") or [])
    metas: List[Dict[str, Any]] = list(res.get("metadatas") or [])

    if not docs:
        print(f"No chunks found for doc_id={args.doc_id}")
        return

    print(f"Found {len(docs)} chunks for doc_id={args.doc_id}")
    for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
        _print_chunk(idx, doc, meta or {}, args.max_chars)


def cmd_search(args: argparse.Namespace) -> None:
    """
    Run a semantic search against the LDU collection and print matching chunks.
    """
    vs = _build_vector_store(args.persist_dir)
    matches: List[Tuple[str, Dict[str, Any]]] = vs.query(args.query, k=args.k)

    if not matches:
        print("No matches found.")
        return

    print(f"Top {len(matches)} matches for query: {args.query!r}")
    for idx, (ldu_id, meta) in enumerate(matches, start=1):
        content = meta.get("content", "")
        _print_chunk(idx, content, meta, args.max_chars)


def cmd_raw(args: argparse.Namespace) -> None:
    """
    Dump raw collection contents (documents + metadatas) as JSON for debugging.
    """
    vs = _build_vector_store(args.persist_dir)
    coll = vs.collection

    total = coll.count()
    if total == 0:
        print("No chunks found in collection.")
        return

    res = coll.get(include=["documents", "metadatas"], limit=total)
    print(json.dumps(res, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vector store explorer for LDU chunks (.refinery/chroma)."
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=".refinery/chroma",
        help="Path to Chroma persistence directory (default: .refinery/chroma).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # summary
    p_summary = subparsers.add_parser(
        "summary", help="Show overall collection statistics."
    )
    p_summary.set_defaults(func=cmd_summary)

    # list-docs
    p_list_docs = subparsers.add_parser(
        "list-docs", help="List all doc_ids with their chunk counts."
    )
    p_list_docs.set_defaults(func=cmd_list_docs)

    # show-doc
    p_show_doc = subparsers.add_parser(
        "show-doc", help="Show chunks for a single document by doc_id."
    )
    p_show_doc.add_argument("doc_id", type=str, help="Document id to inspect.")
    p_show_doc.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Maximum number of characters of content to display per chunk.",
    )
    p_show_doc.set_defaults(func=cmd_show_doc)

    # search
    p_search = subparsers.add_parser(
        "search", help="Semantic search over LDUs using the existing embeddings."
    )
    p_search.add_argument("query", type=str, help="Free-text query.")
    p_search.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of nearest chunks to return (default: 5).",
    )
    p_search.add_argument(
        "--max-chars",
        type=int,
        default=400,
        help="Maximum number of characters of content to display per chunk.",
    )
    p_search.set_defaults(func=cmd_search)

    # raw
    p_raw = subparsers.add_parser(
        "raw",
        help="Dump raw Chroma collection contents (documents + metadatas) as JSON.",
    )
    p_raw.set_defaults(func=cmd_raw)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

