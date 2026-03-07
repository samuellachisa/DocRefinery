from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder, enrich_ldus_with_sections
from src.agents.query_agent import (
    QueryAgent,
    VectorStoreConfig,
    VectorStore,
    FactTable,
    build_langgraph_agent,
)
from src.agents.fact_extractor import FactExtractor
from src.agents.triage import profile_document


def _preview_ldus_from_vector_store(
    doc_id: str,
    vector_store: VectorStore,
    max_chunks: int = 5,
    max_chars: int = 400,
) -> None:
    """
    Best-effort preview of LDU chunks as stored in the vector DB.
    This is purely for human exploration/debugging.
    """
    print("\n=== LDU Preview from Vector Store ===", flush=True)
    coll = vector_store.collection
    res = coll.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
        limit=max_chunks,
    )
    docs: List[str] = list(res.get("documents") or [])
    metas: List[Dict[str, Any]] = list(res.get("metadatas") or [])

    if not docs:
        print("No chunks found in vector store for this document.", flush=True)
        return

    for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
        print(f"\n--- Chunk {idx} ---", flush=True)
        print(f"chunk_type: {meta.get('chunk_type')}", flush=True)
        print(f"page_refs: {meta.get('page_refs')}", flush=True)
        snippet = (doc[: max_chars - 3] + "...") if len(doc) > max_chars else doc
        print("content:", snippet.replace("\n", " "), flush=True)


def run_pipeline(
    pdf_path: Path,
    question: str,
    show_ldu_preview: bool = False,
    ldu_preview_max_chunks: int = 5,
    ldu_preview_max_chars: int = 400,
) -> None:
    pdf_path = pdf_path.expanduser().resolve()

    print(f"\n=== Starting analysis for: {pdf_path.name} ===", flush=True)

    # 1) Triage
    print("\n[1/5] Understanding the document type...", flush=True)
    profile = profile_document(pdf_path)
    print("=== DocumentProfile ===", flush=True)
    print(profile.model_dump_json(indent=2))

    # 2) Extraction (with escalation & ledger)
    print("\n[2/5] Reading the document contents...", flush=True)
    router = ExtractionRouter()
    extraction_result = router.route(profile)
    extracted = extraction_result.document
    print("\n=== Extraction Summary ===", flush=True)
    print(
        {
            "strategy_used": extraction_result.strategy_name,
            "num_text_blocks": len(extracted.text_blocks),
            "num_tables": len(extracted.tables),
            "num_figures": len(extracted.figures),
        }
    )

    # 3) Chunking
    print("\n[3/5] Organizing content into chunks...", flush=True)
    chunker = ChunkingEngine()
    ldus = chunker.chunk(extracted)
    print(f"\n=== Chunking ===\nGenerated {len(ldus)} LDUs", flush=True)

    # 4) PageIndex (with LDUs for data_types and section enrichment)
    print("\n[4/5] Building a table of contents and summaries...", flush=True)
    index_builder = PageIndexBuilder()
    page_index = index_builder.build(extracted, ldus=ldus)
    enrich_ldus_with_sections(ldus, page_index)
    print("\n=== PageIndex Root ===", flush=True)
    print(page_index.root.model_dump())

    # 5) Structured facts + Query Agent & sample question
    print("\n[5/5] Answering your question with provenance...", flush=True)
    vector_store = VectorStore(VectorStoreConfig())
    fact_table = FactTable()

    # Populate structured fact table from extracted tables before querying.
    fact_extractor = FactExtractor()
    fact_extractor.extract(extracted, fact_table)

    agent = QueryAgent(
        page_index=page_index,
        ldus=ldus,
        vector_store=vector_store,
        fact_table=fact_table,
    )

    # Use LangGraph agent wrapper
    graph = build_langgraph_agent(agent)
    final_state = graph.invoke(
        {"question": question, "doc_name": pdf_path.name},
        config={"configurable": {"run_name": "docrefinery_query"}},
    )
    answer = final_state["answer"]
    provenance = final_state["provenance"]

    print("\n=== QA Result ===", flush=True)
    print("Q:", question)
    print("A:", answer)
    print("\nProvenanceChain:", flush=True)
    print(provenance.model_dump_json(indent=2))

    if show_ldu_preview:
        _preview_ldus_from_vector_store(
            doc_id=page_index.doc_id,
            vector_store=vector_store,
            max_chunks=ldu_preview_max_chunks,
            max_chars=ldu_preview_max_chars,
        )

    print(f"\n=== Analysis completed for: {pdf_path.name} ===\n", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run full Document Intelligence Refinery pipeline on a single PDF."
    )
    parser.add_argument("pdf_path", type=str, help="Path to input PDF")
    parser.add_argument(
        "--question",
        type=str,
        default="What are the main findings of this report?",
        help="Sample question to ask the query agent.",
    )
    parser.add_argument(
        "--show-ldu-preview",
        action="store_true",
        help="After running the pipeline, print a preview of stored LDU chunks "
        "for this document from the vector DB.",
    )
    parser.add_argument(
        "--ldu-preview-max-chunks",
        type=int,
        default=5,
        help="Maximum number of chunks to show in the LDU preview (default: 5).",
    )
    parser.add_argument(
        "--ldu-preview-max-chars",
        type=int,
        default=400,
        help="Maximum number of characters of content to display per chunk in the LDU preview.",
    )
    args = parser.parse_args()

    run_pipeline(
        Path(args.pdf_path),
        args.question,
        show_ldu_preview=args.show_ldu_preview,
        ldu_preview_max_chunks=args.ldu_preview_max_chunks,
        ldu_preview_max_chars=args.ldu_preview_max_chars,
    )


if __name__ == "__main__":
    main()

