from __future__ import annotations

from pathlib import Path

from src.agents.chunker import ChunkingEngine
from src.agents.extractor import ExtractionRouter
from src.agents.indexer import PageIndexBuilder
from src.agents.query_agent import (
    QueryAgent,
    VectorStoreConfig,
    VectorStore,
    FactTable,
    build_langgraph_agent,
)
from src.agents.triage import profile_document


def run_pipeline(pdf_path: Path, question: str) -> None:
    pdf_path = pdf_path.expanduser().resolve()

    # 1) Triage
    profile = profile_document(pdf_path)
    print("=== DocumentProfile ===")
    print(profile.model_dump_json(indent=2))

    # 2) Extraction (with escalation & ledger)
    router = ExtractionRouter()
    extraction_result = router.route(profile)
    extracted = extraction_result.document
    print("\n=== Extraction Summary ===")
    print(
        {
            "strategy_used": extraction_result.strategy_name,
            "num_text_blocks": len(extracted.text_blocks),
            "num_tables": len(extracted.tables),
            "num_figures": len(extracted.figures),
        }
    )

    # 3) Chunking
    chunker = ChunkingEngine()
    ldus = chunker.chunk(extracted)
    print(f"\n=== Chunking ===\nGenerated {len(ldus)} LDUs")

    # 4) PageIndex
    index_builder = PageIndexBuilder()
    page_index = index_builder.build(extracted)
    print("\n=== PageIndex Root ===")
    print(page_index.root.model_dump())

    # 5) Query Agent & sample question
    vector_store = VectorStore(VectorStoreConfig())
    fact_table = FactTable()
    agent = QueryAgent(
        page_index=page_index,
        ldus=ldus,
        vector_store=vector_store,
        fact_table=fact_table,
    )

    # Use LangGraph agent wrapper
    graph = build_langgraph_agent(agent)
    final_state = graph.invoke({"question": question, "doc_name": pdf_path.name})
    answer = final_state["answer"]
    provenance = final_state["provenance"]

    print("\n=== QA Result ===")
    print("Q:", question)
    print("A:", answer)
    print("\nProvenanceChain:")
    print(provenance.model_dump_json(indent=2))


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
    args = parser.parse_args()

    run_pipeline(Path(args.pdf_path), args.question)


if __name__ == "__main__":
    main()

