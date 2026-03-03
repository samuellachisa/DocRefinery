from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.agents.triage import profile_document
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.indexer import PageIndexBuilder
from src.agents.fact_extractor import FactExtractor
from src.agents.query_agent import VectorStoreConfig, VectorStore, FactTable


def iter_pdfs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.pdf"):
        yield p


def batch_process(root: Path) -> None:
    router = ExtractionRouter()
    chunker = ChunkingEngine()
    index_builder = PageIndexBuilder()
    fact_extractor = FactExtractor()
    vector_store = VectorStore(VectorStoreConfig())
    fact_table = FactTable()

    for pdf_path in iter_pdfs(root):
        profile = profile_document(pdf_path)
        result = router.route(profile)
        extracted = result.document
        ldus = chunker.chunk(extracted)
        index_builder.build(extracted)
        fact_extractor.extract(extracted, fact_table)
        vector_store.ingest(ldus)
        print(f"Processed {pdf_path.name} (doc_id={profile.doc_id})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch process a directory of PDFs through the refinery pipeline."
    )
    parser.add_argument("root", type=str, help="Root directory containing PDFs.")
    args = parser.parse_args()

    batch_process(Path(args.root).expanduser().resolve())


if __name__ == "__main__":
    main()

