#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DOC_PATH="${1:-}"
QUESTION="${2:-}"

if [[ -z "$DOC_PATH" ]]; then
  echo
  echo "Welcome to the Document Intelligence Refinery."
  echo
  echo "This tool can:"
  echo "  - Analyse a document (triage)"
  echo "  - Read out its contents (extraction)"
  echo "  - Answer a question about it (full pipeline)"
  echo
  echo "How to use it:"
  echo "  1. Put your PDF in the 'data' folder (or anywhere you like)."
  echo "  2. Run this command and follow the prompts:"
  echo "       ./run_pipeline.sh"
  echo
  read -rp "Enter the path to your PDF (for example: data/ETHSWITCH-Annual-Report-202122.pdf): " DOC_PATH
fi

if [[ ! -f "$DOC_PATH" ]]; then
  echo
  echo "I couldn't find the file:"
  echo "  $DOC_PATH"
  echo
  echo "Please check the path and try again."
  echo
  exit 1
fi

if [[ -z "$QUESTION" ]]; then
  read -rp "What question do you want to ask about this document? (Press Enter to skip for now): " QUESTION
fi

echo
echo "📄 Reading your document..."
echo "   File    : $(basename "$DOC_PATH")"
if [[ -n "$QUESTION" ]]; then
  echo "   Question: $QUESTION"
fi
echo

echo "What would you like to do?"
echo "  1) Triage only  – understand the document type"
echo "  2) Extraction   – read out the contents (no question answering)"
echo "  3) Full QA      – answer your question with provenance"
echo "  4) Chunk only   – build chunks & vector store from cached extraction"
echo "  5) Vector DB    – explore stored chunks (LDUs)"
echo
read -rp "Choose an option (1/2/3/4/5): " CHOICE

# Ensure uv is available for all options.
if ! command -v uv >/dev/null 2>&1; then
  echo
  echo "The 'uv' tool is required but not installed."
  echo "You can still run the components manually with:"
  echo "  python -m src.agents.triage \"$DOC_PATH\""
  echo "  python -m src.agents.extractor \"$DOC_PATH\""
  echo "  python scripts/run_pipeline.py \"$DOC_PATH\" --question \"$QUESTION\""
  exit 1
fi

case "$CHOICE" in
  1)
    echo
    echo "Step 1/1 – Understanding the document type..."
    uv run python -m src.agents.triage "$DOC_PATH"
    ;;
  2)
    echo
    echo "Step 1/1 – Reading the document contents..."
    uv run python -m src.agents.extractor "$DOC_PATH"
    ;;
  3)
    if [[ -z "$QUESTION" ]]; then
      read -rp "Please enter your question: " QUESTION
    fi
    echo
    read -rp "Also show a preview of stored chunks (LDUs) from the vector DB after answering? [y/N]: " SHOW_LDUS
    SHOW_LDUS=${SHOW_LDUS:-N}
    LDU_FLAGS=()
    if [[ "$SHOW_LDUS" =~ ^[Yy]$ ]]; then
      LDU_FLAGS+=(--show-ldu-preview)
    fi
    echo
    echo "I will now run the full analysis and answer your question."
    echo "This can take a little while for large reports."
    echo
    uv run python -m scripts.run_pipeline "$DOC_PATH" --question "$QUESTION" "${LDU_FLAGS[@]}" \
      | awk '
          /=== QA Result ===/ {show=1}
          show {print}
        '
    ;;
  4)
    echo
    echo "Step 1/1 – Chunking using cached extraction (no re-extraction)..."
    echo
    uv run python -m scripts.chunk_from_extracted "$DOC_PATH"
    ;;
  5)
    echo
    echo "Vector DB explorer – what would you like to see?"
    echo "  a) Summary    – overall collection statistics"
    echo "  b) List docs  – doc_ids with their chunk counts"
    echo "  c) Show doc   – chunks for a specific doc_id"
    echo "  d) Search     – semantic search over chunks"
    echo "  e) Raw dump   – raw JSON of the collection"
    echo
    read -rp "Choose an explorer option (a/b/c/d/e): " VCHOICE
    case "$VCHOICE" in
      a|A)
        uv run python -m scripts.vector_explorer summary
        ;;
      b|B)
        uv run python -m scripts.vector_explorer list-docs
        ;;
      c|C)
        read -rp "Enter doc_id to inspect: " DOC_ID
        uv run python -m scripts.vector_explorer show-doc "$DOC_ID"
        ;;
      d|D)
        read -rp "Enter semantic search query: " QUERY
        uv run python -m scripts.vector_explorer search "$QUERY"
        ;;
      e|E)
        uv run python -m scripts.vector_explorer raw
        ;;
      *)
        echo
        echo "I did not recognise that explorer option."
        ;;
    esac
    ;;
  *)
    echo
    echo "I did not recognise that option. Please run ./run_pipeline.sh again and choose 1, 2, 3, 4, or 5."
    ;;
esac


