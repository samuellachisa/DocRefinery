#!/usr/bin/env bash
set -euo pipefail

# Simple project launcher that uses uv under the hood.
#
# Usage:
#   ./run.sh pipeline data/doc.pdf "Your question"
#   ./run.sh batch data/
#   ./run.sh tests
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it from https://github.com/astral-sh/uv" >&2
  exit 1
fi

CMD="${1:-}"
shift || true

case "$CMD" in
  pipeline)
    PDF_PATH="${1:-}"
    QUESTION="${2:-What are the main findings of this report?}"
    if [[ -z "$PDF_PATH" ]]; then
      echo "Usage: ./run.sh pipeline <pdf_path> [question]" >&2
      exit 1
    fi
    uv run scripts/run_pipeline.py "$PDF_PATH" --question "$QUESTION"
    ;;

  batch)
    ROOT="${1:-data}"
    uv run scripts/batch_process.py "$ROOT"
    ;;

  tests)
    uv run pytest
    ;;

  *)
    cat <<EOF
Usage:
  ./run.sh pipeline <pdf_path> [question]   # Run full pipeline on a single PDF
  ./run.sh batch [root_dir]                # Batch process all PDFs under root_dir (default: data/)
  ./run.sh tests                           # Run test suite
EOF
    ;;
esac

