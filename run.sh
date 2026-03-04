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
    MODE="${3:-expert}"  # expert | simple
    if [[ -z "$PDF_PATH" ]]; then
      echo "Usage: ./run.sh pipeline <pdf_path> [question] [mode]" >&2
      echo "  mode: 'expert' (default) shows all steps; 'simple' shows only the final answer." >&2
      exit 1
    fi
    if [[ "$MODE" == "simple" ]]; then
      uv run python -m scripts.run_pipeline "$PDF_PATH" --question "$QUESTION" \
        | awk '
            /=== QA Result ===/ {show=1}
            show {print}
          '
    else
      uv run python -m scripts.run_pipeline "$PDF_PATH" --question "$QUESTION"
    fi
    ;;

  batch)
    ROOT="${1:-data}"
    uv run python -m scripts.batch_process "$ROOT"
    ;;

  tests)
    uv run pytest
    ;;

  *)
    cat <<EOF
Usage:
  ./run.sh pipeline <pdf_path> [question] [mode]   # Run pipeline on a single PDF
                                                 #   mode: expert (default) or simple
  ./run.sh batch [root_dir]                # Batch process all PDFs under root_dir (default: data/)
  ./run.sh tests                           # Run test suite
EOF
    ;;
esac

