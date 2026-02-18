#!/bin/bash
# Only enable strict mode in a subshell (not when sourced interactively)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    set -o pipefail
fi

usage() {
    cat <<EOF
Usage: bash bash_scripts/eval.sh [grid]

Arguments:
  grid  Grid resolution: "fine" (default) or "coarse"

Examples:
  bash bash_scripts/eval.sh
  bash bash_scripts/eval.sh coarse
EOF
    exit "${1:-0}"
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage 0

GRID="${1:-fine}"

[[ "$GRID" =~ ^(fine|coarse)$ ]] || { echo "Error: grid must be 'fine' or 'coarse', got '$GRID'"; usage 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs/eval"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_${GRID}_$(date +%Y%m%d_%H%M%S).log"

python -m shipping_route_predictor.eval \
    --grid "$GRID" \
    2>&1 | tee "$LOG_FILE"
