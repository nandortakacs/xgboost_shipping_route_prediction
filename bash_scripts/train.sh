#!/bin/bash
# Only enable strict mode in a subshell (not when sourced interactively)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    set -o pipefail
fi

usage() {
    cat <<EOF
Usage: bash bash_scripts/train.sh [grid] [model_type] [wandb_mode]

Arguments:
  grid        Grid resolution: "fine" (default) or "coarse"
  model_type  Feature set:     "simple" (default) or "full"
  wandb_mode  W&B logging:     "disabled" (default), "online", "offline", or "dryrun"

Examples:
  bash bash_scripts/train.sh
  bash bash_scripts/train.sh coarse full online
EOF
    exit "${1:-0}"
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage 0

GRID="${1:-fine}"
MODEL_TYPE="${2:-simple}"
WANDB_MODE="${3:-disabled}"

[[ "$GRID" =~ ^(fine|coarse)$ ]] || { echo "Error: grid must be 'fine' or 'coarse', got '$GRID'"; usage 1; }
[[ "$MODEL_TYPE" =~ ^(simple|full)$ ]] || { echo "Error: model_type must be 'simple' or 'full', got '$MODEL_TYPE'"; usage 1; }
[[ "$WANDB_MODE" =~ ^(disabled|online|offline|dryrun)$ ]] || { echo "Error: wandb_mode must be 'disabled', 'online', 'offline', or 'dryrun', got '$WANDB_MODE'"; usage 1; }


PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${GRID}_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"

python -m shipping_route_predictor.train \
    --grid "$GRID" \
    --model_type "$MODEL_TYPE" \
    --wandb_mode "$WANDB_MODE" \
    2>&1 | tee "$LOG_FILE"
