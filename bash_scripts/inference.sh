#!/bin/bash
# Only enable strict mode in a subshell (not when sourced interactively)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -e
    set -o pipefail
fi

usage() {
    cat <<EOF
Usage: bash bash_scripts/inference.sh [options]

Predict a vessel route with XGBoost models and baselines.

Options:
  --grid GRID             Grid resolution: "coarse" (default) or "fine"
  --start_port PORT       Start port LOCODE (default: CNSHA)
  --goal_port PORT        Goal port LOCODE (default: USLAX)
  --start_lat LAT         Start latitude  (overrides --start_port)
  --start_lon LON         Start longitude (overrides --start_port)
  --goal_lat LAT          Goal latitude   (overrides --goal_port)
  --goal_lon LON          Goal longitude  (overrides --goal_port)
  --models "m1 m2 ..."    Space-separated model list (default: simple_greedy shortest_path)
                           Valid: simple_greedy simple_cisc full_greedy full_cisc
                                  shortest_path company_baseline
  --speed_knots KN        Vessel speed in knots (default: 14.5)
  --shap                  Run SHAP explainability after inference
  --log_level LEVEL       Log level (default: INFO)
  -h, --help              Show this help

Examples:
  # Default (Shanghai → Long Beach, simple_greedy + shortest_path)
  bash bash_scripts/inference.sh

  # Custom ports with all models on coarse grid
  bash bash_scripts/inference.sh --grid coarse --start_port USLAX --goal_port KRPUS \\
      --models "simple_greedy simple_cisc full_greedy full_cisc shortest_path company_baseline"

  # Custom lat/lon with SHAP
  bash bash_scripts/inference.sh --start_lat 1.3 --start_lon 103.8 \\
      --goal_lat 55.0 --goal_lon 12.5 --shap

  # Coarse grid, specific models
  bash bash_scripts/inference.sh --grid coarse --models "simple_greedy full_cisc"
EOF
    exit "${1:-0}"
}

# Defaults
GRID="coarse"
START_PORT=""
GOAL_PORT=""
START_LAT=""
START_LON=""
GOAL_LAT=""
GOAL_LON=""
MODELS="simple_greedy shortest_path"
SPEED=""
SHAP=""
LOG_LEVEL="INFO"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)      usage 0 ;;
        --grid)         GRID="$2"; shift 2 ;;
        --start_port)   START_PORT="$2"; shift 2 ;;
        --goal_port)    GOAL_PORT="$2"; shift 2 ;;
        --start_lat)    START_LAT="$2"; shift 2 ;;
        --start_lon)    START_LON="$2"; shift 2 ;;
        --goal_lat)     GOAL_LAT="$2"; shift 2 ;;
        --goal_lon)     GOAL_LON="$2"; shift 2 ;;
        --models)       MODELS="$2"; shift 2 ;;
        --speed_knots)  SPEED="$2"; shift 2 ;;
        --shap)         SHAP="--shap"; shift ;;
        --log_level)    LOG_LEVEL="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; usage 1 ;;
    esac
done

[[ "$GRID" =~ ^(fine|coarse)$ ]] || { echo "Error: grid must be 'fine' or 'coarse', got '$GRID'"; usage 1; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs/inference"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/inference_${GRID}_$(date +%Y%m%d_%H%M%S).log"

# Build command
CMD="python -m shipping_route_predictor.inference --grid $GRID --model_types $MODELS --log_level $LOG_LEVEL"

[[ -n "$START_PORT" ]] && CMD="$CMD --start_port $START_PORT"
[[ -n "$GOAL_PORT" ]]  && CMD="$CMD --goal_port $GOAL_PORT"
[[ -n "$START_LAT" ]]  && CMD="$CMD --start_lat $START_LAT"
[[ -n "$START_LON" ]]  && CMD="$CMD --start_lon $START_LON"
[[ -n "$GOAL_LAT" ]]   && CMD="$CMD --goal_lat $GOAL_LAT"
[[ -n "$GOAL_LON" ]]   && CMD="$CMD --goal_lon $GOAL_LON"
[[ -n "$SPEED" ]]      && CMD="$CMD --speed_knots $SPEED"
[[ -n "$SHAP" ]]       && CMD="$CMD $SHAP"

echo "=== Vessel Route Inference ==="
echo "Grid:   $GRID"
echo "Models: $MODELS"
echo "Log:    $LOG_FILE"
echo ""

$CMD 2>&1 | tee "$LOG_FILE"
