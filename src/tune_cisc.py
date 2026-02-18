"""Tune CISC (T, α, β, N) parameters for XGBoost rollout selection.

Grid-searches over temperature, alpha, beta, and number-of-rollouts
to find the combination that minimises Fréchet distance to ground truth.

Usage
-----
    python -m shipping_route_predictor.tune_cisc --grid fine --model_type simple

    python -m shipping_route_predictor.tune_cisc --grid coarse --model_type full --limit 50
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import xgboost as xgb

from shipping_route_predictor.config import (
    CISCTuningConfig,
    GridResolution,
    ModelSpec,
    ModelType,
    RolloutConfig,
    save_config,
)
from shipping_route_predictor.data import AISDataset, EnvDataset
from shipping_route_predictor.env import EnvGraph, build_env
from shipping_route_predictor.rollout import Rollout
from shipping_route_predictor.utils import discrete_frechet_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("tune_cisc")


# ======================================================================
# Result data class
# ======================================================================

@dataclass
class TuningResult:
    """Metrics for a single (T, α, β, N) combination."""
    temperature: float
    alpha: float
    beta: float
    n_rollouts_used: int

    n_gt_trajectories: int    # total GT trajectories evaluated
    n_with_success: int       # trajectories with ≥1 successful rollout
    cisc_mean_frechet: float  # mean Fréchet to GT across trajectories (km)
    cisc_success_rate: float  # fraction of trajectories where CISC selected a successful rollout
    rollout_success_rate: float  # mean fraction of rollouts that reached the goal per trajectory
    cisc_mean_score: float    # mean CISC score of the selected rollout


# ======================================================================
# Per-trajectory evaluation
# ======================================================================

def evaluate_trajectory(
    rollouts: List[dict],
    gt_latlon_path: List[Tuple[float, float]],
    rollout_runner: Rollout,
    alpha: float,
    beta: float,
    n_use: int,
) -> Dict[str, Any]:
    """Evaluate CISC selection for one trajectory's rollouts.

    Parameters
    ----------
    rollouts : list[dict]
        Full set of rollouts (greedy first, then stochastic).
        Only the first *n_use* are considered.
    gt_latlon_path : list[(lat, lon)]
        Ground-truth path in geographic coordinates.
    rollout_runner : Rollout
        Rollout instance (used for CISC selection and grid→latlon).
    alpha, beta : float
        CISC weighting parameters.
    n_use : int
        Number of rollouts to use from the list.
    """
    rollouts = rollouts[:n_use]
    n_total = len(rollouts)
    n_successful = sum(1 for r in rollouts if r["reached_goal"])

    out: Dict[str, Any] = {"n_total": n_total, "n_successful": n_successful}

    selected, score = rollout_runner.select_cisc(rollouts, alpha=alpha, beta=beta)

    if n_successful == 0:
        out["cisc_frechet"] = float("nan")
        out["cisc_success"] = False
        out["cisc_score"] = float("nan")
    else:
        cisc_latlon_path = rollout_runner.grid_to_latlon_path(selected["grid_path"])
        out["cisc_frechet"] = discrete_frechet_distance(cisc_latlon_path, gt_latlon_path)
        out["cisc_success"] = True
        out["cisc_score"] = score
    return out


# ======================================================================
# Main tuning loop
# ======================================================================

def generate_all_rollouts(
    rollout_runner: Rollout,
    trajectories: List[Tuple[Any, List[Tuple[int, int]]]],
    temperatures: List[float],
    n_rollouts: int,
) -> Tuple[Dict[float, List[Tuple[List[dict], List[Tuple[float, float]]]]], Dict[str, float]]:
    """Pre-compute rollouts for every trajectory at every temperature.

    Returns
    -------
    rollouts_by_temp : {temperature → [(rollouts, gt_latlon_path), ...]}
    timing : dict with timing percentiles.
    """
    rollouts_by_temp: Dict[float, list] = {}
    all_times: List[float] = []

    for temp in temperatures:
        log.info("Generating %d rollouts at T=%.2f for %d trajectories …",
                 n_rollouts, temp, len(trajectories))
        temp_data: list = []

        for ti, (spec, grid_path) in enumerate(trajectories):
            rollout_runner.cfg.model.cisc.temperature = temp
            rollout_runner.cfg.model.cisc.n_rollouts = n_rollouts

            gt_latlon_path = [rollout_runner.env.grid_indices_to_latlon(rc) for rc in grid_path]

            t0 = time.perf_counter()
            rollouts = rollout_runner._generate_rollouts(spec, n_rollouts, temp)
            elapsed = time.perf_counter() - t0
            all_times.append(elapsed)

            n_ok = sum(1 for r in rollouts if r["reached_goal"])
            log.info(
                "  [%d/%d] %s  T=%.2f  %d/%d reached goal  %.2fs",
                ti + 1, len(trajectories), spec.name,
                temp, n_ok, len(rollouts), elapsed,
            )

            temp_data.append((rollouts, gt_latlon_path))

        rollouts_by_temp[temp] = temp_data
        log.info("  → %d trajectories completed at T=%.2f", len(temp_data), temp)

    timing: Dict[str, float] = {}
    if all_times:
        arr = np.array(all_times)
        timing = {
            "n_timed": len(arr),
            "mean_s": float(arr.mean()),
            "std_s": float(arr.std()),
            "min_s": float(arr.min()),
            "max_s": float(arr.max()),
            "p50_s": float(np.percentile(arr, 50)),
            "p90_s": float(np.percentile(arr, 90)),
            "total_s": float(arr.sum()),
        }
    return rollouts_by_temp, timing


def grid_search(
    rollouts_by_temp: Dict[float, list],
    rollout_runner: Rollout,
    temperatures: List[float],
    alphas: List[float],
    betas: List[float],
    n_rollouts_list: List[int],
) -> List[TuningResult]:
    """Evaluate all (T, α, β, N) combinations.

    Does NOT regenerate rollouts — reuses the pre-computed *rollouts_by_temp*.
    """
    combos = list(itertools.product(temperatures, n_rollouts_list, alphas, betas))
    log.info("Evaluating %d (T, N, α, β) combinations …", len(combos))

    results: List[TuningResult] = []

    for temp, n_use, alpha, beta in combos:
        temp_data = rollouts_by_temp.get(temp, [])

        cisc_frechets: List[float] = []
        cisc_successes: List[bool] = []
        rollout_success_rates: List[float] = []
        cisc_scores: List[float] = []
        n_with_success = 0

        for rollouts, gt_latlon_path in temp_data:
            ev = evaluate_trajectory(rollouts, gt_latlon_path, rollout_runner, alpha, beta, n_use)

            n_s, n_t = ev["n_successful"], ev["n_total"]
            if n_t > 0:
                rollout_success_rates.append(n_s / n_t)
            if n_s > 0:
                n_with_success += 1

            cisc_f = ev["cisc_frechet"]
            if not np.isnan(cisc_f):
                cisc_frechets.append(cisc_f)
            cisc_successes.append(ev["cisc_success"])

            s = ev.get("cisc_score", float("nan"))
            if not np.isnan(s):
                cisc_scores.append(s)

        def _safe_mean(lst):
            return float(np.mean(lst)) if lst else float("nan")

        results.append(TuningResult(
            temperature=temp,
            alpha=alpha,
            beta=beta,
            n_rollouts_used=n_use,
            n_gt_trajectories=len(temp_data),
            n_with_success=n_with_success,
            cisc_mean_frechet=_safe_mean(cisc_frechets),
            cisc_success_rate=float(np.mean(cisc_successes)) if cisc_successes else 0.0,
            rollout_success_rate=_safe_mean(rollout_success_rates),
            cisc_mean_score=_safe_mean(cisc_scores),
        ))

    return results

def print_results(results: List[TuningResult], timing: Dict[str, float]) -> None:
    """Print a summary table and highlight the best configuration."""
    print("\n" + "=" * 120)
    print("CISC TUNING RESULTS")
    print("=" * 120)

    if timing:
        print("\nROLLOUT TIMING:")
        print(f"  Total batches timed: {timing.get('n_timed', 0)}")
        print(f"  Mean / Std:          {timing.get('mean_s', 0):.2f} s / {timing.get('std_s', 0):.2f} s")
        print(f"  Min / Max:           {timing.get('min_s', 0):.3f} s / {timing.get('max_s', 0):.3f} s")
        print(f"  p50 / p90:           {timing.get('p50_s', 0):.3f} s / {timing.get('p90_s', 0):.3f} s")
        print(f"  Total wall time:     {timing.get('total_s', 0):.1f} s")

    valid = [r for r in results if not np.isnan(r.cisc_mean_frechet)]
    if not valid:
        print("\nNo valid results!")
        return

    temp_values = sorted({r.temperature for r in valid})
    n_values = sorted({r.n_rollouts_used for r in valid})

    print("\n" + "-" * 100)
    print("BEST (α, β) FOR EACH (Temperature, N)")
    print("-" * 100)
    print(f"{'T':>5} {'N':>4} {'α':>6} {'β':>6} "
          f"{'Fréchet':>10} {'Score':>8} | "
          f"{'CISC-SR':>8} {'RollSR':>8}")
    print("-" * 100)

    for temp in temp_values:
        for n in n_values:
            subset = [r for r in valid if r.temperature == temp and r.n_rollouts_used == n]
            if not subset:
                continue
            best = min(subset, key=lambda r: r.cisc_mean_frechet)
            print(
                f"{temp:>5.2f} {n:>4} {best.alpha:>6.1f} {best.beta:>6.1f} "
                f"{best.cisc_mean_frechet:>10.1f} "
                f"{best.cisc_mean_score:>8.2f} | "
                f"{best.cisc_success_rate:>7.1%} "
                f"{best.rollout_success_rate:>7.1%}"
            )

    best = min(valid, key=lambda r: r.cisc_mean_frechet)
    print("-" * 100)
    print(f"\nOVERALL BEST: T={best.temperature:.2f}, N={best.n_rollouts_used}, "
          f"α={best.alpha:.1f}, β={best.beta:.1f}")
    print(f"  CISC Fréchet to GT:      {best.cisc_mean_frechet:.1f} km")
    print(f"  CISC score:              {best.cisc_mean_score:.2f}")
    print(f"  CISC selection success:  {best.cisc_success_rate:.1%}")
    print(f"  Rollout success rate:    {best.rollout_success_rate:.1%}")
    print(f"  Trajectories ≥1 success: {best.n_with_success}/{best.n_gt_trajectories}")

    # Top 10
    top10 = sorted(valid, key=lambda r: r.cisc_mean_frechet)[:10]
    print("\n" + "-" * 100)
    print("TOP 10 (by CISC Fréchet to GT)")
    print("-" * 100)
    print(f"{'T':>5} {'N':>4} {'α':>6} {'β':>6} "
          f"{'Fréchet':>10} {'Score':>8} | "
          f"{'CISC-SR':>8} {'RollSR':>8}")
    print("-" * 100)
    for r in top10:
        print(
            f"{r.temperature:>5.2f} {r.n_rollouts_used:>4} {r.alpha:>6.1f} {r.beta:>6.1f} "
            f"{r.cisc_mean_frechet:>10.1f} "
            f"{r.cisc_mean_score:>8.2f} | "
            f"{r.cisc_success_rate:>7.1%} "
            f"{r.rollout_success_rate:>7.1%}"
        )
    print("=" * 100)


def save_results(
    results: List[TuningResult],
    timing: Dict[str, float],
    cfg: CISCTuningConfig,
    output_dir: str,
) -> Path:
    """Persist tuning results and config to JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "timing": timing,
        "search_grid": {
            "temperatures": cfg.temperatures,
            "alphas": cfg.alphas,
            "betas": cfg.betas,
            "n_rollouts_list": cfg.n_rollouts_list,
        },
        "results": [asdict(r) for r in results],
    }

    json_path = out_path / "tuning_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Results saved to %s", json_path)

    save_config(cfg, str(out_path), filename="tuning_config.json")
    return json_path

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tune CISC (T, α, β, N) for XGBoost rollout selection",
    )
    p.add_argument("--grid", type=str, default="fine", choices=["fine", "coarse"])
    p.add_argument("--model_type", type=str, default="simple", choices=["simple", "full"])
    p.add_argument("--limit", type=int, default=None,
                   help="Override CISCTuningConfig.limit")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override CISCTuningConfig.output_dir")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    grid = GridResolution(args.grid)
    model_type = ModelType(args.model_type)

    # Build config — search grid, dataset, seed all come from CISCTuningConfig defaults
    rollout_cfg = RolloutConfig(
        grid=grid,
        model=ModelSpec(
            name=f"xgb_{grid}_{model_type}",
            model_type=model_type,
        ),
    )
    cfg = CISCTuningConfig(rollout=rollout_cfg)

    if args.limit is not None:
        cfg.limit = args.limit
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    max_n = max(cfg.n_rollouts_list)

    log.info("Grid: %s | model_type: %s | dataset: %s | limit: %d",
             grid, model_type, cfg.dataset, cfg.limit)
    log.info("Search: T ∈ %s, α ∈ %s, β ∈ %s, N ∈ %s",
             cfg.temperatures, cfg.alphas, cfg.betas, cfg.n_rollouts_list)

    env_ds = EnvDataset(rollout_cfg.env, model_type, rollout_cfg.weather)
    env = build_env(env_ds)

    model_path = rollout_cfg.model.path
    if model_path is None:
        raise FileNotFoundError("No model path configured — check _DEFAULT_MODEL_FILES in config.")
    log.info("Loading model from %s", model_path)
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # --- load trajectories -------------------------------------------------
    ais = AISDataset(env, rollout_cfg.env)
    trajectories = ais.load_trajectories(cfg.dataset, limit=cfg.limit)
    log.info("Loaded %d trajectories from '%s' split", len(trajectories), cfg.dataset)

    if not trajectories:
        log.error("No trajectories loaded — nothing to tune.")
        sys.exit(1)

    rollout_runner = Rollout(rollout_cfg, model, env)

    rollouts_by_temp, timing = generate_all_rollouts(
        rollout_runner, trajectories, cfg.temperatures, max_n,
    )

    results = grid_search(
        rollouts_by_temp, rollout_runner,
        cfg.temperatures, cfg.alphas, cfg.betas, cfg.n_rollouts_list,
    )

    print_results(results, timing)
    save_results(results, timing, cfg, cfg.output_dir)


if __name__ == "__main__":
    main()
