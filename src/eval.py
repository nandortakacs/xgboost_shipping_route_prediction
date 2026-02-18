"""Unified evaluation for XGBoost vessel-routing models and baselines.

Evaluates all ``RolloutConfig`` entries in an ``EvalConfig`` — running
both greedy and CISC rollouts for each model — plus optional
shortest-path and company baselines.  Produces per-trajectory CSVs,
summary JSONs, a comparison table, and optionally histograms.

Usage
-----
``python -m shipping_route_predictor.eval --grid coarse``
``python -m shipping_route_predictor.eval --grid fine``
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm

from shipping_route_predictor.baselines import CompanyBaseline, ShortestPathBaseline
from shipping_route_predictor.config import (
    EvalConfig,
    GridResolution,
    RolloutType,
    TrajectorySpec,
)
from shipping_route_predictor.data import AISDataset, EnvDataset
from shipping_route_predictor.env import EnvGraph, build_env
from shipping_route_predictor.rollout import Rollout
from shipping_route_predictor.utils import discrete_frechet_distance

log = logging.getLogger("eval")

class Evaluator:
    """Runs multi-model evaluation over a shared trajectory set.

    Parameters
    ----------
    cfg : EvalConfig
        Full evaluation configuration (grid, rollouts, baselines, etc.).
    """

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

        # ── Build shared env and load trajectories ────────────────
        first_cfg = cfg.rollouts[0]
        self.env: EnvGraph = build_env(EnvDataset(
            first_cfg.env, first_cfg.model.model_type, first_cfg.weather,
        ))

        ais = AISDataset(self.env, first_cfg.env)
        self.trajectories = ais.load_trajectories(cfg.dataset, limit=cfg.limit)
        log.info(
            "Loaded %d trajectories from '%s' split (grid=%s, limit=%s)",
            len(self.trajectories), cfg.dataset, cfg.grid, cfg.limit,
        )
        if not self.trajectories:
            raise RuntimeError(
                f"No trajectories loaded for split={cfg.dataset}, grid={cfg.grid}."
            )

        self.gt_latlon_cache: List[List[Tuple[float, float]]] = [
            [self.env.grid_indices_to_latlon(rc) for rc in gt_path]
            for _spec, gt_path in self.trajectories
        ]

        # ── Build XGBoost model runners ───────────────────────────────────
        self.runners = self.build_runners()

        # ── Build baselines ───────────────────────────────────────
        self.sp_baseline: Optional[ShortestPathBaseline] = None
        self.co_baseline: Optional[CompanyBaseline] = None

        if cfg.include_shortest_path_baseline:
            self.sp_baseline = ShortestPathBaseline(first_cfg.env)
            log.info("ShortestPathBaseline ready")

        if cfg.include_company_baseline:
            self.co_baseline = CompanyBaseline(first_cfg.env, self.sp_baseline)
            log.info("CompanyBaseline ready")

        # ── Per-model result storage ──────────────────────────────
        self.per_model_rows: Dict[str, List[Dict[str, Any]]] = {}

    def run(self) -> Path:
        """Execute the full evaluation and save results.

        Returns the output directory.
        """
        cfg = self.cfg

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path(cfg.output_dir) / f"{cfg.grid}_{ts}"
        out_root.mkdir(parents=True, exist_ok=True)

        with open(out_root / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)
        log.info("Output directory: %s", out_root)

        self.evaluate(out_root)
        self._save_results(out_root)

        log.info("Evaluation complete. Results in %s", out_root)
        return out_root

    def evaluate(self, out_root: Path) -> None:
        """Run all models + baselines over every trajectory."""
        cfg = self.cfg

        progress = tqdm(range(len(self.trajectories)), desc="Eval", unit="traj")
        for i in progress:
            spec, gt_grid_path = self.trajectories[i]
            gt_latlon = self.gt_latlon_cache[i]

            # ── XGBoost model rollouts ────────────────────────────
            max_steps = min(len(gt_grid_path) + cfg.extra_steps_over_gt_length,
                            self.runners[0][2].cfg.max_rollout_steps if self.runners else 400)
            for model_name, rollout_type, runner in self.runners:
                def _run_xgboost_model(r=runner, rt=rollout_type):
                    return r.run_route(spec, rt, max_steps=max_steps)

                self._evaluate_model_on_trajectory(
                    _run_xgboost_model, i, spec, gt_grid_path, gt_latlon,
                    model_name, rollout_type.value,
                )

            # ── Shortest-path baseline ────────────────────────────
            if self.sp_baseline is not None:
                def _run_shortest_path():
                    return self.sp_baseline.predict_route(spec.start_rc, spec.goal_rc)

                self._evaluate_model_on_trajectory(
                    _run_shortest_path, i, spec, gt_grid_path, gt_latlon,
                    "shortest_path", "dijkstra",
                )

            # ── Company baseline ──────────────────────────────────
            if self.co_baseline is not None:
                def _run_company_baseline():
                    return self.co_baseline.predict_route(spec.start_rc, spec.goal_rc)

                self._evaluate_model_on_trajectory(
                    _run_company_baseline, i, spec, gt_grid_path, gt_latlon,
                    "company_baseline", "routing_graph",
                )

            traj_num = i + 1
            if cfg.progress_save_interval > 0 and traj_num % cfg.progress_save_interval == 0:
                self._save_progress(out_root, traj_num)
            progress.set_postfix(models=len(self.per_model_rows))

    def build_runners(self) -> List[Tuple[str, RolloutType, Rollout]]:
        """Load XGBoost models and create Rollout runners.

        Models sharing ``(model_type, path)`` are loaded once.
        Rollout type is inferred from the model name suffix:
        ``_greedy`` → greedy only, ``_cisc`` → CISC only, otherwise both.
        """
        runners: List[Tuple[str, RolloutType, Rollout]] = []
        loaded: Dict[str, Tuple[xgb.XGBClassifier, EnvGraph]] = {}

        for rcfg in self.cfg.rollouts:
            key = f"{rcfg.model.model_type}:{rcfg.model.path}"
            if key not in loaded:
                env = build_env(EnvDataset(rcfg.env, rcfg.model.model_type, rcfg.weather))
                model_path = rcfg.model.path
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                log.info("Loaded %s from %s", rcfg.model.name, model_path)
                loaded[key] = (model, env)

            model, env = loaded[key]
            runner = Rollout(rcfg, model, env)
            name = rcfg.model.name

            name_lower = name.lower()
            if name_lower.endswith("_greedy"):
                runners.append((name, RolloutType.GREEDY, runner))
            elif name_lower.endswith("_cisc"):
                runners.append((name, RolloutType.CISC, runner))
            else:
                runners.append((f"{name}_greedy", RolloutType.GREEDY, runner))
                runners.append((f"{name}_cisc", RolloutType.CISC, runner))

        return runners
    
    def _evaluate_model_on_trajectory(
        self,
        run_fn,
        traj_idx: int,
        spec: TrajectorySpec,
        gt_grid_path: List[Tuple[int, int]],
        gt_latlon: List[Tuple[float, float]],
        model_name: str,
        rollout_type: str,
    ) -> None:
        """Time *run_fn*, build an eval row, and append to results.

        *run_fn* must return ``(result_dict | None, pred_latlon | None)``.
        """
        t0 = time.perf_counter()
        result, pred_latlon = run_fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        row = self._build_eval_row(
            traj_idx, spec, gt_grid_path, gt_latlon,
            model_name, rollout_type, result, pred_latlon, elapsed_ms,
        )
        self.per_model_rows.setdefault(model_name, []).append(row)

    

    def _build_eval_row(
        self,
        traj_idx: int,
        spec: TrajectorySpec,
        gt_grid_path: List[Tuple[int, int]],
        gt_latlon: List[Tuple[float, float]],
        model_name: str,
        rollout_type: str,
        result: Optional[Dict[str, Any]],
        pred_latlon: Optional[List[Tuple[float, float]]],
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        """Construct a single evaluation-result row."""
        reached = False
        pred_steps = 0
        frechet_km = np.nan
        mean_prob = np.nan

        if result is not None:
            reached = bool(result.get("reached_goal", False))
            pred_steps = len(result.get("grid_path", []))
            _mp = result.get("mean_prob")
            mean_prob = float(_mp) if _mp is not None else np.nan

            if pred_latlon and gt_latlon and len(pred_latlon) >= 2 and len(gt_latlon) >= 2:
                frechet_km = discrete_frechet_distance(pred_latlon, gt_latlon)

        pred_path_json = json.dumps(result["grid_path"]) if result and result.get("grid_path") else None
        gt_path_json = json.dumps(gt_grid_path)

        return {
            "trajectory_idx": traj_idx,
            "trajectory_name": spec.name,
            "model_name": model_name,
            "rollout_type": rollout_type,
            "start_coords": spec.start_rc,
            "goal_coords": spec.goal_rc,
            "gt_length": len(gt_grid_path),
            "pred_steps": pred_steps,
            "reached_goal": reached,
            "frechet_km": frechet_km,
            "mean_prob": mean_prob,
            "elapsed_ms": elapsed_ms,
            "pred_grid_path": pred_path_json,
            "gt_grid_path": gt_path_json,
        }

    def _summarize_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-trajectory rows into summary statistics."""
        if not rows:
            return {"num_trajectories": 0, "success_rate": 0.0}

        df = pd.DataFrame(rows)
        reached = df["reached_goal"].fillna(False).astype(bool)
        fr = pd.to_numeric(df["frechet_km"], errors="coerce")
        ms = pd.to_numeric(df["elapsed_ms"], errors="coerce")

        def _safe(series, fn, default=None):
            valid = series.dropna()
            return fn(valid) if len(valid) > 0 else default

        return {
            "num_trajectories": int(len(df)),
            "success_rate": float(reached.mean()),
            "frechet_km_mean": _safe(fr, lambda s: float(s.mean())),
            "frechet_km_median": _safe(fr, lambda s: float(s.median())),
            "frechet_success_mean": _safe(fr[reached], lambda s: float(s.mean())),
            "frechet_success_median": _safe(fr[reached], lambda s: float(s.median())),
            "ms_per_traj_mean": _safe(ms, lambda s: float(s.mean())),
            "ms_per_traj_median": _safe(ms, lambda s: float(s.median())),
            "ms_per_traj_p95": _safe(ms, lambda s: float(s.quantile(0.95))),
            "ms_total": _safe(ms, lambda s: float(s.sum())),
        }

    def _save_results(self, out_root: Path) -> None:
        """Write per-model CSVs, summaries, comparison table, and histogram."""
        cfg = self.cfg
        comparison_rows: List[Dict[str, Any]] = []

        for name, rows in self.per_model_rows.items():
            model_dir = out_root / name.replace("/", "_").replace(" ", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(rows).to_csv(model_dir / "trajectories.csv", index=False)

            summ = self._summarize_rows(rows)
            summ["model_name"] = name
            with open(model_dir / "summary.json", "w") as f:
                json.dump(summ, f, indent=2)

            log.info(
                "%-35s  success=%.1f%%  fréchet_mean=%.1f km  (%d trajs)",
                name,
                summ["success_rate"] * 100,
                summ.get("frechet_km_mean") or 0,
                summ["num_trajectories"],
            )
            comparison_rows.append(summ)

        # Comparison table
        pd.DataFrame(comparison_rows).to_csv(out_root / "comparison.csv", index=False)
        with open(out_root / "comparison.json", "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "grid": cfg.grid, "models": comparison_rows},
                f, indent=2,
            )
        log.info("Comparison saved to %s", out_root / "comparison.csv")

        # Histogram
        if cfg.visualize:
            self._save_histogram(out_root)

    def _save_histogram(self, out_root: Path) -> None:
        """Generate Fréchet histogram across models (success-only)."""
        metric = self.cfg.histogram_metric
        bins = self.cfg.histogram_bins
        title_extra = f" — {self.cfg.grid} grid"
        model_values: Dict[str, np.ndarray] = {}

        for name, rows in self.per_model_rows.items():
            df = pd.DataFrame(rows)
            reached = df["reached_goal"].fillna(False).astype(bool)
            vals = pd.to_numeric(df.loc[reached, metric], errors="coerce").dropna().to_numpy()
            if len(vals) > 0:
                model_values[name] = vals

        if not model_values:
            return

        all_vals = np.concatenate(list(model_values.values()))
        bin_edges = np.linspace(all_vals.min(), np.percentile(all_vals, 98), bins + 1)

        # ---- PNG ----
        n_models = len(model_values)
        fig, axes = plt.subplots(
            n_models, 1, figsize=(10, 3 * n_models), dpi=150,
            sharex=True, squeeze=False,
        )
        axes = axes.ravel()

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for ax, (name, vals) in zip(axes, model_values.items()):
            counts, _ = np.histogram(vals, bins=bin_edges)
            ax.bar(centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.9,
                   alpha=0.7, label=name)
            ax.set_ylabel("count")
            ax.set_title(name, fontsize=9)
            mean_val = np.mean(vals)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1,
                       label=f"mean = {mean_val:.1f}")
            ax.legend(fontsize=7)

        axes[-1].set_xlabel(f"{metric} (success only)")
        fig.suptitle(f"Evaluation — {metric}{title_extra}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_root / f"histogram_{metric}.png")
        plt.close(fig)

        hist_rows = []
        for name, vals in model_values.items():
            counts, _ = np.histogram(vals, bins=bin_edges)
            for j in range(len(counts)):
                hist_rows.append({
                    "model": name,
                    "bin_left": float(bin_edges[j]),
                    "bin_right": float(bin_edges[j + 1]),
                    "count": int(counts[j]),
                })
        pd.DataFrame(hist_rows).to_csv(
            out_root / f"histogram_{metric}.csv", index=False,
        )

    def _save_progress(self, out_root: Path, traj_num: int) -> None:
        """Save intermediate summary to a progress CSV."""
        progress_dir = out_root / "progress"
        progress_dir.mkdir(exist_ok=True)

        rows = []
        for name, model_rows in self.per_model_rows.items():
            summ = self._summarize_rows(model_rows)
            summ["model_name"] = name
            rows.append(summ)

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate XGBoost vessel-routing models and baselines",
    )
    p.add_argument(
        "--grid", type=str, default="fine",
        choices=["fine", "coarse"],
        help="Grid resolution (default: fine)",
    )
    args = p.parse_args()

    cfg = EvalConfig(grid=GridResolution(args.grid))

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )

    evaluator = Evaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
