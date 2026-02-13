"""SHAP explainability for XGBoost vessel-routing models.

Works on **predicted** (rollout) paths — not ground-truth.  Reuses the
``Inference`` engine and ``Rollout`` class from the shipping-route-predictor
pipeline so that features, environments, and models are always in sync.

Called from ``inference.py`` when ``--shap`` is passed::

    python -m shipping_route_predictor.inference --shap \\
        --grid coarse --model_types simple_cisc full_cisc

Or programmatically::

    analyser = ShapAnalysis(envs, grid, cfg)
    analyser.run(inference_results, trajectory, start_label, goal_label)
    analyser.save()
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb

from shipping_route_predictor.config import (
    GridResolution,
    ModelType,
    RolloutConfig,
    SHAPConfig,
    TrajectorySpec,
)
from shipping_route_predictor.env import Action, EnvGraph
from shipping_route_predictor.utils import make_serialisable
from shipping_route_predictor.visualize import save_shap_route_timeline

log = logging.getLogger("shap_xgb")


# ======================================================================
# ShapAnalysis
# ======================================================================

class ShapAnalysis:
    """SHAP explainability wrapper for XGBoost vessel-routing models.

    Holds loaded environments / models, runs TreeExplainer on predicted
    paths, and persists results (CSVs, JSONs, plots).

    Parameters
    ----------
    envs : dict
        ``{ModelType: (env, xgb_model, rollout_cfg)}`` — the loaded
        environments and models from the ``Inference`` instance.
    grid : GridResolution
        Grid resolution (used for the output directory name).
    cfg : SHAPConfig, optional
        Overrides for output directory, background samples, etc.
        A default ``SHAPConfig()`` is created if omitted.
    """

    def __init__(
        self,
        envs: Dict[ModelType, Tuple[EnvGraph, xgb.XGBClassifier, RolloutConfig]],
        grid: GridResolution,
        cfg: Optional[SHAPConfig] = None,
    ) -> None:
        self.envs = envs
        self.grid = grid
        self.cfg = cfg or SHAPConfig(grid=grid)
        self.results: Dict[str, Dict[str, Any]] = {}  # populated by run()
        self._spec: Optional[TrajectorySpec] = None

    # ── public API ────────────────────────────────────────────

    def run(
        self,
        inference_results: Dict[str, Dict[str, Any]],
        trajectory: TrajectorySpec,
        start_label: str,
        goal_label: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Run SHAP on every XGBoost model in *inference_results*.

        Populates ``self.results`` and returns the same dict.

        Parameters
        ----------
        inference_results
            ``{model_name: record}`` as returned by ``Inference.predict()``.
        trajectory
            The ``TrajectorySpec`` with vessel metadata / start time.
        start_label, goal_label
            Human-readable labels for the route endpoints.
        """
        import shap

        first_record = next(iter(inference_results.values()))
        self._spec = TrajectorySpec(
            name=f"{start_label}\u2192{goal_label}",
            start_rc=tuple(first_record["start"]["rc"]),
            goal_rc=tuple(first_record["goal"]["rc"]),
            start_time=trajectory.start_time,
            vessel_loa=trajectory.vessel_loa,
            vessel_company=trajectory.vessel_company,
        )

        all_shap: Dict[str, Dict[str, Any]] = {}

        for run_name, record in inference_results.items():
            mt_str = record.get("model_type", "")
            try:
                mt = ModelType(mt_str)
            except ValueError:
                log.debug("Skipping %s (not an XGBoost run)", run_name)
                continue

            if mt not in self.envs:
                log.debug("Skipping %s (no env loaded for %s)", run_name, mt)
                continue

            env, xgb_model, _rcfg = self.envs[mt]
            grid_path = [tuple(rc) for rc in record.get("grid_path", [])]

            if len(grid_path) < 2:
                log.warning("Skipping %s: predicted path < 2 cells", run_name)
                continue

            log.info("SHAP: extracting features for %s (%d steps) \u2026", run_name, len(grid_path))
            X, step_info = self._extract_features(env, self._spec, grid_path)
            if X.size == 0:
                log.warning("Skipping %s: no features extracted", run_name)
                continue

            feature_names = env.feature_names

            # ── TreeExplainer ─────────────────────────────────────
            bg_n = self.cfg.background_samples
            bg = X if X.shape[0] <= bg_n else X[
                np.random.choice(X.shape[0], bg_n, replace=False)
            ]
            explainer = shap.TreeExplainer(xgb_model, data=bg)

            log.info(
                "SHAP: computing values for %s (%d samples \u00d7 %d features) \u2026",
                run_name, X.shape[0], X.shape[1],
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = explainer.shap_values(X)

            mean_abs_shap, shap_per_sample, shap_for_plot = self._normalize_shap(shap_values)

            # ── Feature importance ────────────────────────────────
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=False)

            log.info("SHAP top-10 for %s:", run_name)
            for _, row in importance_df.head(10).iterrows():
                log.info("  %-35s %.4f", row["feature"], row["mean_abs_shap"])

            category_df = self._aggregate_categories(feature_names, mean_abs_shap, mt)

            # ── Per-step category SHAP ────────────────────────────
            categories = self.cfg.feature_categories(mt)
            cat_names = list(categories.keys())
            step_cat_shap = []   # (n_steps, n_categories)
            for sv in shap_per_sample:
                cat_vals = []
                for cat, pats in categories.items():
                    cat_mask = np.array([
                        any(p in fn for p in pats)
                        for fn in feature_names
                    ])
                    cat_vals.append(float(np.mean(np.abs(sv[cat_mask]))) if cat_mask.any() else 0.0)
                step_cat_shap.append(cat_vals)

            # ── Attach per-step SHAP ──────────────────────────────
            for si, sv in zip(step_info, shap_per_sample):
                si["shap_values"] = sv.tolist()
                si["shap_mean_abs"] = float(np.mean(np.abs(sv)))

            all_shap[run_name] = {
                "X": X,
                "shap_values": shap_values,
                "shap_for_plot": shap_for_plot,
                "shap_per_sample": shap_per_sample,
                "mean_abs_shap": mean_abs_shap,
                "feature_names": feature_names,
                "importance_df": importance_df,
                "category_importance": category_df,
                "step_info": step_info,
                "grid_path": grid_path,
                "explainer": explainer,
                "step_category_shap": np.array(step_cat_shap),  # (n_steps, n_cats)
                "category_names": cat_names,
                "env": env,
            }

        self.results = all_shap
        return all_shap

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Persist SHAP results to disk.

        Creates one sub-directory per model with:
        ``feature_importance.csv``, ``category_importance.csv``,
        ``feature_names.json``, ``step_shap.json``,
        ``shap_summary.png``, ``shap_bar.png``, ``shap_category.png``.

        Parameters
        ----------
        output_dir : Path, optional
            Where to save.  Falls back to
            ``SHAPConfig().output_dir / <grid>_<timestamp>``.

        Returns
        -------
        Path
            The directory where results were written.
        """
        if output_dir is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.cfg.output_dir) / f"{self.grid}_{ts}"

        output_dir.mkdir(parents=True, exist_ok=True)

        for run_name, res in self.results.items():
            model_dir = output_dir / run_name.replace("/", "_").replace(" ", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            # CSVs
            res["importance_df"].to_csv(model_dir / "feature_importance.csv", index=False)
            res["category_importance"].to_csv(model_dir / "category_importance.csv", index=False)

            # Feature names
            with open(model_dir / "feature_names.json", "w") as f:
                json.dump(res["feature_names"], f, indent=2)

            # Per-step SHAP detail
            serialisable = make_serialisable({
                "model": run_name,
                "grid_path": res["grid_path"],
                "steps": res["step_info"],
                "trajectory": {
                    "name": self._spec.name if self._spec else "",
                    "start_rc": list(self._spec.start_rc) if self._spec else [],
                    "goal_rc": list(self._spec.goal_rc) if self._spec else [],
                    "start_time": self._spec.start_time if self._spec else "",
                    "vessel_loa": self._spec.vessel_loa if self._spec else 0.0,
                    "vessel_company": self._spec.vessel_company if self._spec else "",
                },
            })
            with open(model_dir / "step_shap.json", "w") as f:
                json.dump(serialisable, f, indent=2, default=str)

            # Route + SHAP timeline combined plot
            _env = res["env"]
            _start_ll = _env.grid_indices_to_latlon(self._spec.start_rc) if self._spec else (0.0, 0.0)
            _goal_ll = _env.grid_indices_to_latlon(self._spec.goal_rc) if self._spec else (0.0, 0.0)
            _start_lbl = self._spec.name.split("\u2192")[0].strip() if self._spec else ""
            _goal_lbl = self._spec.name.split("\u2192")[1].strip() if self._spec and "\u2192" in self._spec.name else ""
            save_shap_route_timeline(
                step_info=res["step_info"],
                step_category_shap=res["step_category_shap"],
                category_names=res["category_names"],
                start_latlon=_start_ll,
                goal_latlon=_goal_ll,
                start_label=_start_lbl,
                goal_label=_goal_lbl,
                output_path=model_dir / "shap_route_timeline.png",
                title=f"Route & Category SHAP Timeline \u2014 {run_name}",
                env=_env,
            )

            log.info("SHAP results saved to %s", model_dir)

        log.info("All SHAP results saved to %s", output_dir)
        return output_dir

    # ── static / private helpers ───────────────────────────────

    @staticmethod
    def _extract_features(
        env: EnvGraph,
        spec: TrajectorySpec,
        grid_path: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Walk *grid_path* through *env* and collect features at each step.

        The environment is reset with *spec* so that vessel metadata,
        time, and (for full models) weather / history state are correctly
        initialised.  ``env.build_input_features()`` is called *before*
        each step, producing the same feature vector the model saw.
        """
        if len(grid_path) < 2:
            return np.empty((0, 0)), []

        try:
            env.reset(spec)
        except ValueError as exc:
            log.warning("env.reset failed for %s: %s", spec.name, exc)
            return np.empty((0, 0)), []

        features_list: List[np.ndarray] = []
        step_info: List[Dict[str, Any]] = []

        for i in range(len(grid_path) - 1):
            cur = grid_path[i]
            nxt = grid_path[i + 1]

            if env.position != cur:
                log.warning(
                    "%s step %d: env at %s, expected %s \u2014 stopping",
                    spec.name, i, env.position, cur,
                )
                break

            feat = env.build_input_features()
            features_list.append(feat)

            lat, lon = env.grid_indices_to_latlon(cur)
            step_info.append({
                "step": i,
                "cur_rc": list(cur),
                "cur_lat": lat,
                "cur_lon": lon,
            })

            # Advance environment
            action_idx = env.action_from_transition(cur, nxt)
            if action_idx is None:
                log.warning(
                    "%s step %d: no valid action %s \u2192 %s \u2014 stopping",
                    spec.name, i, cur, nxt,
                )
                break
            try:
                env.step(Action.from_index(action_idx))
            except ValueError as exc:
                log.warning("%s step %d: env.step failed \u2014 %s", spec.name, i, exc)
                break

        if not features_list:
            return np.empty((0, 0)), []
        return np.stack(features_list), step_info

    @staticmethod
    def _normalize_shap(shap_values):
        """Return ``(mean_abs_shap, shap_per_sample, shap_for_plot)``."""
        if isinstance(shap_values, list):
            arr = np.stack(shap_values, axis=0)          # (C, N, D)
            mean_abs = np.mean(np.abs(arr), axis=(0, 1))
            per_sample = np.mean(np.abs(arr), axis=0)
            for_plot = arr[0]
        elif shap_values.ndim == 3:
            mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))
            per_sample = np.mean(np.abs(shap_values), axis=2)
            for_plot = shap_values[:, :, 0]
        else:
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            per_sample = np.abs(shap_values)
            for_plot = shap_values

        if mean_abs.ndim > 1:
            mean_abs = mean_abs.flatten()
        return mean_abs, per_sample, for_plot

    def _aggregate_categories(
        self,
        feature_names: List[str],
        mean_abs_shap: np.ndarray,
        model_type: ModelType,
    ) -> pd.DataFrame:
        """Group features into semantic categories and average importance."""
        patterns = self.cfg.feature_categories(model_type)
        rows: List[Dict[str, Any]] = []
        for cat, pats in patterns.items():
            mask = np.zeros(len(feature_names), dtype=bool)
            for i, fn in enumerate(feature_names):
                for p in pats:
                    if p in fn:
                        mask[i] = True
                        break
            if mask.any():
                rows.append({"category": cat, "mean_abs_shap": float(np.mean(mean_abs_shap[mask]))})
        return pd.DataFrame(sorted(rows, key=lambda r: -r["mean_abs_shap"]))
