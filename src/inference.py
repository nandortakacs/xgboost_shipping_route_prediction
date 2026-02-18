"""Single-route inference CLI for XGBoost vessel-routing models.

Predicts a route from a start to a goal (given as lat/lon or port name),
runs one or more XGBoost models and baselines, then saves:

* per-model JSON with step indices, lat/lon, estimated times, and ETA
* a Cartopy route map showing all predicted routes on real geography

Usage examples
--------------
Default (Shanghai → Long Beach, simple_greedy + shortest_path)::

    python -m shipping_route_predictor.inference

Custom route with specific models::

    python -m shipping_route_predictor.inference \\
        --start_port USLAX --goal_port KRPUS \\
        --model_types simple_greedy full_cisc company_baseline

All models on coarse grid::

    python -m shipping_route_predictor.inference --grid coarse \\
        --model_types simple_greedy simple_cisc full_greedy full_cisc \\
                      shortest_path company_baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xgboost as xgb

from shipping_route_predictor.baselines import CompanyBaseline, ShortestPathBaseline
from shipping_route_predictor.config import (
    DEFAULT_SPEED_KMH,
    KNOT_TO_KMH,
    EnvConfig,
    GridResolution,
    InferenceConfig,
    ModelSpec,
    ModelType,
    RolloutConfig,
    RolloutType,
    TrajectorySpec,
    WeatherConfig,
)
from shipping_route_predictor.data import EnvDataset, load_operator_speed_table, lookup_vessel_speed_kmh
from shipping_route_predictor.env import EnvGraph, build_env
from shipping_route_predictor.rollout import Rollout
from shipping_route_predictor.utils import (
    estimate_times,
    haversine_km,
    total_path_distance_km,
)
from shipping_route_predictor.visualize import RouteVisualizer

log = logging.getLogger("inference")


class Inference:
    """Run one or more XGBoost models + baselines on a single route.

    Parameters
    ----------
    cfg : InferenceConfig
        Full inference configuration (grid, runners, trajectory, speed, etc.).
    """

    def __init__(self, cfg: Optional[InferenceConfig] = None) -> None:
        self.cfg = cfg or InferenceConfig()
        self.grid = self.cfg.grid
        self.speed_kmh = self.cfg.speed_kmh

        # Parse runner names into xgb combos + baseline flags
        self._xgb_runs: List[Tuple[ModelType, RolloutType]] = []
        needed_model_types: set = set()
        self.include_shortest_path = False
        self.include_company_baseline = False

        for name in self.cfg.runners:
            if name == "shortest_path":
                self.include_shortest_path = True
            elif name == "company_baseline":
                self.include_company_baseline = True
            else:
                mt_str, rt_str = name.split("_", 1)
                mt, rt = ModelType(mt_str), RolloutType(rt_str)
                self._xgb_runs.append((mt, rt))
                needed_model_types.add(mt)

        # Build environments and load XGBoost models eagerly
        self._envs: Dict[ModelType, Tuple[EnvGraph, xgb.XGBClassifier, RolloutConfig]] = {}
        self.env_cfg: EnvConfig = EnvConfig(grid=self.grid)

        for mt in needed_model_types:
            weather = WeatherConfig(grid=self.grid) if mt == ModelType.FULL else None
            env = build_env(EnvDataset(self.env_cfg, mt, weather))
            rcfg = RolloutConfig(
                grid=self.grid,
                model=ModelSpec(name=f"xgb_{self.grid}_{mt}", model_type=mt),
            )
            model = xgb.XGBClassifier()
            model.load_model(rcfg.model.path)
            log.info("Loaded model: %s from %s", rcfg.model.name, rcfg.model.path)
            self._envs[mt] = (env, model, rcfg)

        # Keep an env for coordinate conversion (build one if only baselines)
        if self._envs:
            self.env: EnvGraph = next(iter(self._envs.values()))[0]
        else:
            self.env = build_env(EnvDataset(self.env_cfg, ModelType.SIMPLE, None))

        # Load operator speed table once for per-vessel speed resolution
        self._speed_table, self._speed_loa_bins = load_operator_speed_table(self.env_cfg)

    def predict(
        self,
        start_latlon: Tuple[float, float],
        goal_latlon: Tuple[float, float],
        start_label: str,
        goal_label: str,
        start_time: str,
        vessel_loa: float,
        vessel_company: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Predict a route with all configured models + baselines.

        Returns ``{model_name: result_dict}`` for each run.

        Raises
        ------
        ValueError
            If any of the supplied arguments are out of expected range.
        """
        for name, latlon in [("start", start_latlon), ("goal", goal_latlon)]:
            lat, lon = latlon
            if not (-90 <= lat <= 90):
                raise ValueError(f"{name} latitude {lat} out of range [-90, 90]")
            if not (-180 <= lon <= 180):
                raise ValueError(f"{name} longitude {lon} out of range [-180, 180]")

        if not start_time:
            raise ValueError("start_time must be a non-empty ISO 8601 string")

        start_rc = self.env.coords.latlon_to_grid_indices(*start_latlon)
        goal_rc = self.env.coords.latlon_to_grid_indices(*goal_latlon)
        actual_start = self.env.grid_indices_to_latlon(start_rc)
        actual_goal = self.env.grid_indices_to_latlon(goal_rc)
        route_km = haversine_km(*actual_start, *actual_goal)

        log.info("Route: %s → %s  (%.1f km great-circle)", start_label, goal_label, route_km)
        log.info("  Start cell: %s → (%.2f°, %.2f°)", start_rc, *actual_start)
        log.info("  Goal  cell: %s → (%.2f°, %.2f°)", goal_rc, *actual_goal)

        spec = TrajectorySpec(
            name=f"{start_label}→{goal_label}",
            start_rc=start_rc,
            goal_rc=goal_rc,
            start_time=start_time,
            vessel_loa=vessel_loa,
            vessel_company=vessel_company,
            vessel_speed_kmh=lookup_vessel_speed_kmh(
                vessel_company, vessel_loa,
                self._speed_table, self._speed_loa_bins,
            ),
        )

        all_results: Dict[str, Dict[str, Any]] = {}

        for mt, rt in self._xgb_runs:
            mt_env, xgb_model, rcfg = self._envs[mt]
            runner = Rollout(rcfg, xgb_model, mt_env)
            run_name = f"{mt}_{rt}"
            log.info("Running %s ...", run_name)
            t0 = time.perf_counter()
            result, latlon_path = runner.run_route(spec, rt)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if result is None or latlon_path is None:
                log.error("  %s failed", run_name)
                continue

            total_km = total_path_distance_km(latlon_path)
            waypoints = estimate_times(latlon_path, start_time, self.speed_kmh)
            eta = waypoints[-1]["time_iso"] if waypoints else None

            log.info(
                "  %s: %d steps, reached=%s, %.1f km, %.0f ms, ETA=%s",
                run_name, len(result["grid_path"]),
                result["reached_goal"], total_km, elapsed_ms, eta,
            )

            all_results[run_name] = self._build_record(
                run_name, str(mt), str(rt), spec,
                actual_start, actual_goal,
                result, total_km, waypoints, eta, elapsed_ms,
            )

        baselines: List[Tuple[str, str, Any]] = []
        if self.include_shortest_path:
            baselines.append(("shortest_path", "dijkstra", ShortestPathBaseline(self.env_cfg)))
        if self.include_company_baseline:
            try:
                baselines.append(("company_baseline", "company", CompanyBaseline(self.env_cfg)))
            except Exception as exc:
                log.error("  Company baseline init failed: %s", exc)

        for run_name, rollout_label, baseline in baselines:
            log.info("Running %s ...", run_name)
            t0 = time.perf_counter()
            result, latlon_path = baseline.predict_route(spec.start_rc, spec.goal_rc)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if result is None or latlon_path is None:
                log.warning("  %s returned no path.", run_name)
                continue

            total_km = total_path_distance_km(latlon_path)
            waypoints = estimate_times(latlon_path, start_time, self.speed_kmh)
            eta = waypoints[-1]["time_iso"] if waypoints else None

            log.info("  %s: %d steps, %.1f km, %.0f ms, ETA=%s",
                     run_name, len(result["grid_path"]), total_km, elapsed_ms, eta)

            all_results[run_name] = self._build_record(
                run_name, rollout_label, rollout_label, spec,
                actual_start, actual_goal,
                result, total_km, waypoints, eta, elapsed_ms,
            )

        return all_results

    def save_results(
        self,
        results: Dict[str, Dict[str, Any]],
        output_dir: Path,
        *,
        visualize: bool = True,
        start_label: str = "",
        goal_label: str = "",
        start_time: str = "",
    ) -> Path:
        """Persist all results to *output_dir*.

        Saves per-model JSONs, a summary JSON, and optionally a route map.
        Returns *output_dir*.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Per-model JSONs
        for name, record in results.items():
            with open(output_dir / f"{name}.json", "w") as f:
                json.dump(record, f, indent=2)
            log.info("  Saved %s", output_dir / f"{name}.json")

        # Summary
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route": f"{start_label} → {goal_label}",
            "grid": str(self.grid),
            "start_time": start_time,
            "speed_kmh": self.speed_kmh,
            "models": {
                name: {
                    "reached_goal": r["reached_goal"],
                    "num_steps": r["num_steps"],
                    "total_distance_km": r["total_distance_km"],
                    "eta": r["eta"],
                    "elapsed_ms": r["elapsed_ms"],
                }
                for name, r in results.items()
            },
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Route map
        if visualize and results:
            routes_latlon: Dict[str, List[Tuple[float, float]]] = {}
            for name, record in results.items():
                wps = record.get("waypoints", [])
                if wps:
                    routes_latlon[name] = [(w["lat"], w["lon"]) for w in wps]

            first = next(iter(results.values()))
            actual_start = (first["start"]["lat"], first["start"]["lon"])
            actual_goal = (first["goal"]["lat"], first["goal"]["lon"])
            title_str = (
                f"{start_label} → {goal_label}  |  "
                f"{self.grid} grid  |  departure {start_time[:10]}"
            )

            viz = RouteVisualizer(self.env)
            viz.save(
                routes=routes_latlon,
                start_latlon=actual_start,
                goal_latlon=actual_goal,
                start_label=start_label,
                goal_label=goal_label,
                output_path=output_dir / "route_map.png",
                title=title_str,
            )

        log.info("All results saved to %s", output_dir)
        return output_dir

    def run_shap(
        self,
        results: Dict[str, Dict[str, Any]],
        *,
        background_samples: int = 200,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Run SHAP analysis on the predicted paths and save results.

        Creates a :class:`~shipping_route_predictor.explainability_shap.ShapAnalysis`
        instance, runs it, and saves the output.

        Parameters
        ----------
        results
            ``{model_name: record}`` as returned by :meth:`predict`.
        background_samples
            Max background samples for the SHAP TreeExplainer.
        output_dir
            Where to save.  Uses ``SHAPConfig.output_dir`` default
            (``results/shap_results/<grid>_<timestamp>``) if *None*.

        Returns
        -------
        Path
            The directory where SHAP results were saved.
        """
        from shipping_route_predictor.explainability_shap import ShapAnalysis
        from shipping_route_predictor.config import SHAPConfig

        cfg = SHAPConfig(grid=self.grid, background_samples=background_samples)
        analyser = ShapAnalysis(self._envs, self.grid, cfg)
        analyser.run(
            results,
            trajectory=self.cfg.trajectory,
            start_label=self.cfg.start_label,
            goal_label=self.cfg.goal_label,
        )
        return analyser.save(output_dir)

    def _build_record(
        self,
        run_name: str,
        model_type: str,
        rollout_type: str,
        spec: TrajectorySpec,
        actual_start: Tuple[float, float],
        actual_goal: Tuple[float, float],
        result: Dict[str, Any],
        total_km: float,
        waypoints: List[Dict[str, Any]],
        eta: Optional[str],
        elapsed_ms: float,
    ) -> Dict[str, Any]:
        """Construct a single inference-result record."""
        return {
            "model": run_name,
            "grid": str(self.grid),
            "model_type": model_type,
            "rollout_type": rollout_type,
            "start": {
                "label": spec.name.split("→")[0] if "→" in spec.name else "",
                "lat": actual_start[0],
                "lon": actual_start[1],
                "rc": list(spec.start_rc),
            },
            "goal": {
                "label": spec.name.split("→")[1] if "→" in spec.name else "",
                "lat": actual_goal[0],
                "lon": actual_goal[1],
                "rc": list(spec.goal_rc),
            },
            "start_time": spec.start_time,
            "vessel_loa_m": spec.vessel_loa,
            "vessel_company": spec.vessel_company,
            "speed_kmh": self.speed_kmh,
            "reached_goal": bool(result.get("reached_goal", False)),
            "num_steps": len(result.get("grid_path", [])),
            "total_distance_km": round(total_km, 1),
            "elapsed_ms": round(elapsed_ms, 1),
            "mean_prob": result.get("mean_prob"),
            "eta": eta,
            "waypoints": waypoints,
            "grid_path": [list(rc) for rc in result["grid_path"]],
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict a vessel route using XGBoost models and baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (Shanghai → Long Beach, simple_greedy + shortest_path)
  python -m shipping_route_predictor.inference

  # Specific models on coarse grid
  python -m shipping_route_predictor.inference --grid coarse \\
      --model_types simple_greedy full_cisc company_baseline

  # All models
  python -m shipping_route_predictor.inference --model_types %(all)s

  # Custom route by port name
  python -m shipping_route_predictor.inference --start_port USLAX --goal_port KRPUS

  # Custom route by lat/lon
  python -m shipping_route_predictor.inference \\
      --start_lat 1.3 --start_lon 103.8 --goal_lat 55.0 --goal_lon 12.5
""" % {"all": " ".join(InferenceConfig.VALID_RUNNERS)},
    )

    # Start location
    start_g = p.add_argument_group("Start location (port name OR lat/lon)")
    start_g.add_argument("--start_port", type=str, default="CNSHA",
                         help="Start port name or LOCODE (default: CNSHA / Shanghai)")
    start_g.add_argument("--start_lat", type=float, default=None,
                         help="Start latitude (overrides --start_port)")
    start_g.add_argument("--start_lon", type=float, default=None,
                         help="Start longitude (overrides --start_port)")

    # Goal location
    goal_g = p.add_argument_group("Goal location (port name OR lat/lon)")
    goal_g.add_argument("--goal_port", type=str, default="USLAX",
                        help="Goal port name or LOCODE (default: USLAX / Los Angeles)")
    goal_g.add_argument("--goal_lat", type=float, default=None,
                        help="Goal latitude (overrides --goal_port)")
    goal_g.add_argument("--goal_lon", type=float, default=None,
                        help="Goal longitude (overrides --goal_port)")

    # Grid & models
    p.add_argument("--grid", type=str, default="coarse", choices=["fine", "coarse"],
                   help="Grid resolution (default: coarse)")
    p.add_argument("--model_types", nargs="+", default=list(InferenceConfig.DEFAULT_RUNNERS),
                   choices=InferenceConfig.VALID_RUNNERS, metavar="MODEL",
                   help="Models/baselines to run (default: %(default)s). "
                        f"Valid: {', '.join(InferenceConfig.VALID_RUNNERS)}")

    # Time & vessel
    p.add_argument("--start_time", type=str, default=None,
                   help="Departure time ISO 8601 (default: from trajectory config)")
    p.add_argument("--vessel_loa", type=float, default=None,
                   help="Vessel length overall in metres (default: from trajectory config)")
    p.add_argument("--vessel_company", type=str, default=None,
                   help="Vessel operator (default: from trajectory config)")
    p.add_argument("--speed_knots", type=float, default=None,
                   help="Vessel speed in knots (default: 14.5 kn)")

    # Output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (auto-generated if omitted)")
    p.add_argument("--no_visualize", action="store_true",
                   help="Skip route map generation")
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # SHAP explainability
    p.add_argument("--shap", action="store_true", default=False,
                   help="Run SHAP explainability on predicted paths (saves to results/shap_results)")
    p.add_argument("--shap_background_samples", type=int, default=200,
                   help="Background samples for SHAP TreeExplainer (default: 200)")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )

    # Build InferenceConfig — all resolution happens in __post_init__
    grid = GridResolution(args.grid)
    speed_kmh = (args.speed_knots * KNOT_TO_KMH) if args.speed_knots else DEFAULT_SPEED_KMH

    cfg = InferenceConfig(
        grid=grid,
        runners=args.model_types,
        speed_kmh=speed_kmh,
        start_port=args.start_port,
        start_lat=args.start_lat,
        start_lon=args.start_lon,
        goal_port=args.goal_port,
        goal_lat=args.goal_lat,
        goal_lon=args.goal_lon,
        start_time_override=args.start_time,
        vessel_loa_override=args.vessel_loa,
        vessel_company_override=args.vessel_company,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
        log_level=args.log_level,
    )

    # Run
    engine = Inference(cfg)
    traj = cfg.trajectory

    # Resolve start/goal lat-lon (may need env for grid-cell fallback)
    start = cfg.start_latlon or engine.env.grid_indices_to_latlon(traj.start_rc)
    goal = cfg.goal_latlon or engine.env.grid_indices_to_latlon(traj.goal_rc)

    results = engine.predict(
        start_latlon=start,
        goal_latlon=goal,
        start_label=cfg.start_label,
        goal_label=cfg.goal_label,
        start_time=traj.start_time,
        vessel_loa=traj.vessel_loa,
        vessel_company=traj.vessel_company,
    )

    engine.save_results(
        results,
        Path(cfg.output_dir),
        visualize=cfg.visualize,
        start_label=cfg.start_label,
        goal_label=cfg.goal_label,
        start_time=traj.start_time,
    )

    if args.shap and results:
        engine.run_shap(results, background_samples=args.shap_background_samples)


if __name__ == "__main__":
    main()
