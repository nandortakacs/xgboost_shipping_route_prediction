"""Data loading and grid-coordinate helpers for the shipping-route-predictor pipeline.

Three main public names:

* :class:`GridCoordinates` — lightweight bidirectional mapping between
  ``(lat, lon)`` and grid ``(row, col)`` indices.
* :class:`EnvDataset` — reads grid CSV files and exposes the raw arrays
  needed to construct an ``EnvGraph`` (but does **not** build one itself).
* :class:`AISDataset` — loads AIS trajectory JSONs, filters them,
  and extracts tabular (X, y) feature–action samples via the env.

``GridCoordinates`` and ``EnvDataset`` are defined *before* the heavy
``env`` / ``baselines`` imports so that those modules can safely import
them without circular-dependency issues.
"""
from __future__ import annotations

import gc
import json
import logging
import os
from datetime import datetime as _dt
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np

from shipping_route_predictor.config import EnvConfig, KNOT_TO_KMH, ModelType, WeatherConfig, DEFAULT_SPEED_KMH




log = logging.getLogger("data")


class GridCoordinates:
    """Bidirectional mapping between ``(lat, lon)`` and grid indices
    ``(row, col)`` for a cropped H×W navigation grid.

    Reads the grid CSV files referenced by an :class:`EnvConfig`
    (mask, latitude, longitude) and exposes conversion helpers used
    throughout the pipeline.

    Parameters
    ----------
    env_cfg : EnvConfig
        Must carry populated ``mask_csv``, ``latitude_csv``, and
        ``longitude_csv`` paths (auto-filled by
        ``EnvConfig.__post_init__``).
    """

    def __init__(self, env_cfg: EnvConfig) -> None:
        mask_raw = np.loadtxt(env_cfg.mask_csv, delimiter=",", dtype=float) > 0.5
        lons = self._read_csv_column(env_cfg.longitude_csv, "longitude")

        lat_idx = self._read_csv_column(env_cfg.latitude_csv, "lat_idx").astype(np.int64)
        lats = self._read_csv_column(env_cfg.latitude_csv, "latitude")
        order = np.argsort(lat_idx)
        lat_idx, lats = lat_idx[order], lats[order]

        start, stop = int(lat_idx[0]), int(lat_idx[-1]) + 1
        self.mask: np.ndarray = mask_raw[start:stop, :]
        self.H: int = self.mask.shape[0]
        self.W: int = self.mask.shape[1]
        self.lats: np.ndarray = np.asarray(lats, dtype=np.float32).ravel()
        self.lons: np.ndarray = np.asarray(lons, dtype=np.float32).ravel()

    # ── conversions ───────────────────────────────────────────

    def grid_indices_to_latlon(
        self,
        indices: Tuple[int, int],
    ) -> Tuple[float, float]:
        """``(row, col)`` → ``(lat, lon)`` in degrees."""
        r, c = indices[0], indices[1] % self.W
        return float(self.lats[r]), float(self.lons[c])

    def latlon_to_grid_indices(
        self,
        lat: float,
        lon: float,
        *,
        snap_to_water: bool = True,
    ) -> Tuple[int, int]:
        """``(lat, lon)`` → nearest grid cell ``(row, col)``.

        When *snap_to_water* is ``True`` (default) and the nearest cell
        is land, BFS-expand outward until a navigable cell is found.

        Raises
        ------
        ValueError
            If no navigable cell can be found within half the grid size.
        """
        lon_norm = ((lon + 180) % 360) - 180
        row = int(np.argmin(np.abs(self.lats.astype(np.float64) - lat)))
        col = int(np.argmin(np.abs(self.lons.astype(np.float64) - lon_norm)))

        if not snap_to_water or self.mask is None or self.mask[row, col]:
            return (row, col)

        log.warning(
            "(%.2f, %.2f) → cell (%d, %d) is land; searching nearby water…",
            lat, lon, row, col,
        )
        for radius in range(1, max(self.H, self.W) // 2):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) != radius and abs(dc) != radius:
                        continue
                    r2 = row + dr
                    c2 = (col + dc) % self.W
                    if 0 <= r2 < self.H and self.mask[r2, c2]:
                        log.info(
                            "  Found water cell (%d, %d) at radius %d",
                            r2, c2, radius,
                        )
                        return (r2, c2)

        raise ValueError(f"No navigable cell near ({lat}, {lon}).")

    def latlon_to_grid_indices_batch(
        self,
        coords: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """``(N, 2)`` array of ``[lat, lon]`` → list of ``(row, col)``.

        Uses the fast uniform-grid formula.  Consecutive duplicate cells
        are removed.
        """
        rows = np.clip(
            np.floor((coords[:, 0] + 90.0) / 180.0 * self.H).astype(int),
            0, self.H - 1,
        )
        cols = np.clip(
            np.floor((coords[:, 1] + 180.0) / 360.0 * self.W).astype(int),
            0, self.W - 1,
        )
        grid_path: List[Tuple[int, int]] = [(int(rows[0]), int(cols[0]))]
        for r, c in zip(rows[1:], cols[1:]):
            cell = (int(r), int(c))
            if cell != grid_path[-1]:
                grid_path.append(cell)
        return grid_path

    def lonlat_to_grid_indices_batch(
        self,
        coords: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """``(N, 2)`` array of ``[lon, lat]`` → list of ``(row, col)``.

        Convenience wrapper that flips columns from ``[lon, lat]`` to
        ``[lat, lon]`` before delegating to
        :meth:`latlon_to_grid_indices_batch`.
        """
        return self.latlon_to_grid_indices_batch(coords[:, ::-1])

    # ── CSV helper (also used by EnvDataset) ──────────────────

    @staticmethod
    def _read_csv_column(path: str, column_name: str) -> np.ndarray:
        """Read a single named column from a headered CSV → float32 1-D."""
        with open(path, "r", encoding="utf-8") as f:
            header = [c.strip().lower() for c in f.readline().strip().split(",")]
        col_idx = header.index(column_name.strip().lower())
        return np.loadtxt(
            path, delimiter=",", dtype=np.float32,
            skiprows=1, usecols=(col_idx,),
        ).reshape(-1)


class EnvDataset:
    """Reads grid CSV files and exposes the raw arrays needed to build
    an :class:`EnvGraph`.

    This class is a **pure data provider** — it loads mask, depth,
    latitude, and longitude arrays from disk but does *not* construct
    any graph objects.  Pass it to ``build_env(ds)`` which dispatches
    to the right :class:`EnvGraph` subclass based on *model_type*.

    Parameters
    ----------
    env_cfg : EnvConfig
        Grid geometry, file paths, normalisation constants.
    model_type : ModelType
        ``ModelType.FULL`` → a downstream :class:`FullEnvGraph` with
        weather (196-dim features).  ``ModelType.SIMPLE`` → plain
        :class:`EnvGraph` (14-dim features).
    weather_cfg : WeatherConfig, optional
        Explicit weather config for FULL models.  When ``None`` and
        *model_type* is ``FULL``, a default ``WeatherConfig`` is
        created from ``env_cfg.grid``.

    Examples
    --------
    >>> from shipping_route_predictor.env import build_env
    >>> ds  = EnvDataset(EnvConfig(), ModelType.SIMPLE)
    >>> env = build_env(ds)
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        model_type: ModelType = ModelType.SIMPLE,
        weather_cfg: Optional[WeatherConfig] = None,
    ) -> None:
        self.env_cfg = env_cfg
        self.model_type = model_type
        if model_type == ModelType.FULL:
            self.weather_cfg: Optional[WeatherConfig] = weather_cfg or WeatherConfig(grid=env_cfg.grid)
        else:
            self.weather_cfg = None

    # ---- CSV readers (static helpers) — delegated to GridCoordinates ------

    _read_headered_column = staticmethod(GridCoordinates._read_csv_column)

    @staticmethod
    def read_mask_csv(path: str) -> np.ndarray:
        """Read headerless HxW mask CSV.  Returns bool 2-D (True = water)."""
        return np.loadtxt(path, delimiter=",", dtype=float) > 0.5

    @classmethod
    def read_longitude_csv(cls, path: str) -> np.ndarray:
        """Read longitude_indices.csv (header: index,longitude).  Returns float32 1-D of length W."""
        return cls._read_headered_column(path, "longitude")

    @classmethod
    def read_latitude_mapping(cls, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read lat_mapping_table.csv (header includes lat_idx, latitude).

        Returns
        -------
        lat_idx : int64 1-D — original row indices, sorted ascending.
        lats    : float32 1-D — corresponding latitude values.
        """
        idx = cls._read_headered_column(path, "lat_idx").astype(np.int64)
        lats = cls._read_headered_column(path, "latitude")
        order = np.argsort(idx)
        return idx[order], lats[order]

    @staticmethod
    def read_depth_csv(path: str) -> np.ndarray:
        """Read headerless HxW depth CSV.  Returns float32 2-D."""
        return np.loadtxt(path, delimiter=",", dtype=np.float32)

    # ---- public grid loader ------------------------------------------------

    def load_grid_arrays(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read mask, depth, lat, lon CSVs and crop to valid latitude band.

        Returns ``(mask, lats, lons, depth)`` ready for ``EnvGraph.__init__``.
        """
        cfg = self.env_cfg
        full_mask = self.read_mask_csv(cfg.mask_csv)
        full_depth = self.read_depth_csv(cfg.depth_csv)
        lons = self.read_longitude_csv(cfg.longitude_csv)
        lat_idx, lats = self.read_latitude_mapping(cfg.latitude_csv)

        start, stop = int(lat_idx[0]), int(lat_idx[-1]) + 1
        return full_mask[start:stop, :], lats, lons, full_depth[start:stop, :]


# ──────────────────────────────────────────────────────────────────────
# Heavy imports — safe now that GridCoordinates and EnvDataset are
# both fully defined.
# ──────────────────────────────────────────────────────────────────────
from shipping_route_predictor.config import TrajectorySpec  # noqa: E402
from shipping_route_predictor.env import Action, EnvGraph  # noqa: E402

from tqdm import tqdm  # noqa: E402

def _parse_loa_bins(labels: List[str]) -> List[Tuple[float, str]]:
    """Derive ``(upper_bound, label)`` pairs from ``_meta.loa_bins``.

    Bin labels follow the pattern ``"150-200m"`` or ``">400m"``.
    """
    bins: List[Tuple[float, str]] = []
    for label in labels:
        if label.startswith(">"):
            bins.append((float("inf"), label))
        else:
            upper_str = label.rstrip("m").split("-")[-1]
            bins.append((float(upper_str), label))
    return bins


def load_operator_speed_table(
    env_cfg: EnvConfig,
) -> Tuple[Optional[Dict[str, Any]], List[Tuple[float, str]]]:
    """Load the operator speed JSON referenced by *env_cfg*, if it exists.

    Returns ``(table_dict, loa_bins)`` — both are empty/None when the
    file is absent.
    """
    path = env_cfg.operator_speed_json
    if path and os.path.isfile(path):
        with open(path, "r") as fh:
            tbl = json.load(fh)
        bins = _parse_loa_bins(tbl.get("_meta", {}).get("loa_bins", []))
        log.info("Loaded operator speed table from %s", path)
        return tbl, bins
    log.info("Operator speed table not found; using fixed speed %.2f km/h.", DEFAULT_SPEED_KMH)
    return None, []


def lookup_vessel_speed_kmh(
    company: str,
    loa_m: float,
    table: Optional[Dict[str, Any]],
    loa_bins: List[Tuple[float, str]],
) -> float:
    """Return vessel speed (km/h) from the operator table, or fixed fallback.

    Cascading fall-through:
    1. operator + LOA bin   (exact match)
    2. ``"other"`` + LOA bin (generic row)
    3. ``_meta.fallback_fleet_speed_kmh``  (table-level fallback)
    4. ``DEFAULT_SPEED_KMH``               (module-level constant)
    """
    if table is None:
        return DEFAULT_SPEED_KMH

    # Determine LOA bin
    loa_bin = ""
    if loa_m and loa_bins:
        for upper, label in loa_bins:
            if loa_m <= upper:
                loa_bin = label
                break

    company_key = (company or "").strip().lower()

    # 1. exact operator row
    if company_key in table and loa_bin in table[company_key]:
        return table[company_key][loa_bin] * KNOT_TO_KMH

    # 2. generic "other" row
    other = table.get("other", {})
    if loa_bin in other:
        return other[loa_bin] * KNOT_TO_KMH

    # 3. meta fallback
    meta = table.get("_meta", {})
    if "fallback_fleet_speed_kmh" in meta:
        return float(meta["fallback_fleet_speed_kmh"])

    # 4. module-level constant
    return DEFAULT_SPEED_KMH


class AISDataset:
    """Loads, filters, and extracts feature–action samples from AIS
    trajectory JSON files.

    Bundles trajectory parsing (:meth:`load_trajectories`), per-step
    feature extraction (:meth:`get_ground_truth_trajectory`), and the
    full tabular-dataset pipeline (:meth:`get_gt_trajectories`).

    Parameters
    ----------
    env : EnvGraph
        Grid environment (navigability checks, feature extraction,
        action mapping).
    env_cfg : EnvConfig
        Configuration carrying vessel-registry path, trajectory file
        lists, and filtering defaults (``max_traj_days``,
        ``max_coord_jump``, etc.).

    Examples
    --------
    >>> ais = AISDataset(env, cfg.env)
    >>> X, y = ais.get_gt_trajectories(cfg.env.train_track_files, limit=5000, desc="train")
    """

    def __init__(self, env: EnvGraph, env_cfg: EnvConfig) -> None:
        from shipping_route_predictor.baselines import ShortestPathBaseline

        self.env = env
        self.env_cfg = env_cfg
        self._vessel_registry: Optional[Dict[int, Dict[str, Any]]] = None
        self._baseline = ShortestPathBaseline(env_cfg)

        # ---- Operator speed table (optional) ----
        self._operator_speed_table, self._loa_bins = load_operator_speed_table(env_cfg)

    def get_ais_dataset(
        self,
        split: str,
        *,
        limit: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load trajectories and extract tabular ``(X, y)`` arrays.

        Combines :meth:`load_trajectories` and
        :meth:`get_ground_truth_trajectory` over every trajectory.

        Returns
        -------
        X : ndarray, float32, shape (N, D)
            Feature matrix.
        y : ndarray, int32, shape (N,)
            Action labels (0–3).
        """
        trajs = self.load_trajectories(split, limit=limit)
        if desc is None:
            desc = split
        log.info("%s trajectories loaded: %d (limit=%s)", desc, len(trajs), limit)

        X_all: List[np.ndarray] = []
        y_all: List[int] = []
        skipped = 0

        pbar = tqdm(trajs, desc=f"Extract {desc}", unit="traj")
        for spec, rc_path in pbar:
            feats, acts = self.prepare_gt_traj_inputs_outputs(spec, rc_path)
            if not feats:
                skipped += 1
                continue
            X_all.extend(feats)
            y_all.extend(acts)
            pbar.set_postfix(samples=len(X_all), skip=skipped)

        if not X_all:
            raise RuntimeError(
                f"No samples extracted for {desc} — check trajectory files and grid."
            )

        X = np.stack(X_all, axis=0).astype(np.float32)
        y = np.asarray(y_all, dtype=np.int32)

        log.info(
            "%s data: %d samples, %d features (%d trajs skipped)",
            desc, X.shape[0], X.shape[1], skipped,
        )

        gc.collect()
        return X, y
        # --- Debug sample dump ---
        # try:
        #     debug_path = os.path.join(
        #         self.env_cfg.project_dir,
        #         "debug_feature_samples",
        #         f"{desc}_feature_samples.csv",
        #     )
        #     self._dump_debug_feature_samples(X, y, debug_path)
        # except Exception as exc:
        #     log.warning("Debug feature dump failed: %s", exc)

        # gc.collect()
        # return X, y

    def load_trajectories(
        self,
        split: str,
        *,
        limit: Optional[int] = None,
        max_traj_days: Optional[float] = None,
    ) -> List[Tuple[TrajectorySpec, List[Tuple[int, int]]]]:
        """Parse trajectory JSONs into ``(TrajectorySpec, rc_path)`` pairs.

        Each trajectory is converted from ``(lon_idx, lat_idx)`` in the
        JSON to ``(row, col)`` = ``(lat_idx, lon_idx)`` and filtered:

        * Non-navigable start/goal cells are skipped.
        * Coordinate jumps > ``env_cfg.max_coord_jump`` cause rejection.
        * Trajectories longer than *max_traj_days* are skipped.
        * Consecutive duplicate cells are removed; paths < 2 dropped.

        Parameters
        ----------
        split : ``"train"`` | ``"val"`` | ``"eval"``
            Dataset split — resolved to file list via ``env_cfg``.
        limit : int, optional
            Stop after collecting this many trajectories.
        max_traj_days : float, optional
            Maximum passage duration in days.  Falls back to
            ``self.env_cfg.max_traj_days`` when ``None``.

        Raises
        ------
        FileNotFoundError
            If any JSON file does not exist.
        """
        json_files = self._get_track_files(split)
        if max_traj_days is None:
            max_traj_days = self.env_cfg.max_traj_days
        max_coord_jump = self.env_cfg.max_coord_jump

        # -- inner helpers (closures over max_traj_days / max_coord_jump) ----

        def _exceeds_duration(rec: dict) -> bool:
            """Return True when the passage exceeds *max_traj_days*."""
            start_str = rec.get("passage_start", "")
            end_str = rec.get("passage_end", "")
            if not start_str or not end_str:
                return False
            t0 = _dt.fromisoformat(start_str.replace("Z", "+00:00"))
            t1 = _dt.fromisoformat(end_str.replace("Z", "+00:00"))
            return (t1 - t0).total_seconds() / 86400.0 > max_traj_days

        def _to_rc_path(points: list) -> List[Tuple[int, int]]:
            """Convert ``(lon_idx, lat_idx)`` coords to ``(row, col)``."""
            return [(pt["cell_coords"][1], pt["cell_coords"][0]) for pt in points]

        def _deduplicate(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """Remove consecutive duplicate cells."""
            deduped: List[Tuple[int, int]] = [path[0]]
            for rc in path[1:]:
                if rc != deduped[-1]:
                    deduped.append(rc)
            return deduped

        def _densify_or_reject(
            path: List[Tuple[int, int]],
        ) -> Optional[List[Tuple[int, int]]]:
            """Single-pass: reject jumps > *max_coord_jump* and densify
            non-cardinal steps to 4-connected moves.

            Returns the densified path, or ``None`` to reject.

            Note: we intentionally do NOT reuse
            ``ShortestPathBaseline.densify_to_cardinal`` here because
            that method only densifies.  This function also *rejects*
            trajectories with gaps > ``max_coord_jump``, which would
            require a separate pass over the path first.  Keeping both
            checks in a single loop avoids traversing every trajectory
            twice.
            """
            W = self.env.W
            out: List[Tuple[int, int]] = [path[0]]

            for a, b in zip(path[:-1], path[1:]):
                r1, c1 = a
                r2, c2 = b
                dr = abs(r2 - r1)
                dc_raw = abs(c2 - c1)
                dc = min(dc_raw, W - dc_raw)
                manhattan = dr + dc

                if manhattan > max_coord_jump:
                    return None  # too large a jump

                if manhattan <= 1:
                    out.append(b)  # already cardinal
                    continue

                # Small gap (2..max_coord_jump): Dijkstra to fill
                segment = self._baseline.cardinal_path(a, b)
                if segment is None:
                    return None  # no water path between cells
                out.extend(segment[1:])

            return out
        results: List[Tuple[TrajectorySpec, List[Tuple[int, int]]]] = []

        for fpath in json_files:
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Trajectory file not found: {fpath}")
            with open(fpath, "r") as f:
                records = json.load(f)

            for rec in records:
                if limit is not None and len(results) >= limit:
                    return results

                imo = rec.get("imo")
                traj_id = rec.get("id", f"imo={imo}")
                track = rec.get("track", {})
                points = track.get("points", [])

                if len(points) < 2:
                    log.debug("Skipping %s: only %d point(s)", traj_id, len(points))
                    continue

                if _exceeds_duration(rec):
                    continue

                rc_path = _to_rc_path(points)
                deduped = _deduplicate(rc_path)

                if len(deduped) < 2:
                    log.debug("Skipping %s: path length < 2 after dedup", traj_id)
                    continue

                # Single pass: reject large jumps + densify to cardinal
                dense = _densify_or_reject(deduped)
                if dense is None:
                    log.debug(
                        "Skipping %s: coord jump > %d or no water path",
                        traj_id, max_coord_jump,
                    )
                    continue

                start_rc = dense[0]
                goal_rc = dense[-1]
                if not self.env.is_navigable(*start_rc) or not self.env.is_navigable(*goal_rc):
                    continue

                vessel_info = self._get_vessel_info(imo)

                spec = TrajectorySpec(
                    name=traj_id,
                    start_rc=start_rc,
                    goal_rc=goal_rc,
                    start_time=rec.get("passage_start", "") or "",
                    vessel_loa=vessel_info.get("loa", self.env_cfg.default_vessel_loa) or self.env_cfg.default_vessel_loa,
                    vessel_company=vessel_info.get("operator", self.env_cfg.default_vessel_company) or self.env_cfg.default_vessel_company,
                    vessel_speed_kmh=self._lookup_speed(
                        vessel_info.get("operator", self.env_cfg.default_vessel_company) or self.env_cfg.default_vessel_company,
                        vessel_info.get("loa", self.env_cfg.default_vessel_loa) or self.env_cfg.default_vessel_loa,
                    ),
                    imo=imo,
                )
                results.append((spec, dense))

        return results

    def prepare_gt_traj_inputs_outputs(
        self,
        spec: TrajectorySpec,
        rc_path: List[Tuple[int, int]],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Reset *env* with *spec*, step through *rc_path*, collect
        ``(features, actions)``.

        Returns ``(features_list, actions_list)`` where each feature is
        the vector from ``env.build_input_features()`` and each action
        is an int in ``{0, 1, 2, 3}``.

        Raises
        ------
        RuntimeError
            If the env position desyncs from the expected path cell.
        ValueError
            If no valid cardinal action maps a transition in the path.
        """
        try:
            self.env.reset(spec)
        except ValueError as exc:
            log.warning("Skipping %s: env.reset failed — %s", spec.name, exc)
            return [], []

        features: List[np.ndarray] = []
        actions: List[int] = []

        for i in range(len(rc_path) - 1):
            cur = rc_path[i]
            nxt = rc_path[i + 1]

            if self.env.position != cur:
                raise RuntimeError(
                    f"{spec.name} step {i}: env at {self.env.position}, expected {cur}"
                )

            action_idx = self.env.action_from_transition(cur, nxt)
            if action_idx is None:
                raise ValueError(
                    f"{spec.name} step {i}: no valid action for {cur} → {nxt}"
                )

            feat = self.env.build_input_features()
            features.append(feat)
            actions.append(action_idx)

            action = list(Action)[action_idx]
            try:
                self.env.step(action)
            except ValueError as exc:
                log.warning(
                    "%s step %d: env.step(%s) failed — %s; dropping trajectory",
                    spec.name, i, action.name, exc,
                )
                return [], []
        return features, actions

    @property
    def vessel_registry(self) -> Dict[int, Dict[str, Any]]:
        """Lazily-loaded IMO → vessel-metadata lookup.

        Returns an empty dict when ``vessels_json`` is not set or the file
        does not exist — callers fall back to ``EnvConfig`` defaults.
        """
        if self._vessel_registry is None:
            path = self.env_cfg.vessels_json
            if path and os.path.isfile(path):
                with open(path, "r") as f:
                    raw = json.load(f)
                self._vessel_registry = {int(k): v for k, v in raw.items()}
                log.info("Loaded vessel registry (%d vessels) from %s", len(self._vessel_registry), path)
            else:
                log.info("Vessel registry not found at %s — using defaults (LOA=%.1f, company=%r)",
                         path, self.env_cfg.default_vessel_loa, self.env_cfg.default_vessel_company)
                self._vessel_registry = {}
        return self._vessel_registry
    
    def _lookup_speed(self, company: str, loa_m: float) -> float:
        """Delegate to module-level :func:`lookup_vessel_speed_kmh`."""
        return lookup_vessel_speed_kmh(
            company, loa_m,
            self._operator_speed_table, self._loa_bins,
        )

    def _get_vessel_info(self, imo: Optional[int]) -> Dict[str, Any]:
        """Look up vessel metadata by IMO, returning ``{}`` if unavailable."""
        if imo:
            return self.vessel_registry.get(imo, {})
        return {}
    
    def _dump_debug_feature_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        out_path: str,
        sample_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Save selected feature rows to CSV for debugging.

        Parameters
        ----------
        X : ndarray (N, D)
        y : ndarray (N,)
        out_path : str
            CSV file path.
        sample_indices : list[int]
            Which rows to export. Default = [0, 10, 100, 1000]
        """
        import csv

        if sample_indices is None:
            sample_indices = [0, 10, 11, 12, 100, 1000]

        feature_names = self.env.feature_names

        # Filter indices safely
        valid_indices = [i for i in sample_indices if i < len(X)]

        if not valid_indices:
            log.warning("No valid debug indices for dataset of size %d", len(X))
            return

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["sample_idx", *feature_names, "label"])

            # Rows
            for idx in valid_indices:
                writer.writerow([idx, *X[idx].tolist(), int(y[idx])])

        log.info(
            "Debug feature samples saved to %s (indices=%s)",
            out_path,
            valid_indices,
        )

    def _get_track_files(self, split: str) -> List[str]:
        """Resolve *split* to the corresponding track file list."""
        cfg = self.env_cfg
        try:
            return {"train": cfg.train_track_files,
                    "val": cfg.val_track_files,
                    "eval": cfg.eval_track_files}[split]
        except KeyError:
            raise ValueError(f"Unknown split {split!r}; expected train/val/eval")

