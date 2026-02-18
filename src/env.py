"""Lightweight grid environment for XGBoost vessel-routing."""
from __future__ import annotations

import logging
import math
import os
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from shipping_route_predictor.config import (
    EnvConfig,
    TrajectorySpec,
    WeatherConfig,
)
from shipping_route_predictor.data import EnvDataset, GridCoordinates
from shipping_route_predictor.utils import (
    bearing_unit,
    haversine_km,
    parse_iso_timestamp,
    unix_to_date_key,
)

import xarray as xr


class Action(Enum):
    """Cardinal moves on the grid.

    Each member stores ``(index, (row_delta, col_delta))``.
    Access via ``action.index`` and ``action.delta``.
    """
    S = (0, (1, 0))
    N = (1, (-1, 0))
    E = (2, (0, 1))
    W = (3, (0, -1))

    @property
    def index(self) -> int:
        return self.value[0]

    @property
    def delta(self) -> Tuple[int, int]:
        return self.value[1]

    @classmethod
    def from_index(cls, idx: int) -> "Action":
        """Look up an Action by its integer index (0–3)."""
        for a in cls:
            if a.index == idx:
                return a
        raise ValueError(f"{idx} is not a valid Action index")

    @classmethod
    def __len__(cls) -> int:
        return len(cls.__members__)


class EnvGraph:
    """Full-grid environment graph for XGBoost routing.

    4-neighbor connectivity with longitude wrapping.

    Attributes
    ----------
    H, W : int
        Grid height (latitude rows) and width (longitude columns).
    grid : structured ndarray (H, W)
        Fields: *mask* (bool), *lat* (f32), *lon* (f32), *depth* (f32).
        ``grid[r, c]`` → single cell record.
        ``grid['mask']`` → full (H, W) bool layer, etc.
    """

    CELL_DTYPE = np.dtype([
        ("mask", np.bool_),
        ("lat", np.float32),
        ("lon", np.float32),
        ("depth", np.float32),
    ])

    def __init__(self, ds: EnvDataset) -> None:
        mask, lats, lons, depth = ds.load_grid_arrays()
        env_cfg = ds.env_cfg

        H, W = mask.shape
        if env_cfg.grid_height and H != env_cfg.grid_height:
            raise ValueError(
                f"Mask height {H} != env_cfg.grid_height {env_cfg.grid_height}"
            )
        if env_cfg.grid_width and W != env_cfg.grid_width:
            raise ValueError(
                f"Mask width {W} != env_cfg.grid_width {env_cfg.grid_width}"
            )
        self.H = H
        self.W = W
        self.env_cfg = env_cfg
        self.dataset = ds

        self.grid = np.zeros((H, W), dtype=self.CELL_DTYPE)
        self.grid["mask"] = mask
        self.grid["depth"] = depth

        self.grid["lat"] = np.repeat(
            np.asarray(lats, dtype=np.float32).reshape(H, 1), W, axis=1,
        )
        self.grid["lon"] = np.repeat(
            np.asarray(lons, dtype=np.float32).reshape(1, W), H, axis=0,
        )

        # Shared coordinate converter
        self.coords = GridCoordinates(ds.env_cfg)

        # Normalised depth (H, W) — shared by simple and full models
        depth_f = self.grid["depth"].astype(np.float32)
        depth_f[depth_f <= 0] = 0.0
        depth_f[np.isnan(depth_f)] = 0.0
        self.depth_norm: np.ndarray = depth_f / self.env_cfg.depth_median_m

        # Vessel state — set via reset()
        self.position: Optional[Tuple[int, int]] = None
        self.start: Optional[Tuple[int, int]] = None
        self.goal: Optional[Tuple[int, int]] = None
        self.trajectory: Optional[TrajectorySpec] = None
        self.loa_norm: float = 1.0
        self.company_id: int = 0

    def reset(
        self,
        trajectory: TrajectorySpec,
    ) -> None:
        """Begin a new episode from *trajectory*.  Sets ``position``, ``start``, ``goal``."""
        start = tuple(trajectory.start_rc)
        goal = tuple(trajectory.goal_rc)
        if not self.is_navigable(*start):
            raise ValueError(f"Start cell {start} is not navigable.")
        if not self.is_navigable(*goal):
            raise ValueError(f"Goal cell {goal} is not navigable.")
        self.start = start
        self.goal = goal
        self.position = self.start
        self.trajectory = trajectory

        # Vessel metadata
        loa = trajectory.vessel_loa
        self.loa_norm = loa / self.env_cfg.vessel_median_loa_m if loa else 1.0
        company = trajectory.vessel_company or ""
        self.company_id = hash(company) % 24 if company else 0

    def step(self, action: Action) -> None:
        """Apply *action* from the current position, update ``self.position``.

        Raises ``RuntimeError`` if no episode is active (call :meth:`reset` first).
        Raises ``ValueError`` if the move leaves the grid or lands on land.
        """
        if self.position is None:
            raise RuntimeError("No active episode — call reset() before step().")
        row, col = self.position
        row_delta, col_delta = action.delta
        new_row = row + row_delta
        new_col = (col + col_delta) % self.W
        if new_row < 0 or new_row >= self.H or not self.grid["mask"][new_row, new_col]:
            raise ValueError(f"Invalid move: ({row},{col}) + {action.name} -> ({new_row},{new_col})")
        self.position = (new_row, new_col)

    def reached_goal(self) -> bool:
        """True when the vessel is at the goal cell."""
        return self.position is not None and self.position == self.goal

    def build_input_features(self) -> np.ndarray:
        """Construct the 14-dim base feature vector for the simple model.

        Layout (matches ``XGBoostFeatureExtractor.extract_step_features``
        when ``use_full_model=False``):

        ``[dgoal_east, dgoal_north, loa_norm, depth, company_id,
          here_flag, start_flag, goal_flag,
          cur_r, cur_c, start_r, start_c, goal_r, goal_c]``
        """
        if self.position is None or self.start is None or self.goal is None:
            raise RuntimeError("Call reset() before build_features().")

        i, j = self.position
        si, sj = self.start
        gi, gj = self.goal

        dgoal_e, dgoal_n = self.scaled_vector(
            self.position, self.goal, self.env_cfg.goal_vector_norm_km,
        )

        return np.array([
            dgoal_e, dgoal_n,
            self.loa_norm,
            float(self.depth_norm[i, j]),
            float(self.company_id),
            1.0,  # here_flag
            1.0 if self.position == self.start else 0.0,
            1.0 if self.position == self.goal else 0.0,
            float(i), float(j),
            float(si), float(sj),
            float(gi), float(gj),
        ], dtype=np.float32)

    def is_navigable(self, row: int, col: int) -> bool:
        return 0 <= row < self.H and 0 <= col < self.W and bool(self.grid["mask"][row, col])

    def action_from_transition(
        self,
        u: Tuple[int, int],
        v: Tuple[int, int],
    ) -> Optional[int]:
        """Return action index (0=S,1=N,2=E,3=W) for a single grid step, or ``None``."""
        r1, c1 = u
        r2, c2 = v
        for action in Action:
            dr, dc = action.delta
            if r2 == r1 + dr and c2 == (c1 + dc) % self.W:
                return action.index
        return None

    def valid_actions(self, row: int, col: int) -> np.ndarray:
        """Return bool array [Action.count()] indexed by action.index."""
        neighbors = self.neighbors(row, col)
        valid_actions = np.zeros(len(Action), dtype=bool)
        for action in neighbors:
            valid_actions[action.index] = True
        return valid_actions

    @property
    def feature_names(self) -> List[str]:
        """Ordered feature names matching ``build_input_features()`` output."""
        return [
            "dgoal_east_scaled", "dgoal_north_scaled",
            "loa_norm", "depth_norm", "company_id",
            "here_flag", "start_flag", "goal_flag",
            "cur_row", "cur_col", "start_row", "start_col", "goal_row", "goal_col",
        ]

    
    def neighbors(self, row: int, col: int) -> Dict[Action, Tuple[int, int]]:
        """Return valid water neighbors as {Action: (row, col)}."""
        neighbors: Dict[Action, Tuple[int, int]] = {}
        for action in Action:
            row_delta, col_delta = action.delta
            new_row = row + row_delta
            new_col = (col + col_delta) % self.W
            if 0 <= new_row < self.H and self.grid["mask"][new_row, new_col]:
                neighbors[action] = (new_row, new_col)
        return neighbors

    def scaled_vector(
        self,
        from_rc: Tuple[int, int],
        to_rc: Tuple[int, int],
        norm_km: float,
    ) -> Tuple[float, float]:
        """Direction × (distance / *norm_km*) — matches ``_compute_scaled_vector``."""
        lat1, lon1 = self.grid_indices_to_latlon(from_rc)
        lat2, lon2 = self.grid_indices_to_latlon(to_rc)
        d = haversine_km(lat1, lon1, lat2, lon2)
        if d < 1e-6:
            return 0.0, 0.0
        e, n = bearing_unit(lat1, lon1, lat2, lon2)
        s = d / norm_km
        return e * s, n * s
    
    def grid_indices_to_latlon(self, indices: Tuple[int, int]) -> Tuple[float, float]:
        """Return ``(lat, lon)`` in degrees for the cell at ``(row, col)``."""
        return self.coords.grid_indices_to_latlon(indices)

    def cell_distance_km(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Haversine distance in km between two grid cells."""
        return haversine_km(*self.grid_indices_to_latlon(a), *self.grid_indices_to_latlon(b))


class FullEnvGraph(EnvGraph):
    """Full-model environment that mirrors the feature pipeline in
    ``XGBoostFeatureExtractor`` (see ``sequential_modeling/xgboost/trainer.py``).

    Adds:

    * **Beam selection** – angle-gated BFS identical to ``_select_beam_nodes``.
    * **Weather** – preloads all daily NetCDF grids into a single
      ``(N_days, 5, H, W)`` float32 tensor (~590 MB for 944 days on the
      128×256 grid).
    * **Feature construction** – ``build_features()`` returns the same 196-dim
      vector consumed by the XGBoost classifier during rollout.

    Typical usage::

        ds  = EnvDataset(EnvConfig(), ModelType.FULL, WeatherConfig())
        env = build_env(ds)   # returns FullEnvGraph
        traj = TrajectorySpec(name="A→B", start_rc=(40,100), goal_rc=(60,200),
                              start_time="2024-01-15T00:00:00Z",
                              vessel_loa=300.0, vessel_company="msc")
        env.reset(traj)
        while not env.reached_goal():
            feats = env.build_features()
            action = model_predict(feats)
            env.step(action)
    """

    #: A single history entry: grid position + unix timestamp at that step.
    _HISTORY_ENTRY = Tuple[Tuple[int, int], float]
    #: Feature dimensions per beam / history context node (not the vessel node).
    _DIM_HISTORY_BEAM_NODES: int = 13
    # NetCDF variable names → channel index (supports both naming conventions)
    _VAR_TO_CHANNEL: Dict[str, int] = {
        "Uw": 0, "eastward_wind": 0,
        "Vw": 1, "northward_wind": 1,
        "Uc": 2, "uo": 2,
        "Vc": 3, "vo": 3,
        "H": 4, "VHM0": 4,
    }

    def __init__(self, ds: EnvDataset) -> None:
        super().__init__(ds)
        self.weather_cfg = ds.weather_cfg
        self.log = logging.getLogger("env.full")

        # ---- Pre-computed arrays ----
        self._neighbor_map = self._build_neighbor_map()

        # ---- Episode bookkeeping ----
        self._history: List[FullEnvGraph._HISTORY_ENTRY] = []
        self._history_actions: List[int] = []
        self._current_time_unix: float = 0.0
        self._vessel_speed_kmh: float = self.weather_cfg.fixed_speed_kmh
        self._date_key: Optional[str] = None

        # ---- Weather (always preload when directory exists) ----
        self._weather_array: Optional[np.ndarray] = None
        self._weather_date_index: Optional[Dict[str, int]] = None
        wdir = self.weather_cfg.weather_dir
        if wdir and os.path.isdir(wdir):
            self._preload_all_weather()

    def reset(self, trajectory: TrajectorySpec) -> None:
        """Begin episode from a :class:`TrajectorySpec`.

        Vessel metadata (LOA, company, departure time) are extracted
        from the trajectory.
        """
        super().reset(trajectory)
        self._history.clear()
        self._history_actions.clear()
        self._vessel_speed_kmh = (
            trajectory.vessel_speed_kmh
            or self.weather_cfg.fixed_speed_kmh
        )

        # Time
        if trajectory.start_time:
            ts, dk = parse_iso_timestamp(trajectory.start_time)
            self._current_time_unix = ts
            self._date_key = dk
        else:
            self._current_time_unix = 0.0
            self._date_key = None

    def step(self, action: Action) -> Tuple[Tuple[int, int], float]:
        """Take a step, update history / cumulative time.

        Returns ``(position, current_time_unix)``.
        """
        if self.position is None:
            raise RuntimeError("No active episode — call reset() before step().")
        prev = self.position

        # Record history *before* moving
        self._history.append((prev, self._current_time_unix))
        self._history_actions.append(action.index)

        super().step(action)

        # Advance cumulative time
        if self._current_time_unix and self._vessel_speed_kmh > 0:
            dist_km = self.cell_distance_km(prev, self.position)
            self._current_time_unix += (dist_km / self._vessel_speed_kmh) * 3600.0

        return self.position, self._current_time_unix

   
    def select_beam_nodes(self) -> List[Tuple[int, int]]:
        """Return up to ``num_beam_slots`` water cells inside the goal-cone.

        Algorithm mirrors ``XGBoostFeatureExtractor._select_beam_nodes``.
        Uses ``self.position`` and ``self.goal`` from the current episode.
        """
        wcfg = self.weather_cfg
        cur_rc = self.position
        goal_rc = self.goal
        if cur_rc is None or goal_rc is None:
            raise RuntimeError("No active episode — call reset() first.")

        goal_dir = self._goal_direction_grid(cur_rc, goal_rc)
        cos_thresh = math.cos(math.radians(wcfg.beam_sector_deg / 2.0))
        stride = max(1, wcfg.beam_stride)

        visited = {cur_rc}
        frontier = [cur_rc]
        beam: List[Tuple[int, int]] = []

        for depth in range(1, wcfg.beam_radius_steps + 1):
            next_frontier: List[Tuple[int, int]] = []
            keep = (depth % stride == 0)
            for fi, fj in frontier:
                for a in range(4):
                    ni, nj = int(self._neighbor_map[fi, fj, a, 0]), int(self._neighbor_map[fi, fj, a, 1])
                    if ni < 0:
                        continue
                    if (ni, nj) in visited:
                        continue
                    node_dir = self._goal_direction_grid(cur_rc, (ni, nj))
                    dot = goal_dir[0] * node_dir[0] + goal_dir[1] * node_dir[1]
                    if dot >= cos_thresh or len(beam) == 0:
                        visited.add((ni, nj))
                        next_frontier.append((ni, nj))
                        if keep:
                            beam.append((ni, nj))
                            if len(beam) >= wcfg.num_beam_slots:
                                return beam[: wcfg.num_beam_slots]
            frontier = next_frontier
            if not frontier:
                break
        return beam[: wcfg.num_beam_slots]

    def build_input_features(self) -> np.ndarray:
        """Construct the full 196-dim feature vector for the current state.

        Layout (matches ``XGBoostFeatureExtractor.extract_step_features``
        when ``use_full_model=True``):

        * **[0–13]** base (14): dgoal_east, dgoal_north, loa_norm, depth, company,
          here/start/goal flags, cur_r/c, start_r/c, goal_r/c
        * **[14–117]** beam (8 × 13 = 104)
        * **[118–195]** history (6 × 13 = 78)
        """
        if self.position is None or self.start is None or self.goal is None:
            raise RuntimeError("Call reset() before build_features().")

        wcfg = self.weather_cfg
        cur = self.position
        start = self.start
        goal = self.goal

        i, j = cur
        si, sj = start
        gi, gj = goal

        # --- base features (14) ---
        dgoal_e, dgoal_n = self.scaled_vector(cur, goal, self.env_cfg.goal_vector_norm_km)
        depth_n = float(self.depth_norm[i, j])

        vessel_node = np.array([
            dgoal_e, dgoal_n,
            self.loa_norm, depth_n, float(self.company_id),
            1.0,  # here_flag
            1.0 if cur == start else 0.0,
            1.0 if cur == goal else 0.0,
            float(i), float(j),
            float(si), float(sj),
            float(gi), float(gj),
        ], dtype=np.float32)

        beam_nodes = self._build_beam_features()
        history_nodes = self._build_history_features()

        return np.concatenate([vessel_node, beam_nodes, history_nodes])

    @property
    def feature_names(self) -> List[str]:
        """Ordered feature names matching ``build_input_features()`` (196-dim)."""
        base = super().feature_names
        ctx_names = [
            "here", "start", "goal",
            "vessel_rel_east", "vessel_rel_north",
            "dgoal_east", "dgoal_north",
            "depth_norm",
            "wind_u", "wind_v", "current_u", "current_v", "wave_height",
        ]
        wcfg = self.weather_cfg
        for prefix, n_slots in [("beam", wcfg.num_beam_slots), ("hist", wcfg.num_history_slots)]:
            for s in range(n_slots):
                for cn in ctx_names:
                    base.append(f"{prefix}_{s}_{cn}")
        return base

    def _build_beam_features(self) -> np.ndarray:
        """Build ``(num_beam_slots × 13)`` features for beam nodes.

        Weather is looked up at each node's ETA-based date.
        """
        wcfg = self.weather_cfg
        cur_rc = self.position
        nodes = self.select_beam_nodes()
        n = min(len(nodes), wcfg.num_beam_slots)

        # Compute per-node date keys from ETA
        if n > 0 and self._current_time_unix and self._vessel_speed_kmh > 0:
            date_keys: List[Optional[str]] = []
            for ni, nj in nodes[:n]:
                d = self.cell_distance_km(cur_rc, (ni, nj))
                eta_s = (d / self._vessel_speed_kmh) * 3600.0
                date_keys.append(unix_to_date_key(self._current_time_unix + eta_s))
            weather = self._gather_weather_by_date(nodes[:n], date_keys)
        else:
            weather = self._get_weather_at(nodes[:n], self._date_key)

        return self._set_node_input_features(
            nodes, weather, wcfg.num_beam_slots,
        )

    def _build_history_features(self) -> np.ndarray:
        """Build ``(num_history_slots × 13)`` features for history nodes.

        History entries are reversed (most recent first), strided, and
        truncated. Weather is looked up at each step's recorded timestamp.
        """
        wcfg = self.weather_cfg
        stride = max(1, wcfg.time_stride)

        entries = list(reversed(self._history))
        if stride > 1:
            entries = entries[::stride]
        entries = entries[: wcfg.num_history_slots]
        n = len(entries)

        positions = [e[0] for e in entries]
        timestamps = [e[1] for e in entries]

        # Per-step date keys from recorded timestamps
        if n > 0 and timestamps[0]:
            date_keys = [unix_to_date_key(t) if t else None for t in timestamps]
            weather = self._gather_weather_by_date(positions, date_keys)
        else:
            weather = self._get_weather_at(positions, self._date_key)

        return self._set_node_input_features(
            positions, weather, wcfg.num_history_slots,
        )

    def _set_node_input_features(
        self,
        nodes: List[Tuple[int, int]],
        weather: Dict[str, np.ndarray],
        max_slots: int,
    ) -> np.ndarray:
        """Write the 13-dim feature block for each node into a flat array."""
        wcfg = self.weather_cfg
        cur_rc = self.position
        start_rc = self.start
        goal_rc = self.goal
        out = np.full(max_slots * self._DIM_HISTORY_BEAM_NODES, wcfg.padding_value, dtype=np.float32)
        n = min(len(nodes), max_slots)
        for slot in range(n):
            ni, nj = nodes[slot]
            off = slot * self._DIM_HISTORY_BEAM_NODES
            out[off + 0] = 0.0  # here_flag (beam/history are never current)
            out[off + 1] = 1.0 if (ni, nj) == start_rc else 0.0
            out[off + 2] = 1.0 if (ni, nj) == goal_rc else 0.0
            ve, vn = self.scaled_vector(cur_rc, (ni, nj), wcfg.vessel_vector_norm_km)
            out[off + 3] = ve
            out[off + 4] = vn
            de, dn = self.scaled_vector((ni, nj), goal_rc, self.env_cfg.goal_vector_norm_km)
            out[off + 5] = de
            out[off + 6] = dn
            out[off + 7] = float(self.depth_norm[ni, nj])
            out[off + 8] = weather["wind_u"][slot]
            out[off + 9] = weather["wind_v"][slot]
            out[off + 10] = weather["current_u"][slot]
            out[off + 11] = weather["current_v"][slot]
            out[off + 12] = weather["wave_height"][slot]
        return out

    def _gather_weather_by_date(
        self,
        nodes: List[Tuple[int, int]],
        date_keys: List[Optional[str]],
    ) -> Dict[str, np.ndarray]:
        """Batch-fetch weather for *nodes* whose per-slot date may differ.

        Groups nodes by date key, fetches weather per group, then
        scatters results back into per-slot arrays.
        """
        n = len(nodes)
        w_flat: Dict[str, np.ndarray] = {
            k: np.zeros(n, np.float32)
            for k in ("wind_u", "wind_v", "current_u", "current_v", "wave_height")
        }
        unique_dks = {dk for dk in date_keys if dk is not None}
        for dk in unique_dks:
            slot_indices = [i for i, d in enumerate(date_keys) if d == dk]
            positions = [nodes[i] for i in slot_indices]
            w = self._get_weather_at(positions, dk)
            for k in w_flat:
                for src, dst in enumerate(slot_indices):
                    w_flat[k][dst] = w[k][src]
        return w_flat

    def _preload_all_weather(self) -> None:
        """Load every daily NetCDF into RAM as ``(N_days, 5, H, W)`` float32.

        Memory: ~590 MB for 944 days on a 128×256 grid.
        """
        weather_dir = self.weather_cfg.weather_dir
        if not weather_dir:
            raise ValueError("weather_dir is not set.")
        files = sorted(f for f in os.listdir(weather_dir) if f.endswith(".nc"))
        if not files:
            raise FileNotFoundError(f"No .nc files in {weather_dir}")

        n_days = len(files)
        self.log.info(
            f"Preloading {n_days} weather days ({n_days * 5 * self.H * self.W * 4 / 1e6:.0f} MB) …"
        )

        weather_array = np.empty((n_days, 5, self.H, self.W), dtype=np.float32)
        date_index: Dict[str, int] = {}

        for idx, fname in enumerate(files):
            dk = fname[:10].replace("-", "")
            date_index[dk] = idx
            path = os.path.join(weather_dir, fname)
            ds = xr.open_dataset(path)
            for vname in list(ds.data_vars):
                ch = self._VAR_TO_CHANNEL.get(vname)
                if ch is None:
                    raise ValueError(
                        f"Unknown variable '{vname}' in {fname}. "
                        f"Expected one of {list(self._VAR_TO_CHANNEL)}"
                    )
                weather_array[idx, ch] = self._read_nc_grid(ds[vname])
            ds.close()

        # Normalise: m/s → km/h then divide by max
        weather_array[:, 0:2] *= 3.6   # wind u, v
        weather_array[:, 2:4] *= 3.6   # current u, v
        wind_max_kmh = 40.0 * 3.6     # 144
        current_max_kmh = 3.0 * 3.6   # 10.8
        wave_max_m = 12.0
        weather_array[:, 0] /= wind_max_kmh
        weather_array[:, 1] /= wind_max_kmh
        weather_array[:, 2] /= current_max_kmh
        weather_array[:, 3] /= current_max_kmh
        weather_array[:, 4] /= wave_max_m

        self._weather_array = weather_array
        self._weather_date_index = date_index
        self.log.info(f"Weather preloaded: {n_days} days, shape {weather_array.shape}")

    def _read_nc_grid(self, da: "xr.DataArray") -> np.ndarray:
        """Extract a single ``(H, W)`` float32 grid from a DataArray.

        Handles the common case where a trailing ``time=1`` dimension is present
        and transposes ``(W, H)`` → ``(H, W)`` if the axes are swapped.
        """
        if "time" in da.dims and da.sizes.get("time", 1) == 1:
            da = da.isel(time=0)
        arr = da.values.astype(np.float32).squeeze()
        if arr.shape == (self.W, self.H):
            arr = arr.T
        if arr.shape != (self.H, self.W):
            raise ValueError(
                f"Expected shape ({self.H}, {self.W}), got {arr.shape} "
                f"for variable '{da.name}'"
            )
        return arr

    def _get_weather_at(
        self,
        positions: List[Tuple[int, int]],
        date_key: Optional[str],
    ) -> Dict[str, np.ndarray]:
        """Return 5 normalised weather channels for a list of cell positions.

        Returns dict with keys ``wind_u, wind_v, current_u, current_v, wave_height``,
        each shaped ``(len(positions),)``.
        """
        n = len(positions)
        zeros: Dict[str, np.ndarray] = {
            k: np.zeros(n, dtype=np.float32)
            for k in ("wind_u", "wind_v", "current_u", "current_v", "wave_height")
        }
        if n == 0 or date_key is None:
            return zeros

        if self._weather_array is None or self._weather_date_index is None:
            return zeros
        idx = self._weather_date_index.get(date_key)
        if idx is None:
            return zeros
        day = self._weather_array[idx]  # (5, H, W)
        rows = np.array([p[0] for p in positions], dtype=np.intp)
        cols = np.array([p[1] for p in positions], dtype=np.intp)
        return {
            "wind_u": day[0, rows, cols].copy(),
            "wind_v": day[1, rows, cols].copy(),
            "current_u": day[2, rows, cols].copy(),
            "current_v": day[3, rows, cols].copy(),
            "wave_height": day[4, rows, cols].copy(),
        }
    
    def _goal_direction_grid(
        self, cur_rc: Tuple[int, int], goal_rc: Tuple[int, int],
    ) -> Tuple[float, float]:
        """Grid-space unit direction ``(east, north)`` for beam cone check."""
        di = goal_rc[0] - cur_rc[0]
        dj_raw = goal_rc[1] - cur_rc[1]
        dj = dj_raw - np.sign(dj_raw) * self.W if abs(dj_raw) > self.W / 2 else dj_raw
        length = math.sqrt(di * di + dj * dj)
        if length < 1e-8:
            return 0.0, 0.0
        return dj / length, di / length  # east, north in grid space

    def _build_neighbor_map(self) -> np.ndarray:
        """Pre-compute ``[H, W, 4, 2]`` neighbor indices (S, N, E, W)."""
        nmap = np.full((self.H, self.W, 4, 2), -1, dtype=np.int32)
        for i in range(self.H):
            for j in range(self.W):
                for action, (ni, nj) in self.neighbors(i, j).items():
                    nmap[i, j, action.index] = [ni, nj]
        return nmap

def build_env(ds: EnvDataset) -> EnvGraph:
    """Build the right :class:`EnvGraph` subclass from *ds*.

    Returns :class:`FullEnvGraph` when ``ds.model_type`` is
    ``ModelType.FULL``, plain :class:`EnvGraph` otherwise.

    This is the standard entry point for constructing an environment
    from an :class:`EnvDataset`::

        from shipping_route_predictor.data import EnvDataset
        from shipping_route_predictor.env import build_env

        ds  = EnvDataset(env_cfg, model_type, weather_cfg)
        env = build_env(ds)
    """
    from shipping_route_predictor.config import ModelType

    if ds.model_type == ModelType.FULL:
        return FullEnvGraph(ds)
    return EnvGraph(ds)
