"""Grid-based pathfinding utilities and baselines.

Provides:
* :class:`ShortestPathBaseline` — shortest-path baseline using real geographic
  edge lengths from ``lat_mapping_table.csv``.  Builds both 8-connected and
  4-connected networkx graphs; :meth:`~ShortestPathBaseline.cardinal_path`
  provides 4-connected routing used by data loading.
* :class:`CompanyBaseline` — company routing-graph baseline using the
  ``sailing`` package's :class:`RoutingGraph`.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from shipping_route_predictor.config import EnvConfig
from shipping_route_predictor.data import GridCoordinates

log = logging.getLogger("baselines")


# ======================================================================
# ShortestPathBaseline — geographic Dijkstra baseline
# ======================================================================

class ShortestPathBaseline:
    """Shortest-path baseline using real geographic edge lengths.

    Builds an 8-connected ``networkx.DiGraph`` over water cells with
    direction-dependent edge weights from ``lat_mapping_table.csv``:

    * N/S edges: ``lat_edge_km``
    * E/W edges: ``lon_edge_km``  (varies by latitude)
    * Diagonal edges: ``sqrt(lat² + lon²)`` of the destination row

    :meth:`shortest_path` runs Dijkstra on the 8-connected graph and
    coarsifies the result to 4-connected cardinal steps (vertical then
    horizontal for each diagonal).

    Parameters
    ----------
    env_cfg : EnvConfig
        Must have ``mask_csv`` and ``latitude_csv`` populated
        (auto-filled by ``EnvConfig.__post_init__``).

    Examples
    --------
    >>> baseline = ShortestPathBaseline(EnvConfig(grid=GridResolution.FINE))
    >>> path, dist_km = baseline.shortest_path((10, 20), (50, 100))
    """

    # Offsets: 4 cardinal + 4 diagonal
    _OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
    _OFFSETS_8 = _OFFSETS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

    def __init__(self, env_cfg: EnvConfig) -> None:
        self.env_cfg = env_cfg
        self.coords = GridCoordinates(env_cfg)

        # Load mask and crop to valid latitude band
        mask_raw = np.loadtxt(env_cfg.mask_csv, delimiter=",", dtype=float)
        lat_mapping = self._read_lat_mapping(env_cfg.latitude_csv)
        lat_idx = lat_mapping["lat_idx"].astype(int)
        start, stop = int(lat_idx[0]), int(lat_idx[-1]) + 1
        self.mask: np.ndarray = mask_raw[start:stop, :] > 0.5

        self.H, self.W = self.mask.shape

        # Edge lengths per row (km) from lat_mapping_table.csv.
        # Reindex to full mask height with interpolation for any gaps.
        full_H = mask_raw.shape[0]
        ew_full = np.full(full_H, np.nan)
        ns_full = np.full(full_H, np.nan)
        for k, idx in enumerate(lat_idx):
            ew_full[idx] = lat_mapping["lon_edge_km"][k]
            ns_full[idx] = lat_mapping["lat_edge_km"][k]

        valid = ~np.isnan(ew_full)
        if valid.any():
            xp = np.where(valid)[0]
            ew_full = np.interp(np.arange(full_H), xp, ew_full[xp])
            ns_full = np.interp(np.arange(full_H), xp, ns_full[xp])

        self._ew_km: np.ndarray = ew_full[start:stop].astype(np.float64)
        self._ns_km: np.ndarray = ns_full[start:stop].astype(np.float64)

        # Build both 8-connected and 4-connected networkx graphs.
        # The 4-connected graph serves as fallback when coarsifying
        # diagonal steps would route through land cells.
        self._graph_8 = self._build_graph(self._OFFSETS_8)
        self._graph_4 = self._build_graph(self._OFFSETS_4)
        log.info(
            "ShortestPathBaseline: grid %dx%d, %d water cells, "
            "%d edges (8-conn), %d edges (4-conn)",
            self.H, self.W, self._graph_8.number_of_nodes(),
            self._graph_8.number_of_edges(),
            self._graph_4.number_of_edges(),
        )

    def predict_route(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Tuple[Optional[dict], Optional[List[Tuple[float, float]]]]:
        """Run shortest-path and return standardized ``(result, latlon_path)``.

        Returns ``(None, None)`` when no path exists.
        """
        sp_result = self.shortest_path(start, goal)
        if sp_result is None:
            return None, None
        sp_path, _dist_km = sp_result
        latlon = [self.coords.grid_indices_to_latlon(rc) for rc in sp_path]
        result = {"grid_path": sp_path, "reached_goal": True, "mean_prob": None}
        return result, latlon
    
    def shortest_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[Tuple[List[Tuple[int, int]], float]]:
        """Geographically shortest water-only path (4-connected output).

        Tries 8-connected routing first and coarsifies diagonals to
        cardinal pairs.  Falls back to 4-connected routing if any
        diagonal intermediate cell is land.

        Returns ``(path, distance_km)`` or ``None``.
        """
        result_8 = self._find_path(self._graph_8, start, goal)
        if result_8 is None:
            return None

        path_4 = self._coarsify_8_to_4(result_8[0])
        if path_4 is not None:
            return path_4, result_8[1]

        # Coarsification hit land — fall back to 4-connected graph
        log.debug(
            "Coarsify failed for %s → %s; falling back to 4-conn",
            start, goal,
        )
        return self._find_path(self._graph_4, start, goal)
    
    def cardinal_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Shortest 4-connected water-only path.

        Returns list[(row, col)] or ``None``.
        """
        result = self._find_path(self._graph_4, start, goal)
        return result[0] if result is not None else None

    def _find_path(
            self,
            graph: nx.DiGraph,
            start: Tuple[int, int],
            goal: Tuple[int, int],
        ) -> Optional[Tuple[List[Tuple[int, int]], float]]:
            """Run bidirectional Dijkstra on *graph*.

            Returns ``(path, distance_km)`` or ``None``.
            """
            if start == goal:
                return [start], 0.0
            if start not in graph or goal not in graph:
                return None
            try:
                dist, path = nx.bidirectional_dijkstra(
                    graph, start, goal, weight="weight",
                )
            except nx.NetworkXNoPath:
                return None
            return path, float(dist)

    def _edge_weight(
        self, r1: int, c1: int, r2: int, c2: int,
    ) -> float:
        """Geographic edge cost (km) for a single step."""
        dr = r2 - r1
        dc_raw = c2 - c1
        # Determine cardinal vs diagonal
        is_ns = (dc_raw == 0 or abs(dc_raw) == self.W)  # pure vertical
        is_ew = (dr == 0)  # pure horizontal
        if is_ns:
            return float(self._ns_km[r2])
        if is_ew:
            return float(self._ew_km[r1])
        # diagonal: sqrt(ns² + ew²) at destination row
        ns = float(self._ns_km[r2])
        ew = float(self._ew_km[r2])
        return math.sqrt(ns * ns + ew * ew)

    def _build_graph(
        self, offsets: Tuple[Tuple[int, int], ...],
    ) -> nx.DiGraph:
        """Create a DiGraph over water cells with km-weighted edges."""
        G = nx.DiGraph()
        H, W = self.H, self.W
        mask = self.mask

        for i in range(H):
            for j in range(W):
                if not mask[i, j]:
                    continue
                G.add_node((i, j))
                for di, dj in offsets:
                    ni = i + di
                    nj = (j + dj) % W  # wrap columns
                    if 0 <= ni < H and mask[ni, nj]:
                        w = self._edge_weight(i, j, ni, nj)
                        G.add_edge((i, j), (ni, nj), weight=w)
        return G


    @staticmethod
    def _read_lat_mapping(path: str) -> dict:
        """Read lat_mapping_table.csv → dict with arrays for each column."""
        with open(path, "r", encoding="utf-8") as f:
            header = [c.strip().lower() for c in f.readline().strip().split(",")]
        data = np.loadtxt(path, delimiter=",", dtype=np.float64, skiprows=1)
        return {col: data[:, i] for i, col in enumerate(header)}


    def _coarsify_8_to_4(
        self, path: List[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        """Expand diagonal steps into cardinal pairs, checking for land.

        For each diagonal step, tries row-first, then col-first ordering.
        Returns ``None`` if any diagonal cannot be expanded without crossing
        a land cell (caller should fall back to 4-connected routing).
        """
        if len(path) < 2:
            return list(path)
        W = self.W
        mask = self.mask
        out: List[Tuple[int, int]] = [path[0]]
        for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
            # Detect wrap
            is_east = (c2 == (c1 + 1) % W)
            is_west = (c2 == (c1 - 1) % W)
            is_col_same = (c2 == c1)
            is_cardinal = (
                (abs(r2 - r1) <= 1 and (is_east or is_west) and r1 == r2)
                or (is_col_same and abs(r2 - r1) == 1)
            )
            if is_cardinal:
                out.append((r2, c2))
            else:
                # Diagonal → try row-first, then col-first
                mid_row = (r2, c1)   # move row first
                mid_col = (r1, c2)   # move col first
                if mask[mid_row]:
                    out.append(mid_row)
                    out.append((r2, c2))
                elif mask[mid_col]:
                    out.append(mid_col)
                    out.append((r2, c2))
                else:
                    return None  # both intermediates are land
        return out

    # ---- helpers -----------------------------------------------------------

    def path_distance_km(self, path: List[Tuple[int, int]]) -> float:
        """Sum of per-step geographic edge lengths along a 4-connected *path*."""
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
            dc = abs(c2 - c1)
            if dc == 0 or dc == self.W:
                total += self._ns_km[r2]
            else:
                total += self._ew_km[r1]
        return total

    def densify_to_cardinal(
        self, grid_path: List[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        """Densify a coarse grid path to 4-connected cardinal steps.

        For each pair of consecutive cells that are not already cardinal
        neighbours, fills the gap with :meth:`cardinal_path`.

        Returns list[(row, col)] or ``None`` if any gap has no water path.
        """
        out: List[Tuple[int, int]] = [grid_path[0]]
        W = self.W
        for a, b in zip(grid_path[:-1], grid_path[1:]):
            r1, c1 = a
            r2, c2 = b
            dr = abs(r2 - r1)
            dc = min(abs(c2 - c1), W - abs(c2 - c1))
            if dr + dc <= 1:
                out.append(b)
                continue
            segment = self.cardinal_path(a, b)
            if segment is None:
                return None
            out.extend(segment[1:])
        return out


# ======================================================================
# CompanyBaseline — company routing-graph baseline
# ======================================================================

class CompanyBaseline:
    """Company routing-graph baseline using the ``sailing`` package.

    Wraps :class:`sailing.predictors.RoutingGraph` to produce a grid-cell
    path from start to goal.  The routing graph finds waypoints in
    (lon, lat) space; this class maps them back to grid coordinates and
    densifies to cardinal steps via :class:`ShortestPathBaseline`.

    Parameters
    ----------
    env_cfg : EnvConfig
        Provides ``mask_csv`` and ``latitude_csv`` for grid dimensions.
    shortest_path_baseline : ShortestPathBaseline, optional
        Reused for cardinal-step densification.  Built automatically
        from *env_cfg* when ``None``.

    Examples
    --------
    >>> baseline = CompanyBaseline(EnvConfig(grid=GridResolution.FINE))
    >>> grid_path = baseline.predict((10, 20), (50, 100))
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        shortest_path_baseline: Optional[ShortestPathBaseline] = None,
    ) -> None:
        from sailing.predictors import RoutingGraph

        self.env_cfg = env_cfg
        self._sp = shortest_path_baseline or ShortestPathBaseline(env_cfg)
        self.H = self._sp.H
        self.W = self._sp.W
        self._router = RoutingGraph()
        self.coords = GridCoordinates(env_cfg)

        log.info("CompanyBaseline: grid %dx%d (H×W)", self.H, self.W)

    def predict_route(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Tuple[Optional[dict], Optional[List[Tuple[float, float]]]]:
        """Run company routing and return standardized ``(result, latlon_path)``.

        Returns ``(None, None)`` when routing fails.
        """
        co_path = self._predict(start, goal)
        if co_path is None:
            return None, None
        latlon = [self.coords.grid_indices_to_latlon(rc) for rc in co_path]
        result = {"grid_path": co_path, "reached_goal": True, "mean_prob": None}
        return result, latlon


    def _predict(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Predict a route from *start* to *goal* using company routing.

        Parameters
        ----------
        start, goal : (row, col)
            Grid coordinates.

        Returns
        -------
        list[(row, col)] including both endpoints, or ``None`` if routing
        fails.
        """
        lat1, lon1 = self.coords.grid_indices_to_latlon(start)
        lat2, lon2 = self.coords.grid_indices_to_latlon(goal)
        coord1 = np.array([lon1, lat1])
        coord2 = np.array([lon2, lat2])

        route = self._router.get_route_from_point_to_point(coord1, coord2)
        waypoints = route.waypoints

        if waypoints is None:
            raise ValueError(
                f"Routing returned no waypoints for {start} → {goal}"
            )

        if len(waypoints) < 2:
            # Very short route — just return start and goal grid cells
            return [start, goal]

        # sailing API returns [lon, lat]
        coarse_grid_path = self.coords.lonlat_to_grid_indices_batch(
            waypoints,
        )
        return self._sp.densify_to_cardinal(coarse_grid_path)
