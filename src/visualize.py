"""Route visualisation on geographic maps.

Provides :class:`RouteVisualizer` which renders one or more predicted
routes onto a Cartopy (preferred) or plain matplotlib map, with optional
navigation-mask overlay and grid lines.

Can be used standalone with any ``{name: [(lat, lon), ...]}`` route dict.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from shipping_route_predictor.config import (
    CATEGORY_COLORS,
    COLOR_CYCLE,
    InferenceConfig,
)
from shipping_route_predictor.env import EnvGraph

log = logging.getLogger("visualize")

# Colour palettes — canonical definitions live in config.py
_MODEL_COLORS: Dict[str, str] = InferenceConfig.RUNNER_COLORS
_COLOR_CYCLE = list(COLOR_CYCLE)
_CATEGORY_COLORS: Dict[str, str] = CATEGORY_COLORS

class RouteVisualizer:
    """Renders predicted routes on a map with optional grid-mask overlay.

    Parameters
    ----------
    env : EnvGraph
        Grid environment (used for mask overlay and grid lines).
    """

    def __init__(self, env: EnvGraph) -> None:
        self.env = env

    # ── colour helpers ────────────────────────────────────────

    @staticmethod
    def _color_for(name: str, idx: int) -> str:
        for key, c in _MODEL_COLORS.items():
            if key in name.lower():
                return c
        return _COLOR_CYCLE[idx % len(_COLOR_CYCLE)]

    # ── antimeridian helpers ────────────────────────────────

    @staticmethod
    def _collect_latlons(
        routes: Dict[str, List[Tuple[float, float]]],
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Tuple[List[float], List[float]]:
        """Gather all latitude / longitude values from routes + endpoints."""
        all_lats = [start[0], goal[0]]
        all_lons = [start[1], goal[1]]
        for path in routes.values():
            for lat, lon in path:
                all_lats.append(lat)
                all_lons.append(lon)
        return all_lats, all_lons

    @staticmethod
    def _crosses_antimeridian(all_lons: List[float]) -> bool:
        """Return True when longitudes span across the ±180° boundary.

        Heuristic: if the [0, 360] span is *smaller* than the [-180, 180]
        span, the data wraps around the antimeridian.
        """
        if len(all_lons) < 2:
            return False
        span_normal = max(all_lons) - min(all_lons)
        shifted = [lon % 360 for lon in all_lons]
        span_shifted = max(shifted) - min(shifted)
        return span_shifted < span_normal

    @staticmethod
    def _split_segments_at_antimeridian(
        lats: List[float], lons: List[float],
    ) -> List[Tuple[List[float], List[float]]]:
        """Split a polyline into segments that do not cross ±180°."""
        segments: List[Tuple[List[float], List[float]]] = []
        cur_la = [lats[0]]
        cur_lo = [lons[0]]
        for i in range(1, len(lats)):
            if abs(lons[i] - lons[i - 1]) > 180:
                segments.append((cur_la, cur_lo))
                cur_la = [lats[i]]
                cur_lo = [lons[i]]
            else:
                cur_la.append(lats[i])
                cur_lo.append(lons[i])
        if cur_la:
            segments.append((cur_la, cur_lo))
        return segments

    # ── public API ────────────────────────────────────────────

    def save(
        self,
        routes: Dict[str, List[Tuple[float, float]]],
        start_latlon: Tuple[float, float],
        goal_latlon: Tuple[float, float],
        start_label: str,
        goal_label: str,
        output_path: Path,
        title: str = "",
    ) -> None:
        """Render all routes on a Cartopy map and save to *output_path* (PNG)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

        env = self.env

        all_lats, all_lons = self._collect_latlons(routes, start_latlon, goal_latlon)
        wraps = self._crosses_antimeridian(all_lons)

        # Choose projection: Pacific-centred when the route wraps
        central_lon = 180.0 if wraps else 0.0
        proj = ccrs.PlateCarree(central_longitude=central_lon)
        data_crs = ccrs.PlateCarree()  # coordinates are always standard lon/lat

        extent = self._extent(all_lats, all_lons, wraps, central_lon)

        lats_1d = env.grid["lat"][:, 0].astype(float)
        lons_1d = env.grid["lon"][0, :].astype(float)
        lat_step = abs(lats_1d[1] - lats_1d[0]) if len(lats_1d) > 1 else 1.0
        lon_step = abs(lons_1d[1] - lons_1d[0]) if len(lons_1d) > 1 else 1.0

        # Determine the longitude span in degrees (accounting for wrap)
        lon_span = extent[1] - extent[0]
        lat_span = extent[3] - extent[2]
        # Keep width=16, scale height so the map is never too narrow
        fig_height = max(8, min(14, 16 * (lat_span / max(lon_span, 1))))
        fig = plt.figure(figsize=(16, fig_height))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent(extent, crs=proj)

        # Natural features
        ax.add_feature(cfeature.OCEAN, facecolor="#cce5ff", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f3f3f3", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#666666", zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", edgecolor="#999999", zorder=2)

        # Navigation-mask overlay (pcolormesh — fast at any zoom)
        mask = env.grid["mask"]
        # Build cell-edge arrays for pcolormesh (shape H+1, W+1 not needed;
        # we pass cell-centre coords and use shading="nearest").
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        mask_rgba[~mask] = [0.4, 0.4, 0.4, 0.45]  # land cells
        ax.pcolormesh(
            lons_1d, lats_1d, mask_rgba[..., 3],
            cmap="Greys", vmin=0, vmax=1, alpha=0.45,
            shading="nearest", transform=data_crs, zorder=3,
        )

        # Grid lines at cell boundaries
        # For wide views use thinner, more transparent lines
        if lon_span > 90:
            grid_lw, grid_alpha = 0.15, 0.25
        else:
            grid_lw, grid_alpha = 0.3, 0.4

        for i in range(len(lons_1d)):
            b = float(lons_1d[i]) - lon_step / 2
            ax.plot(
                [b, b], [extent[2], extent[3]], color="gray",
                linewidth=grid_lw, alpha=grid_alpha,
                transform=data_crs, zorder=4,
            )
        for i in range(len(lats_1d)):
            b = float(lats_1d[i]) - lat_step / 2
            if extent[2] - lat_step <= b <= extent[3] + lat_step:
                ax.plot(
                    [extent[0] + central_lon, extent[1] + central_lon],
                    [b, b], color="gray",
                    linewidth=grid_lw, alpha=grid_alpha,
                    transform=data_crs, zorder=4,
                )

        # Tick formatters
        lon_ticks = np.arange(
            np.floor(extent[0] / 10) * 10,
            np.ceil(extent[1] / 10) * 10 + 10, 10,
        )
        lat_ticks = np.arange(
            np.floor(extent[2] / 10) * 10,
            np.ceil(extent[3] / 10) * 10 + 10, 10,
        )
        ax.set_xticks(lon_ticks, crs=proj)
        ax.set_yticks(lat_ticks, crs=proj)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(labelsize=10)

        # Plot routes
        for idx, (name, path) in enumerate(routes.items()):
            if not path:
                continue
            p_lats = [p[0] for p in path]
            p_lons = [p[1] for p in path]
            color = self._color_for(name, idx)

            if wraps:
                # Draw each continuous segment separately to avoid
                # horizontal lines spanning the whole map.
                segments = self._split_segments_at_antimeridian(p_lats, p_lons)
                for si, (seg_la, seg_lo) in enumerate(segments):
                    ax.plot(
                        seg_lo, seg_la, "-", color=color, linewidth=2.5,
                        label=name if si == 0 else None,
                        transform=data_crs, zorder=8 + idx,
                    )
                    ax.scatter(
                        seg_lo, seg_la, c=color, s=20, zorder=8 + idx,
                        transform=data_crs, edgecolors="white", linewidths=0.3,
                    )
            else:
                ax.plot(
                    p_lons, p_lats, "-", color=color, linewidth=2.5,
                    label=name, transform=data_crs, zorder=8 + idx,
                )
                ax.scatter(
                    p_lons, p_lats, c=color, s=20, zorder=8 + idx,
                    transform=data_crs, edgecolors="white", linewidths=0.3,
                )

        # Start / Goal markers
        ax.scatter(
            [start_latlon[1]], [start_latlon[0]], c="green", s=300, marker="*",
            label=f"Origin ({start_label})", transform=data_crs,
            zorder=15, edgecolors="white", linewidths=1,
        )
        ax.scatter(
            [goal_latlon[1]], [goal_latlon[0]], c="red", s=300, marker="*",
            label=f"Destination ({goal_label})", transform=data_crs,
            zorder=15, edgecolors="white", linewidths=1,
        )

        if title:
            ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log.info("Map saved to %s", output_path)

    @staticmethod
    def _extent(
        all_lats: List[float],
        all_lons: List[float],
        wraps: bool,
        central_lon: float,
    ) -> List[float]:
        """Compute ``[lon0, lon1, lat0, lat1]`` in the *projection* CRS.

        When *wraps* is True the longitudes are shifted into the
        ``[central_lon - 180, central_lon + 180)`` range so the extent
        straddles the antimeridian correctly.
        """
        if wraps:
            # Shift longitudes relative to central_lon so they're continuous
            shifted = [((lon - central_lon + 180) % 360) - 180 for lon in all_lons]
        else:
            shifted = list(all_lons)

        lat_span = max(all_lats) - min(all_lats)
        lon_span = max(shifted) - min(shifted)
        lat_margin = max(lat_span * 0.25, 5)
        lon_margin = max(lon_span * 0.15, 5)
        # Ensure the latitude extent is at least 40% of the longitude extent
        # so the map never looks too squished.
        desired_lat_span = lon_span * 0.40
        if (lat_span + 2 * lat_margin) < desired_lat_span:
            lat_margin = (desired_lat_span - lat_span) / 2

        return [
            min(shifted) - lon_margin,
            max(shifted) + lon_margin,
            max(min(all_lats) - lat_margin, -85),
            min(max(all_lats) + lat_margin, 85),
        ]

def save_shap_route_timeline(
    step_info: List[Dict],
    step_category_shap: "np.ndarray",
    category_names: List[str],
    start_latlon: Tuple[float, float],
    goal_latlon: Tuple[float, float],
    start_label: str,
    goal_label: str,
    output_path: "Path",
    title: str = "",
    env: "EnvGraph | None" = None,
) -> None:
    """Save a combined figure: predicted route map (top) + SHAP line chart (bottom).

    The top panel renders the predicted route on a Cartopy map identical
    to :class:`RouteVisualizer` (with antimeridian handling).  The bottom
    panel shows one **line** per feature category with the *sum* of
    absolute SHAP values at each step.

    Parameters
    ----------
    step_info : list[dict]
        Per-step metadata (must contain ``cur_lat``, ``cur_lon``).
    step_category_shap : ndarray (n_steps, n_categories)
        Per-step sum |SHAP| values per category.
    category_names : list[str]
        Category labels matching the columns of *step_category_shap*.
    start_latlon, goal_latlon : (lat, lon)
        Route endpoint coordinates.
    start_label, goal_label : str
        Human-readable endpoint names.
    output_path : Path
        Where to save the PNG.
    title : str
        Figure super-title.
    env : EnvGraph, optional
        Grid environment for navigation-mask and grid-line overlay.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not step_info or step_category_shap.size == 0:
        log.warning("No step data for route+timeline plot — skipping")
        return

    lats = [s["cur_lat"] for s in step_info]
    lons = [s["cur_lon"] for s in step_info]
    steps = np.arange(len(step_info))

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    # ── Detect antimeridian wrap ──────────────────────────────
    all_lons = list(lons) + [start_latlon[1], goal_latlon[1]]
    all_lats = list(lats) + [start_latlon[0], goal_latlon[0]]
    wraps = RouteVisualizer._crosses_antimeridian(all_lons)
    central_lon = 180.0 if wraps else 0.0
    extent = RouteVisualizer._extent(all_lats, all_lons, wraps, central_lon)

    lon_span = extent[1] - extent[0]

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1, 1],
        hspace=0.22,
    )

    # ── Top panel: Cartopy route map ──────────────────────────
    if has_cartopy:
        proj = ccrs.PlateCarree(central_longitude=central_lon)
        data_crs = ccrs.PlateCarree()

        ax_map = fig.add_subplot(gs[0], projection=proj)
        ax_map.set_extent(extent, crs=proj)

        ax_map.add_feature(cfeature.OCEAN, facecolor="#cce5ff", zorder=0)
        ax_map.add_feature(cfeature.LAND, facecolor="#f3f3f3", zorder=1)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#666666", zorder=2)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--",
                           edgecolor="#999999", zorder=2)

        # Grid overlay (mask + grid lines) when env is available
        if env is not None:
            lats_1d = env.grid["lat"][:, 0].astype(float)
            lons_1d = env.grid["lon"][0, :].astype(float)
            lat_step = abs(lats_1d[1] - lats_1d[0]) if len(lats_1d) > 1 else 1.0
            lon_step = abs(lons_1d[1] - lons_1d[0]) if len(lons_1d) > 1 else 1.0

            # Navigation-mask via pcolormesh
            mask = env.grid["mask"]
            ax_map.pcolormesh(
                lons_1d, lats_1d, (~mask).astype(float),
                cmap="Greys", vmin=0, vmax=1, alpha=0.45,
                shading="nearest", transform=data_crs, zorder=3,
            )

            # Grid lines
            grid_lw = 0.15 if lon_span > 90 else 0.3
            grid_alpha = 0.25 if lon_span > 90 else 0.4
            for i in range(len(lons_1d)):
                b = float(lons_1d[i]) - lon_step / 2
                ax_map.plot(
                    [b, b], [extent[2], extent[3]], color="gray",
                    linewidth=grid_lw, alpha=grid_alpha,
                    transform=data_crs, zorder=4,
                )
            for i in range(len(lats_1d)):
                b = float(lats_1d[i]) - lat_step / 2
                if extent[2] - lat_step <= b <= extent[3] + lat_step:
                    ax_map.plot(
                        [extent[0] + central_lon, extent[1] + central_lon],
                        [b, b], color="gray",
                        linewidth=grid_lw, alpha=grid_alpha,
                        transform=data_crs, zorder=4,
                    )

        # Route line (coloured by dominant category at each step)
        dominant_cats = np.argmax(step_category_shap, axis=1)
        if wraps:
            for i in range(len(lats) - 1):
                if abs(lons[i + 1] - lons[i]) > 180:
                    continue  # skip antimeridian jump segment
                cat = category_names[dominant_cats[i]]
                color = _CATEGORY_COLORS.get(cat, "#333333")
                ax_map.plot(
                    [lons[i], lons[i + 1]], [lats[i], lats[i + 1]],
                    "-", color=color, linewidth=3, transform=data_crs, zorder=5,
                )
        else:
            for i in range(len(lats) - 1):
                cat = category_names[dominant_cats[i]]
                color = _CATEGORY_COLORS.get(cat, "#333333")
                ax_map.plot(
                    [lons[i], lons[i + 1]], [lats[i], lats[i + 1]],
                    "-", color=color, linewidth=3, transform=data_crs, zorder=5,
                )

        ax_map.scatter(lons, lats, c="white", s=12, zorder=6,
                       transform=data_crs, edgecolors="grey", linewidths=0.3)

        # Start / Goal markers
        ax_map.scatter(
            [start_latlon[1]], [start_latlon[0]], c="green", s=250, marker="*",
            label=f"Origin ({start_label})", transform=data_crs,
            zorder=10, edgecolors="white", linewidths=1,
        )
        ax_map.scatter(
            [goal_latlon[1]], [goal_latlon[0]], c="red", s=250, marker="*",
            label=f"Destination ({goal_label})", transform=data_crs,
            zorder=10, edgecolors="white", linewidths=1,
        )

        # Tick formatters
        lon_ticks = np.arange(
            np.floor(extent[0] / 10) * 10,
            np.ceil(extent[1] / 10) * 10 + 10, 10,
        )
        lat_ticks = np.arange(
            np.floor(extent[2] / 10) * 10,
            np.ceil(extent[3] / 10) * 10 + 10, 10,
        )
        ax_map.set_xticks(lon_ticks, crs=proj)
        ax_map.set_yticks(lat_ticks, crs=proj)
        ax_map.xaxis.set_major_formatter(LongitudeFormatter())
        ax_map.yaxis.set_major_formatter(LatitudeFormatter())
        ax_map.tick_params(labelsize=9)

        ax_map.legend(loc="upper left", fontsize=9, framealpha=0.9)
    else:
        ax_map = fig.add_subplot(gs[0])
        ax_map.plot(lons, lats, "-o", markersize=3, linewidth=2)
        ax_map.scatter([start_latlon[1]], [start_latlon[0]], c="green", s=200,
                       marker="*", zorder=10, label=f"Origin ({start_label})")
        ax_map.scatter([goal_latlon[1]], [goal_latlon[0]], c="red", s=200,
                       marker="*", zorder=10, label=f"Destination ({goal_label})")
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # ── Bottom panel: line-plot SHAP timeline ─────────────────
    ax_shap = fig.add_subplot(gs[1])

    for j, cat in enumerate(category_names):
        vals = step_category_shap[:, j]
        color = _CATEGORY_COLORS.get(cat, _COLOR_CYCLE[j % len(_COLOR_CYCLE)])
        ax_shap.plot(steps, vals, "-", linewidth=2, label=cat, color=color)

    ax_shap.set_xlabel("Step along predicted route", fontsize=11)
    ax_shap.set_ylabel("Sum |SHAP value| by category", fontsize=11)
    ax_shap.set_title("Category SHAP Importance Along Route", fontsize=12,
                       fontweight="bold")
    ax_shap.set_xlim(0, len(steps) - 1)
    ax_shap.legend(loc="upper right", fontsize=9,
                   ncol=min(len(category_names), 5), framealpha=0.9)
    ax_shap.grid(axis="both", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Route + SHAP timeline saved to %s", output_path)
