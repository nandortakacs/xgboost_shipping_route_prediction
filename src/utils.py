"""Shared utilities for the shipping-route-predictor XGBoost pipeline."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from shipping_route_predictor.config import DEFAULT_SPEED_KMH, R_EARTH_KM


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points given in degrees."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return R_EARTH_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_unit(
    lat1: float, lon1: float, lat2: float, lon2: float,
) -> Tuple[float, float]:
    """Unit bearing vector ``(east, north)`` from point 1 → point 2 (degrees)."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.cos(lat2_r) * math.sin(dlon)
    y = (
        math.cos(lat1_r) * math.sin(lat2_r)
        - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    )
    bearing = math.atan2(x, y)
    return math.sin(bearing), math.cos(bearing)

def discrete_frechet_distance(P, Q):
    """Compute the discrete Fréchet distance between two sequences of (lat, lon) points.
    Args:
        P: list of (lat, lon) tuples
        Q: list of (lat, lon) tuples
    Returns:
        Fréchet distance (float, in km)
    """
    import numpy as np
    ca = np.full((len(P), len(Q)), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        d = haversine_km(P[i][0], P[i][1], Q[j][0], Q[j][1])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i-1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j-1), d)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i-1, j), c(i-1, j-1), c(i, j-1)), d)
        else:
            ca[i, j] = float('inf')
        return ca[i, j]
    if not P or not Q:
        return float('inf')
    return c(len(P)-1, len(Q)-1)

def parse_iso_timestamp(iso_str: str) -> Tuple[float, str]:
    """Parse an ISO 8601 string into ``(unix_timestamp, date_key)``.

    *date_key* is ``"YYYYMMDD"`` (UTC).

    Raises ``ValueError`` if the string cannot be parsed.
    """
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.timestamp(), dt.strftime("%Y%m%d")


def unix_to_date_key(unix_ts: float) -> str:
    """Convert a Unix timestamp to a UTC date key ``"YYYYMMDD"``."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y%m%d")


def resolve_port(name: str) -> Tuple[float, float]:
    """Resolve a port name or LOCODE to ``(lat, lon)`` in degrees.

    Accepts UN/LOCODEs (e.g. ``"SGSIN"``) or friendly names
    (e.g. ``"singapore"``).

    Raises ``ValueError`` if not found.
    """
    from shipping_route_predictor.config import WELL_KNOWN_PORTS, PORT_ALIASES

    upper = name.upper().strip()
    if upper in WELL_KNOWN_PORTS:
        return WELL_KNOWN_PORTS[upper]
    lower = name.lower().strip()
    if lower in PORT_ALIASES:
        return WELL_KNOWN_PORTS[PORT_ALIASES[lower]]
    raise ValueError(
        f"Unknown port '{name}'. Use a LOCODE (e.g. SGSIN) or name "
        f"(e.g. singapore). Known: {', '.join(sorted(WELL_KNOWN_PORTS))}"
    )

def total_path_distance_km(
    latlon_path: List[Tuple[float, float]],
) -> float:
    """Sum of haversine segment lengths along a ``[(lat, lon), ...]`` path."""
    if len(latlon_path) < 2:
        return 0.0
    return sum(
        haversine_km(*latlon_path[i], *latlon_path[i + 1])
        for i in range(len(latlon_path) - 1)
    )


def estimate_times(
    latlon_path: List[Tuple[float, float]],
    start_time_iso: str,
    speed_kmh: float = DEFAULT_SPEED_KMH,
) -> List[Dict[str, Any]]:
    """Estimate wall-clock time at each waypoint assuming constant speed.

    Returns list of dicts with ``step``, ``lat``, ``lon``, ``time_iso``,
    ``elapsed_hours``, ``cumulative_km``, ``segment_km``.
    """
    dt = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
    result: List[Dict[str, Any]] = []
    cumulative_km = 0.0
    cumulative_h = 0.0

    for i, (lat, lon) in enumerate(latlon_path):
        if i > 0:
            prev_lat, prev_lon = latlon_path[i - 1]
            seg_km = haversine_km(prev_lat, prev_lon, lat, lon)
            seg_h = seg_km / speed_kmh if speed_kmh > 0 else 0.0
            cumulative_km += seg_km
            cumulative_h += seg_h
        else:
            seg_km = 0.0

        t = dt + timedelta(hours=cumulative_h)
        result.append({
            "step": i,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "time_iso": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_hours": round(cumulative_h, 2),
            "cumulative_km": round(cumulative_km, 1),
            "segment_km": round(seg_km, 1),
        })

    return result


def make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy / tuple types for JSON serialisation."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serialisable(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

