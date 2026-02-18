"""Configuration classes for the shipping-route-predictor XGBoost pipeline.

Classes:
    WeatherConfig      — weather data dir, normalization, feature toggles
    EnvConfig          — grid geometry, trajectory paths, vessel info, normalization
    CISCConfig         — CISC rollout selection parameters (α, β, T, N)
    ModelSpec          — single model: name, model_type, path, per-model CISC
    TrajectorySpec     — single trajectory: start/goal rc, time, vessel metadata
    RolloutConfig      — shared rollout core: model + grid + env + weather + rollout limits
    InferenceConfig    — single-route inference: runners, trajectory, speed, output
    CISCTuningConfig   — grid-search tuning of CISC parameters
    TrainingConfig     — XGBoost hyperparams + training controls
    EvalConfig         — multi-model evaluation via RolloutConfig list
    SHAPConfig         — SHAP explainability analysis

Enum flags (on top-level configs):
    grid        — Grid.FINE (128×256) or Grid.COARSE (32×64)

ModelSpec carries per-model identity:
    model_type  — ModelType.SIMPLE (14-dim) or ModelType.FULL (beam + history + weather)
    rollout     — RolloutType.GREEDY or RolloutType.CISC
    path        — model file (auto-filled from defaults)
    cisc        — per-model CISC params (auto-filled for CISC rollouts)

RolloutConfig bundles ModelSpec with the grid environment:
    Reusable sub-config for CISCTuningConfig, EvalConfig, and SHAPConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace as _dc_replace
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, List, Optional, Any, Dict, Tuple
import json
import os


PROJECT_DIR = str(Path(__file__).resolve().parents[1])
_DATA_DIR = os.path.join(PROJECT_DIR, "data")

# Grid construction CSVs (mask, latitude, longitude)
_FINE_CONSTRUCTION_DIR = os.path.join(_DATA_DIR, "fine_128x256_grid/grid_construction_128x256")
_COARSE_CONSTRUCTION_DIR = os.path.join(_DATA_DIR, "coarse_32x64_grid/grid_construction_32x64")

# Trajectory directories
_FINE_TRAJ_DIR = os.path.join(_DATA_DIR, "fine_128x256_grid/trajectories_ais_128x256")
_COARSE_TRAJ_DIR = os.path.join(_DATA_DIR, "coarse_32x64_grid/trajectories_ais_32x64")
_FINE_TRAIN_FILES = (
    os.path.join(_FINE_TRAJ_DIR, "coarse_tracks_from_20230101_to_20230401.json"),
    os.path.join(_FINE_TRAJ_DIR, "coarse_tracks_from_20230401_to_20230701.json"),
    os.path.join(_FINE_TRAJ_DIR, "coarse_tracks_from_20230701_to_20231001.json"),
    os.path.join(_FINE_TRAJ_DIR, "coarse_tracks_from_20231001_to_20240101.json"),
)
_FINE_VAL_FILES = (
    os.path.join(_FINE_TRAJ_DIR, "coarse_tracks_from_20240101_to_20240401.json"),
    os.path.join(_FINE_TRAJ_DIR, "coarse_tracks_from_20240701_to_20241001.json"),
)
_FINE_EVAL_FILES = (
    os.path.join(_FINE_TRAJ_DIR, "coarse_tracks_from_20240401_to_20240701.json"),
    os.path.join(_FINE_TRAJ_DIR, "coarse_tracks_from_20241001_to_20250101.json"),
)
_COARSE_TRAIN_FILES = (
    os.path.join(_COARSE_TRAJ_DIR, "coarse_32x64_tracks_from_20230101_to_20230401.json"),
    os.path.join(_COARSE_TRAJ_DIR, "coarse_32x64_tracks_from_20230401_to_20230701.json"),
    os.path.join(_COARSE_TRAJ_DIR, "coarse_32x64_tracks_from_20230701_to_20231001.json"),
    os.path.join(_COARSE_TRAJ_DIR, "coarse_32x64_tracks_from_20231001_to_20240101.json"),
)
_COARSE_VAL_FILES = (
    os.path.join(_COARSE_TRAJ_DIR, "coarse_32x64_tracks_from_20240101_to_20240401.json"),
    os.path.join(_COARSE_TRAJ_DIR, "coarse_32x64_tracks_from_20240701_to_20241001.json"),
)
_COARSE_EVAL_FILES = (
    os.path.join(_COARSE_TRAJ_DIR, "coarse_32x64_tracks_from_20240401_to_20240701.json"),
    os.path.join(_COARSE_TRAJ_DIR, "coarse_32x64_tracks_from_20241001_to_20250101.json"),
)

# Depth CSVs (one level above grid_construction, at grid root)
_FINE_DEPTH_CSV = os.path.join(_DATA_DIR, "fine_128x256_grid/depth_128x256.csv")
_COARSE_DEPTH_CSV = os.path.join(_DATA_DIR, "coarse_32x64_grid/depth_32x64.csv")

# Weather directories
_FINE_WEATHER_DIR = os.path.join(_DATA_DIR, "fine_128x256_grid/weather_copernicus_marine_128x256")
_COARSE_WEATHER_DIR = os.path.join(_DATA_DIR, "coarse_32x64_grid/weather_copernicus_marine_32x64")

R_EARTH_KM: float = 6371.0088
KNOT_TO_KMH: float = 1.852
DEFAULT_SPEED_KN: float = 14.5     # "other" operator, 150-200 m LOA bin
DEFAULT_SPEED_KMH: float = DEFAULT_SPEED_KN * KNOT_TO_KMH

# ═══════════════════════════════════════════════════════════════════════════
# Shared colour palettes (used by visualize, SHAP plots)
# ═══════════════════════════════════════════════════════════════════════════

CATEGORY_COLORS: Dict[str, str] = {
    "vectors":                "#1f77b4",
    "flags":                  "#ff7f0e",
    "indices":                "#2ca02c",
    "depth":                  "#d62728",
    "vessel_characteristics": "#9467bd",
    "wind":                   "#8c564b",
    "current":                "#e377c2",
    "wave":                   "#7f7f7f",
    "beam":                   "#bcbd22",
    "history":                "#17becf",
}

COLOR_CYCLE: Tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
)

# ═══════════════════════════════════════════════════════════════════════════
# Well-known ports  —  (lat, lon) by UN/LOCODE
# ═══════════════════════════════════════════════════════════════════════════

WELL_KNOWN_PORTS: Dict[str, Tuple[float, float]] = {
    # Asia
    "SGSIN": (1.3, 103.8),        # Singapore
    "CNSHA": (31.4, 121.5),       # Shanghai
    "KRPUS": (35.1, 129.0),       # Busan
    "HKHKG": (22.3, 114.2),       # Hong Kong
    "JPTYO": (35.6, 139.8),       # Tokyo
    "TWKHH": (22.6, 120.3),       # Kaohsiung
    "VNSGN": (10.8, 106.7),       # Ho Chi Minh City
    # Europe
    "DKCPH": (55.7, 12.6),        # Copenhagen
    "NLRTM": (51.9, 4.5),         # Rotterdam
    "DEHAM": (53.5, 10.0),        # Hamburg
    "GBFXT": (51.4, 1.4),         # Felixstowe
    "ESALG": (36.7, -2.5),        # Algeciras
    "GRPIR": (37.9, 23.6),        # Piraeus
    "ITGOA": (44.4, 8.9),         # Genoa
    # Americas
    "USLAX": (33.7, -118.3),      # Los Angeles / Long Beach
    "USOAK": (37.8, -122.3),      # Oakland
    "BRSSZ": (23.9, -46.3),       # Santos
    "PAMIT": (9.4, -79.9),        # Panama (Miraflores)
    "MXZLO": (19.1, -104.3),      # Manzanillo
    # Middle East / Africa
    "AEJEA": (25.0, 55.1),        # Jebel Ali
    "EGPSD": (31.3, 32.3),        # Port Said
    "ZADUR": (-29.9, 31.0),       # Durban
    # Oceania
    "AUMEL": (-37.8, 144.9),      # Melbourne
    "NZAKL": (-36.8, 174.8),      # Auckland
}

PORT_ALIASES: Dict[str, str] = {
    "singapore": "SGSIN",
    "copenhagen": "DKCPH",
    "shanghai": "CNSHA",
    "busan": "KRPUS",
    "hongkong": "HKHKG",
    "hong kong": "HKHKG",
    "tokyo": "JPTYO",
    "rotterdam": "NLRTM",
    "hamburg": "DEHAM",
    "losangeles": "USLAX",
    "los angeles": "USLAX",
    "longbeach": "USLAX",
    "long beach": "USLAX",
    "oakland": "USOAK",
    "santos": "BRSSZ",
    "panama": "PAMIT",
    "piraeus": "GRPIR",
    "genoa": "ITGOA",
    "algeciras": "ESALG",
    "felixstowe": "GBFXT",
    "durban": "ZADUR",
    "melbourne": "AUMEL",
    "auckland": "NZAKL",
    "jebel ali": "AEJEA",
    "portsaid": "EGPSD",
    "port said": "EGPSD",
    "kaohsiung": "TWKHH",
    "ho chi minh": "VNSGN",
    "manzanillo": "MXZLO",
}


def _resolve_port(name: str) -> Tuple[float, float]:
    """Resolve a port name or LOCODE to ``(lat, lon)``.

    Raises ``ValueError`` if not found.
    """
    upper = name.upper().strip()
    if upper in WELL_KNOWN_PORTS:
        return WELL_KNOWN_PORTS[upper]
    lower = name.lower().strip()
    if lower in PORT_ALIASES:
        return WELL_KNOWN_PORTS[PORT_ALIASES[lower]]
    raise ValueError(
        f"Unknown port {name!r}. Known: {', '.join(sorted(WELL_KNOWN_PORTS))}"
    )


class GridResolution(StrEnum):
    FINE = "fine"
    COARSE = "coarse"


class ModelType(StrEnum):
    SIMPLE = "simple"
    FULL = "full"


class RolloutType(StrEnum):
    GREEDY = "greedy"
    CISC = "cisc"

_PRETRAINED_MODELS_DIR = os.path.join(PROJECT_DIR, "models_pretrained")
_DEFAULT_MODEL_DIRS = {
    (GridResolution.FINE,   ModelType.SIMPLE): os.path.join(_PRETRAINED_MODELS_DIR, "xgboost_fine_simple_128x256_d12_n500_20260108_193015"),
    (GridResolution.FINE,   ModelType.FULL):   os.path.join(_PRETRAINED_MODELS_DIR, "xgboost_fine_full_128x256_d12_n500_20260109_234538"),
    (GridResolution.COARSE, ModelType.SIMPLE): os.path.join(_PRETRAINED_MODELS_DIR, "xgboost_coarse_simple_32x64_d12_n500_20260108_165708"),
    (GridResolution.COARSE, ModelType.FULL):   os.path.join(_PRETRAINED_MODELS_DIR, "xgboost_coarse_full_32x64_d12_n500_20260109_071709"),
}

# Default model files (dir + filename)
_DEFAULT_MODEL_FILES = {
    k: os.path.join(v, "xgboost_model_final.ubj") for k, v in _DEFAULT_MODEL_DIRS.items()
}

# Tuned CISC parameters per (grid, model_type) — from pipeline runs Jan 8-10, 2026
_DEFAULT_CISC = {
    (GridResolution.FINE,   ModelType.SIMPLE): {"temperature": 1.1, "beta": 1.0,  "alpha": 1.0, "n_rollouts": 10},
    (GridResolution.FINE,   ModelType.FULL):   {"temperature": 0.9, "beta": 0.25, "alpha": 1.0, "n_rollouts": 10},
    (GridResolution.COARSE, ModelType.SIMPLE): {"temperature": 1.1, "beta": 0.75, "alpha": 1.0, "n_rollouts": 10},
    (GridResolution.COARSE, ModelType.FULL):   {"temperature": 0.8, "beta": 0.0,  "alpha": 1.0, "n_rollouts": 10},
}

@dataclass
class WeatherConfig:
    """Weather data directory, normalization constants and speed estimation.
    """
    grid: GridResolution = GridResolution.FINE
    weather_dir: Optional[str] = None

    # Spatial weather sampling (full model only)
    num_beam_slots: int = 8
    num_history_slots: int = 6
    beam_sector_deg: float = 180.0
    beam_radius_steps: int = 10
    beam_stride: int = 3            # keep every k-th BFS depth
    time_stride: int = 2            # keep every N-th history step
    padding_value: float = -1000.0

    # Normalization means (features are divided by these — "scale" policy)
    wind_mean_kmh: float = 26.316
    current_mean_kmh: float = 0.5976
    wave_mean_m: float = 2.45
    vessel_vector_norm_km: float = 1190.0

    # Speed estimation for time-aware weather lookups
    fixed_speed_kmh: float = DEFAULT_SPEED_KMH

    def __post_init__(self):
        if self.weather_dir is None:
            self.weather_dir = _COARSE_WEATHER_DIR if self.grid == GridResolution.COARSE else _FINE_WEATHER_DIR

@dataclass
class EnvConfig:
    """Grid geometry, trajectory paths, vessel info and normalization.

    Set ``grid=Grid.COARSE`` to select 32×64 grid paths and trajectory files;
    ``Grid.FINE`` (default) uses the 128×256 fine grid.  Fields left as ``None``
    or ``0`` are filled from the ``grid`` flag — any explicit value is kept.
    """
    grid: GridResolution = GridResolution.FINE
    project_dir: str = PROJECT_DIR
    mask_csv: Optional[str] = None
    latitude_csv: Optional[str] = None
    longitude_csv: Optional[str] = None
    depth_csv: Optional[str] = None
    grid_height: int = 0
    grid_width: int = 0
    train_track_files: Optional[List[str]] = None
    val_track_files: Optional[List[str]] = None
    eval_track_files: Optional[List[str]] = None

    # Trajectory filtering
    max_traj_days: float = 42.0
    max_coord_jump: int = 0          # 0 = auto: coarse→3, fine→5

    # Normalization
    depth_norm_policy: str = "median"   # mean | median
    depth_median_m: float = 3990.81
    vessel_median_loa_m: float = 184.8
    goal_vector_norm_km: float = 2990.0

    # Default vessel metadata (used when per-vessel data is unavailable)
    vessels_json: Optional[str] = os.path.join(_DATA_DIR, "vessel_data.json")
    operator_speed_json: Optional[str] = os.path.join(_DATA_DIR, "operator_speed_kn_by_loa_bin.json")
    default_vessel_loa: float = 184.8     # median LOA across AIS fleet (= vessel_median_loa_m)
    default_vessel_company: str = "other"

    def __post_init__(self):
        coarse = self.grid == GridResolution.COARSE
        cdir = _COARSE_CONSTRUCTION_DIR if coarse else _FINE_CONSTRUCTION_DIR
        self.mask_csv = self.mask_csv or os.path.join(cdir, "mask.csv")
        self.latitude_csv = self.latitude_csv or os.path.join(cdir, "lat_mapping_table.csv")
        self.longitude_csv = self.longitude_csv or os.path.join(cdir, "longitude_indices.csv")
        self.depth_csv = self.depth_csv or (_COARSE_DEPTH_CSV if coarse else _FINE_DEPTH_CSV)
        if not self.grid_height:
            self.grid_height = 32 if coarse else 128
        if not self.grid_width:
            self.grid_width = 64 if coarse else 256
        # Trajectory file defaults
        if self.train_track_files is None:
            self.train_track_files = list(_COARSE_TRAIN_FILES if coarse else _FINE_TRAIN_FILES)
        if self.val_track_files is None:
            self.val_track_files = list(_COARSE_VAL_FILES if coarse else _FINE_VAL_FILES)
        if self.eval_track_files is None:
            self.eval_track_files = list(_COARSE_EVAL_FILES if coarse else _FINE_EVAL_FILES)
        if not self.max_coord_jump:
            self.max_coord_jump = 3 if coarse else 5

@dataclass
class CISCConfig:
    """CISC (Consensus Inference with Stochastic Competition) parameters.

    Used by EvalConfig, SHAPConfig, and CISCTuningConfig whenever
    rollout == RolloutType.CISC.
    """
    n_rollouts: int = 10              # stochastic rollouts per step
    alpha: float = 1.0                # self-consistency weight
    beta: float = 0.0                 # probability weight
    temperature: float = 1.0          # softmax temperature for sampling
    hybrid_rollout: bool = True       # greedy-first, then stochastic-until-success


@dataclass
class ModelSpec:
    """A single model to evaluate or explain.

    Fields ``path`` and ``cisc`` can be left as ``None`` — they are
    auto-populated in ``EvalConfig.__post_init__`` from the tuned
    defaults using ``(grid, model_type)``.
    """
    name: str                                     # display name (e.g. "xgb_coarse_simple_cisc")
    model_type: ModelType                         # SIMPLE or FULL — determines feature set
    # rollout: RolloutType = RolloutType.CISC       # REMOVED: rollout type is now specified at call time
    path: Optional[str] = None                    # model file; None → default for (grid, model_type)
    cisc: Optional[CISCConfig] = None             # per-model CISC; None → tuned default (CISC only)


@dataclass
class TrajectorySpec:
    """A specific trajectory for inference or SHAP analysis.

    Bundles start/goal grid cells, departure time, and vessel metadata.
    """
    name: str                                     # display label (e.g. "Oakland→Busan")
    start_rc: Tuple[int, int]                     # (row, col) on the grid
    goal_rc: Tuple[int, int]                      # (row, col) on the grid
    start_time: str                               # ISO 8601 departure time
    vessel_loa: float                             # length overall (metres); 0 → use median
    vessel_company: str                           # operator name
    vessel_speed_kmh: Optional[float] = None      # speed (km/h); None → use fixed fallback
    imo: Optional[int] = None                     # IMO number (informational)


# Pre-defined demo trajectories (fine grid)
_TRAJECTORY_OAKLAND_BUSAN = TrajectorySpec(
    name="Oakland→Busan",
    start_rc=(90, 41),        # Oakland   (37.3°N, 121.6°W)
    goal_rc=(88, 219),        # Busan     (34.5°N, 128.7°E)
    start_time="2024-08-09T12:11:00Z",
    vessel_loa=334.8,
    vessel_company="evergreen",
    imo=9595450,
)
_TRAJECTORY_SHANGHAI_LONGBEACH = TrajectorySpec(
    name="Shanghai→Long Beach",
    start_rc=(86, 214),       # Shanghai  (31.6°N, 121.6°E)
    goal_rc=(87, 43),         # Long Beach (33.1°N, 118.8°W)
    start_time="2024-03-30T06:55:00Z",
    vessel_loa=399.0,
    vessel_company="msc",
    imo=9606314,
)


# ═══════════════════════════════════════════════════════════════════════════
# TrainingConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """XGBoost training configuration.

    Uses ``ModelSpec`` to identify the model being trained (name +
    model_type).  ``ModelSpec.path``, ``rollout`` and ``cisc`` are
    unused during training — they are filled in post-training by
    eval/SHAP configs.

    Sub-configs ``env`` and ``weather`` are auto-constructed from
    ``grid`` and ``model.model_type`` when not supplied.  ``weather``
    is ``None`` for simple models.
    """
    grid: GridResolution = GridResolution.FINE
    device: str = "cpu"          # cpu | cuda

    # Model identity (None → default simple model for this grid)
    model: Optional[ModelSpec] = None

    # Sub-configs (None → auto-constructed from flags)
    env: Optional[EnvConfig] = None
    weather: Optional[WeatherConfig] = None  # None when model_type=SIMPLE

    # --- XGBoost hyperparameters ---
    n_estimators: int = 500
    max_depth: int = 12
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0      # L1 regularization
    reg_lambda: float = 1.0     # L2 regularization

    # --- Training data controls ---
    train_limit: Optional[int] = 1100
    val_limit: Optional[int] = 1000
    shuffle_files: bool = True
    same_day_block_size: Optional[int] = 2
    day_progression: str = "random"  # random | sequential

    # --- Training loop ---
    n_jobs: int = -1             # all available cores
    random_state: int = 42
    early_stopping_rounds: Optional[int] = 50
    eval_metric: List[str] = field(default_factory=lambda: ["mlogloss", "merror"])

    # --- Logging ---
    log_level: str = "INFO"
    wandb_mode: str = "disabled"   # disabled | online | offline
    wandb_project: str = "shipping-route-predictor"
    wandb_entity: Optional[str] = None

    # Output
    models_dir: str = os.path.join(PROJECT_DIR, "models")




    def __post_init__(self):
        if self.model is None:
            self.model = ModelSpec(
                name=f"xgb_{self.grid}_simple",
                model_type=ModelType.SIMPLE,
            )
        if self.env is None:
            self.env = EnvConfig(grid=self.grid)
        else:
            self.env.grid = self.grid
        if self.weather is None and self.model.model_type == ModelType.FULL:
            self.weather = WeatherConfig(grid=self.grid)
        elif self.weather is not None:
            self.weather.grid = self.grid
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"


# ═══════════════════════════════════════════════════════════════════════════
# RolloutConfig  — reusable core for running a single model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RolloutConfig:
    """Shared rollout core: model + grid environment + rollout limits.

    Embeddable in EvalConfig, SHAPConfig, and CISCTuningConfig.
    For standalone single-route inference, see ``InferenceConfig``.

    ``model.path`` and ``model.cisc`` are auto-populated from tuned
    defaults when left as ``None``.
    """
    grid: GridResolution = GridResolution.FINE
    device: str = "cpu"

    # Model (None → default simple+CISC for this grid)
    model: Optional[ModelSpec] = None

    # Sub-configs (auto-constructed from grid + model_type)
    env: Optional[EnvConfig] = None
    weather: Optional[WeatherConfig] = None

    # Rollout limits
    max_rollout_steps: int = 400

    def __post_init__(self):
        if self.model is None:
            self.model = ModelSpec(
                name=f"xgb_{self.grid}_simple",
                model_type=ModelType.SIMPLE,
            )
        if self.model.path is None:
            self.model.path = _DEFAULT_MODEL_FILES.get((self.grid, self.model.model_type))
        if self.model.cisc is None:
            cisc_kw = _DEFAULT_CISC.get((self.grid, self.model.model_type), {})
            self.model.cisc = CISCConfig(**cisc_kw)
        if self.env is None:
            self.env = EnvConfig(grid=self.grid)
        else:
            self.env.grid = self.grid
        if self.weather is None and self.model.model_type == ModelType.FULL:
            self.weather = WeatherConfig(grid=self.grid)
        elif self.weather is not None:
            self.weather.grid = self.grid
        if self.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"


@dataclass
class InferenceConfig:
    """Single-route inference configuration.

    Bundles grid resolution, runner selection, trajectory defaults,
    speed, rollout limits, and output directory for one prediction run.

    Default trajectory: Shanghai → Long Beach.
    Default runners: simple_greedy + shortest_path.
    """

    VALID_RUNNERS: ClassVar[Tuple[str, ...]] = (
        "simple_greedy", "simple_cisc",
        "full_greedy", "full_cisc",
        "shortest_path", "company_baseline",
    )
    DEFAULT_RUNNERS: ClassVar[Tuple[str, ...]] = ("simple_greedy", "shortest_path")

    # Human-readable labels for each runner (used by logs)
    RUNNER_LABELS: ClassVar[Dict[str, str]] = {
        "simple_greedy":    "XGB Simple Greedy",
        "simple_cisc":      "XGB Simple CISC",
        "full_greedy":      "XGB Full Greedy",
        "full_cisc":        "XGB Full CISC",
        "shortest_path":    "Shortest Path (Dijkstra)",
        "company_baseline": "Company Baseline",
    }

    # Per-runner colour (hex) for route visualisation
    RUNNER_COLORS: ClassVar[Dict[str, str]] = {
        "simple_greedy":    "#1f77b4",
        "simple_cisc":      "#ff7f0e",
        "full_greedy":      "#2ca02c",
        "full_cisc":        "#d62728",
        "shortest_path":    "#9467bd",
        "company_baseline": "#8c564b",
    }

    # Known vessel operator names accepted by the pipeline
    KNOWN_OPERATORS: ClassVar[Tuple[str, ...]] = (
        "msc", "maersk", "cma cgm", "cosco", "hapag-lloyd",
        "evergreen", "one", "yang ming", "hmm", "zim",
        "pil", "wan hai", "other",
    )

    grid: GridResolution = GridResolution.COARSE

    # Which models/baselines to run
    runners: List[str] = field(default_factory=lambda: list(InferenceConfig.DEFAULT_RUNNERS))

    # Default trajectory
    trajectory: TrajectorySpec = field(
        default_factory=lambda: _dc_replace(_TRAJECTORY_SHANGHAI_LONGBEACH),
    )

    # Vessel speed for time estimation
    speed_kmh: float = DEFAULT_SPEED_KMH

    # Optional start/goal overrides (lat/lon or port name)
    start_port: Optional[str] = "CNSHA"
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    goal_port: Optional[str] = "USLAX"
    goal_lat: Optional[float] = None
    goal_lon: Optional[float] = None

    # Optional trajectory metadata overrides
    start_time_override: Optional[str] = None
    vessel_loa_override: Optional[float] = None
    vessel_company_override: Optional[str] = None

    # Output
    output_dir: Optional[str] = None
    visualize: bool = True
    log_level: str = "INFO"

    # ── computed by __post_init__ ──────────────────────────────
    start_latlon: Optional[Tuple[float, float]] = field(init=False, default=None)
    goal_latlon: Optional[Tuple[float, float]] = field(init=False, default=None)
    start_label: str = field(init=False, default="")
    goal_label: str = field(init=False, default="")

    def __post_init__(self):
        # Validate runners
        for r in self.runners:
            if r not in self.VALID_RUNNERS:
                raise ValueError(
                    f"Unknown runner {r!r}. Valid: {', '.join(self.VALID_RUNNERS)}"
                )

        traj = self.trajectory

        # Resolve start: explicit lat/lon > port name > trajectory default
        if self.start_lat is not None and self.start_lon is not None:
            self.start_latlon = (self.start_lat, self.start_lon)
            self.start_label = f"({self.start_lat:.2f}, {self.start_lon:.2f})"
        elif self.start_port is not None:
            self.start_latlon = _resolve_port(self.start_port)
            self.start_label = self.start_port.upper()
        else:
            self.start_latlon = None  # resolved by Inference from trajectory start_rc
            self.start_label = traj.name.split("→")[0] if "→" in traj.name else "start"

        # Resolve goal: explicit lat/lon > port name > trajectory default
        if self.goal_lat is not None and self.goal_lon is not None:
            self.goal_latlon = (self.goal_lat, self.goal_lon)
            self.goal_label = f"({self.goal_lat:.2f}, {self.goal_lon:.2f})"
        elif self.goal_port is not None:
            self.goal_latlon = _resolve_port(self.goal_port)
            self.goal_label = self.goal_port.upper()
        else:
            self.goal_latlon = None
            self.goal_label = traj.name.split("→")[1] if "→" in traj.name else "goal"

        # Validate vessel company override
        if self.vessel_company_override is not None:
            vc = self.vessel_company_override.strip().lower()
            if vc and vc not in self.KNOWN_OPERATORS:
                raise ValueError(
                    f"Unknown vessel company {self.vessel_company_override!r}. "
                    f"Known: {', '.join(self.KNOWN_OPERATORS)}"
                )

        # Apply trajectory metadata overrides
        overrides: Dict[str, Any] = {}
        if self.start_time_override is not None:
            overrides["start_time"] = self.start_time_override
        if self.vessel_loa_override is not None:
            overrides["vessel_loa"] = self.vessel_loa_override
        if self.vessel_company_override is not None:
            overrides["vessel_company"] = self.vessel_company_override
        if overrides:
            self.trajectory = _dc_replace(traj, **overrides)

        # Auto-generate output_dir if not specified
        if self.output_dir is None:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            route_slug = (
                f"{self.start_label}_{self.goal_label}"
                .replace(" ", "").replace("(", "").replace(")", "").replace(",", "_")
            )
            self.output_dir = os.path.join(
                PROJECT_DIR, "results", "inference_results",
                f"{route_slug}_{self.grid}_{ts}",
            )


@dataclass
class CISCTuningConfig:
    """Grid-search tuning of CISC (α, β, T, N) on a validation set."""

    # Core rollout settings (model, grid, env, weather, device)
    rollout: RolloutConfig = field(default_factory=lambda: RolloutConfig(
        grid=GridResolution.COARSE,
        model=ModelSpec(name="xgb_coarse_simple", model_type=ModelType.SIMPLE),
    ))

    # Trajectory source
    dataset: str = "val"
    limit: int = 10
    seed: int = 42

    # Search grid
    temperatures: List[float] = field(default_factory=lambda: [0.8, 0.9, 1.0, 1.1,1.2])
    alphas: List[float] = field(default_factory=lambda: [0.0,0.1,1.0])
    betas: List[float] = field(default_factory=lambda: [0.0,0.2,0.8,1.0,1.2,1.8,2.0])
    n_rollouts_list: List[int] = field(default_factory=lambda: [5,10,15])

    # Output
    output_dir: str = os.path.join(PROJECT_DIR, "results", "cisc_tuning_results")
    log_level: str = "INFO"

@dataclass
class EvalConfig:
    """Multi-model evaluation configuration for XGBoost models.

    Models are specified via ``rollouts`` — a list of ``RolloutConfig``
    instances, each carrying a model, rollout strategy, grid environment,
    and rollout limits.  This allows evaluating simple and full models
    with greedy and CISC rollouts side-by-side.

    When ``rollouts`` is empty, four default configs are generated:
    simple×greedy, simple×cisc, full×greedy, full×cisc.
    """
    grid: GridResolution = GridResolution.FINE
    rollouts: List[RolloutConfig] = field(default_factory=list)
    dataset: str = "eval"         # train | val | eval
    limit: Optional[int] = 50   # max trajectories to evaluate
    extra_steps_over_gt_length: int = 20  # max_rollout_steps = gt_length + this
    include_company_baseline: bool = True
    include_shortest_path_baseline: bool = True
    output_dir: str = os.path.join(PROJECT_DIR, "results", "evaluation_results")
    progress_save_interval: int = 50
    visualize: bool = True
    histogram_bins: int = 50
    histogram_metric: str = "frechet_km"
    log_level: str = "INFO"
    wandb_on: bool = False
    wandb_project: str = "shipping-route-predictor"

    def __post_init__(self):
        if not self.rollouts:
            self.rollouts = [
                RolloutConfig(
                    grid=self.grid,
                    model=ModelSpec(name=f"xgb_{self.grid}_simple_greedy", model_type=ModelType.SIMPLE),
                ),
                RolloutConfig(
                    grid=self.grid,
                    model=ModelSpec(name=f"xgb_{self.grid}_simple_cisc", model_type=ModelType.SIMPLE),
                ),
                RolloutConfig(
                    grid=self.grid,
                    model=ModelSpec(name=f"xgb_{self.grid}_full_greedy", model_type=ModelType.FULL),
                ),
                RolloutConfig(
                    grid=self.grid,
                    model=ModelSpec(name=f"xgb_{self.grid}_full_cisc", model_type=ModelType.FULL),
                ),
            ]

@dataclass
class SHAPConfig:
    """SHAP explainability analysis for XGBoost models.

    Like EvalConfig, uses a list of ``RolloutConfig`` instances to
    run SHAP on multiple models.  Defaults to simple + full models
    for the selected grid, both using CISC rollout.
    """
    grid: GridResolution = GridResolution.FINE

    # --- Models to explain (each is a self-contained rollout config) ---
    rollouts: List[RolloutConfig] = field(default_factory=list)

    # --- Trajectory source ---
    dataset: str = "eval"         # train | val | eval
    limit: int = 30               # trajectories per model
    seed: int = 42

    # --- Named trajectories for single-route SHAP ---
    trajectories: List[TrajectorySpec] = field(default_factory=lambda: [
        _dc_replace(_TRAJECTORY_OAKLAND_BUSAN),
        _dc_replace(_TRAJECTORY_SHANGHAI_LONGBEACH),
    ])

    # --- Output ---
    output_dir: str = os.path.join(PROJECT_DIR, "results", "shap_results")
    background_samples: int = 200
    log_level: str = "INFO"

    # --- Feature categories for aggregated SHAP importance ----------
    #     Simple models only use the base 14 features (no weather).
    SIMPLE_FEATURE_CATEGORIES: Dict[str, List[str]] = field(default=None, repr=False)
    FULL_FEATURE_CATEGORIES: Dict[str, List[str]] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.rollouts:
            self.rollouts = [
                RolloutConfig(
                    grid=self.grid,
                    model=ModelSpec(name=f"xgb_{self.grid}_simple_cisc", model_type=ModelType.SIMPLE),
                ),
                RolloutConfig(
                    grid=self.grid,
                    model=ModelSpec(name=f"xgb_{self.grid}_full_cisc", model_type=ModelType.FULL),
                ),
            ]
        if self.SIMPLE_FEATURE_CATEGORIES is None:
            self.SIMPLE_FEATURE_CATEGORIES = {
                "vectors": ["dgoal_east", "dgoal_north"],
                "flags": ["here_flag", "start_flag", "goal_flag"],
                "indices": ["cur_row", "cur_col", "start_row", "start_col", "goal_row", "goal_col"],
                "depth": ["depth"],
                "vessel_characteristics": ["loa_norm", "company_id"],
            }
        if self.FULL_FEATURE_CATEGORIES is None:
            self.FULL_FEATURE_CATEGORIES = {
                **self.SIMPLE_FEATURE_CATEGORIES,
                "wind": ["wind_u", "wind_v"],
                "current": ["current_u", "current_v"],
                "wave": ["wave_height"],
                "beam": ["beam_"],
                "history": ["hist_"],
            }

    def feature_categories(self, model_type: ModelType) -> Dict[str, List[str]]:
        """Return the feature-category mapping for the given model type."""
        if model_type == ModelType.SIMPLE:
            return self.SIMPLE_FEATURE_CATEGORIES
        return self.FULL_FEATURE_CATEGORIES


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclass instances to JSON-friendly dicts."""
    if isinstance(obj, (WeatherConfig, EnvConfig, CISCConfig, ModelSpec,
                        TrajectorySpec, RolloutConfig, InferenceConfig,
                        CISCTuningConfig, TrainingConfig, EvalConfig, SHAPConfig,
                        )):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def save_config(cfg, directory: str | Path, filename: str = "config.json") -> Path:
    """Persist any config dataclass to JSON."""
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    out = p / filename
    with open(out, "w") as f:
        json.dump(_to_jsonable(cfg), f, indent=2, sort_keys=True)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON config file into a plain dict."""
    with open(path, "r") as f:
        return json.load(f)


__all__ = [
    "GridResolution",
    "ModelType",
    "RolloutType",
    "KNOT_TO_KMH",
    "DEFAULT_SPEED_KN",
    "DEFAULT_SPEED_KMH",
    "WELL_KNOWN_PORTS",
    "PORT_ALIASES",
    "WeatherConfig",
    "EnvConfig",
    "CISCConfig",
    "ModelSpec",
    "TrajectorySpec",
    "TrainingConfig",
    "RolloutConfig",
    "InferenceConfig",
    "CISCTuningConfig",
    "EvalConfig",
    "SHAPConfig",
    "save_config",
    "load_config",
    "CATEGORY_COLORS",
    "COLOR_CYCLE",
]
