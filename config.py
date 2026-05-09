"""
Configuration loader for NarcPartrol.

Resolution order (first match wins):
  1. ~/Documents/NarcPartrol/narcpartrol.toml   ← user edits this
  2. ~/.config/narcpartrol/narcpartrol.toml      ← XDG fallback
  3. Built-in defaults below                     ← always works, no install needed

All values are exposed as module-level constants so the rest of the codebase
can do  `import config; config.JPEG_QUALITY`  without caring where the value
came from.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


# ---------------------------------------------------------------------------
# Built-in defaults  (mirrors narcpartrol.toml so the two never drift)
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "gps": {
        "min_lot_frontage_m": 15.0,
    },
    "sampling": {
        "fps": 4,
    },
    "detection": {
        "model": "yolov8x-oiv7.pt",
        "building_conf_min": 0.30,
        "building_coverage_min": 0.15,
        "building_classes": ["Building", "House", "Tower", "Skyscraper", "Shed"],
        "occluder_classes": ["Car", "Truck", "Bus", "Van", "Person", "Tree", "Motorcycle"],
        "batch_size": 16,
    },
    "scoring": {
        "top_n_candidates": 3,
        "w_sharpness": 0.30,
        "w_coverage": 0.25,
        "w_frontality": 0.25,
        "w_exposure": 0.20,
        "occlusion_penalty_weight": 0.50,
    },
    "ocr": {
        "search_radius_m": 75,
        "timeout_s": 10,
        "conf_min": 0.40,
    },
    "cloud": {
        "quality_threshold": 0.45,
        "model": "claude-haiku-4-5-20251001",
        "max_image_bytes": 5 * 1024 * 1024,
    },
    "export": {
        "crop_padding": 0.12,
        "jpeg_quality": 95,
        "output_subdir": "snapshots",
        "log_filename": "log.csv",
    },
}

# ---------------------------------------------------------------------------
# Locate and load the user config file
# ---------------------------------------------------------------------------
_SEARCH_PATHS: list[Path] = [
    Path.home() / "Documents" / "NarcPartrol" / "narcpartrol.toml",
    Path.home() / ".config" / "narcpartrol" / "narcpartrol.toml",
]

_CONFIG_FILE: Path | None = None
_user: dict = {}

for _candidate in _SEARCH_PATHS:
    if _candidate.exists():
        _CONFIG_FILE = _candidate
        with open(_candidate, "rb") as _fh:
            _user = tomllib.load(_fh)
        break


def _get(section: str, key: str):
    """Return user value if present, otherwise the built-in default."""
    return _user.get(section, {}).get(key, _DEFAULTS[section][key])


def config_file_path() -> Path | None:
    """Returns the path of the loaded config file, or None if using defaults."""
    return _CONFIG_FILE


# ---------------------------------------------------------------------------
# Public constants — same names as before so no other file changes
# ---------------------------------------------------------------------------

# GPS Segmentation (S2)
MIN_LOT_FRONTAGE_M: float       = _get("gps", "min_lot_frontage_m")

# Frame Sampling (S3)
SAMPLE_FPS: int                 = _get("sampling", "fps")

# Building Detection (S4)
YOLO_MODEL: str                 = _get("detection", "model")
BUILDING_CONF_MIN: float        = _get("detection", "building_conf_min")
BUILDING_COVERAGE_MIN: float    = _get("detection", "building_coverage_min")
BUILDING_CLASS_NAMES: frozenset = frozenset(_get("detection", "building_classes"))
OCCLUDER_CLASS_NAMES: frozenset = frozenset(_get("detection", "occluder_classes"))
YOLO_BATCH_SIZE: int            = _get("detection", "batch_size")

# Quality Scoring (S5)
TOP_N_CANDIDATES: int           = _get("scoring", "top_n_candidates")
W_SHARPNESS: float              = _get("scoring", "w_sharpness")
W_COVERAGE: float               = _get("scoring", "w_coverage")
W_FRONTALITY: float             = _get("scoring", "w_frontality")
W_EXPOSURE: float               = _get("scoring", "w_exposure")
OCCLUSION_PENALTY_WEIGHT: float = _get("scoring", "occlusion_penalty_weight")

# Address / OCR (S6)
OSM_SEARCH_RADIUS_M: int        = _get("ocr", "search_radius_m")
OSM_TIMEOUT_S: int              = _get("ocr", "timeout_s")
OCR_CONF_MIN: float             = _get("ocr", "conf_min")

# Cloud Quality Gate (S7)
QUALITY_THRESHOLD: float        = _get("cloud", "quality_threshold")
CLOUD_MODEL: str                = _get("cloud", "model")
CLOUD_MAX_IMAGE_BYTES: int      = _get("cloud", "max_image_bytes")

# Export (S8)
CROP_PADDING: float             = _get("export", "crop_padding")
JPEG_QUALITY: int               = _get("export", "jpeg_quality")
OUTPUT_SUBDIR: str              = _get("export", "output_subdir")
LOG_FILENAME: str               = _get("export", "log_filename")
