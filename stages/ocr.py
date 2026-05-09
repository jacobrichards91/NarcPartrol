"""
Stage 6 — Address Identification
Two independent tracks, merged into a single result per house:

  Track A — EasyOCR on the full frame.
    House numbers can appear anywhere: fascia, gable, door surround, porch
    post, mailbox, curb.  We scan the whole image and filter for numeric
    strings that look like street addresses (1–5 digits).

  Track B — Overpass API (OpenStreetMap).
    Given the GPS centroid of the house segment, query OSM for all nodes/ways
    with addr:housenumber within OSM_SEARCH_RADIUS_M metres.  No API key;
    free; rate-limiting is handled by caching results per street segment.

Merge logic:
  - OCR number ∈ OSM result set  →  high-confidence match
  - OCR number not in OSM set    →  low-confidence (log it, keep it)
  - No OCR result                →  use nearest OSM address by GPS distance
  - No OSM coverage              →  use raw OCR result or blank
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

import config

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class AddressResult:
    address_ocr: str        # raw OCR winner (empty if none found)
    address_osm: str        # nearest OSM address (empty if no coverage)
    address_final: str      # the value to log and use in filenames
    confidence: str         # "high" | "low" | "osm_only" | "ocr_only" | "none"


# ---------------------------------------------------------------------------
# OCR (Track A)
# ---------------------------------------------------------------------------

_ocr_reader = None  # loaded lazily; requires GPU/CPU depending on availability


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # gpu=True will use CUDA if available; falls back silently to CPU
        _ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _ocr_reader


_ADDRESS_PATTERN = re.compile(r"^\d{1,5}$")


def ocr_address(image_path: Path) -> str:
    """
    Run EasyOCR on the full frame and return the most prominent numeric
    string that matches a house-number pattern, or '' if none found.
    """
    reader = _get_reader()
    try:
        detections = reader.readtext(str(image_path), detail=1)
    except Exception:
        return ""

    candidates: list[tuple[float, str]] = []  # (bbox_area, number_string)
    for (bbox_pts, text, conf) in detections:
        if conf < config.OCR_CONF_MIN:
            continue
        cleaned = text.strip().replace(" ", "").replace("-", "")
        if not _ADDRESS_PATTERN.match(cleaned):
            continue
        # Use bounding-box area as a proxy for prominence
        pts = bbox_pts  # [[x,y], [x,y], [x,y], [x,y]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        candidates.append((area, cleaned))

    if not candidates:
        return ""
    # Return the detection with the largest bounding box (most prominent)
    return max(candidates, key=lambda c: c[0])[1]


# ---------------------------------------------------------------------------
# OSM lookup (Track B)
# ---------------------------------------------------------------------------

# Simple in-memory cache keyed by (rounded_lat, rounded_lon) to avoid
# re-querying the same block for every house on the same street.
_osm_cache: dict[tuple[float, float], list[dict]] = {}

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

_OVERPASS_QUERY = """
[out:json][timeout:{timeout}];
(
  node["addr:housenumber"](around:{radius},{lat},{lon});
  way["addr:housenumber"](around:{radius},{lat},{lon});
);
out center;
"""


def osm_addresses_near(lat: float, lon: float) -> list[dict]:
    """
    Return a list of OSM elements near (lat, lon).  Each element dict has at
    least 'housenumber', 'lat', 'lon'.  Results are cached by ~100m grid cell.
    """
    if lat == 0.0 and lon == 0.0:
        return []

    # Cache key: round to ~0.001° ≈ 111m grid
    key = (round(lat, 3), round(lon, 3))
    if key in _osm_cache:
        return _osm_cache[key]

    query = _OVERPASS_QUERY.format(
        timeout=config.OSM_TIMEOUT_S,
        radius=config.OSM_SEARCH_RADIUS_M,
        lat=lat,
        lon=lon,
    )
    try:
        resp = requests.post(
            _OVERPASS_URL,
            data={"data": query},
            timeout=config.OSM_TIMEOUT_S + 5,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:
        _osm_cache[key] = []
        return []

    parsed = []
    for el in elements:
        hn = el.get("tags", {}).get("addr:housenumber", "")
        if not hn:
            continue
        # OSM ways expose a 'center' key; nodes have lat/lon directly.
        el_lat = el.get("lat") or el.get("center", {}).get("lat", 0.0)
        el_lon = el.get("lon") or el.get("center", {}).get("lon", 0.0)
        parsed.append({"housenumber": hn, "lat": float(el_lat), "lon": float(el_lon)})

    _osm_cache[key] = parsed
    # Polite delay so we don't hammer the free Overpass endpoint
    time.sleep(0.2)
    return parsed


def nearest_osm_address(lat: float, lon: float) -> str:
    """Return the housenumber of the nearest OSM address to (lat, lon)."""
    addresses = osm_addresses_near(lat, lon)
    if not addresses:
        return ""

    import math

    def dist(a: dict) -> float:
        dlat = a["lat"] - lat
        dlon = a["lon"] - lon
        return math.hypot(dlat, dlon)

    closest = min(addresses, key=dist)
    return closest["housenumber"]


# ---------------------------------------------------------------------------
# Merge (Track A + Track B)
# ---------------------------------------------------------------------------

def resolve_address(
    image_path: Path,
    seg_lat: float,
    seg_lon: float,
) -> AddressResult:
    ocr_val = ocr_address(image_path)
    osm_val = nearest_osm_address(seg_lat, seg_lon)

    osm_set = {a["housenumber"] for a in osm_addresses_near(seg_lat, seg_lon)}

    if ocr_val and osm_set:
        if ocr_val in osm_set:
            return AddressResult(ocr_val, osm_val, ocr_val, "high")
        else:
            # OCR found something but it doesn't match OSM — keep OCR but flag
            final = ocr_val
            return AddressResult(ocr_val, osm_val, final, "low")
    elif ocr_val:
        return AddressResult(ocr_val, "", ocr_val, "ocr_only")
    elif osm_val:
        return AddressResult("", osm_val, osm_val, "osm_only")
    else:
        return AddressResult("", "", "", "none")
