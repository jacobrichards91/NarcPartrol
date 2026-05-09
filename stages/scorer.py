"""
Stage 5 — Frame Quality Scoring
Scores each FrameDetections on four axes, then subtracts an occlusion penalty.
Returns top-N candidates per house segment, sorted best-first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config
from stages.detector import BBox, FrameDetections


@dataclass
class ScoredFrame:
    frame_path: Path
    house_index: int
    timestamp_s: float
    total_score: float
    sharpness: float
    coverage: float
    frontality: float
    exposure: float
    occlusion_fraction: float
    building_bbox: BBox | None   # best building box used for cropping later


def score_frame(fd: FrameDetections, timestamp_s: float, house_index: int) -> ScoredFrame:
    bb = fd.best_building

    img_bgr = cv2.imread(str(fd.frame_path))
    if img_bgr is None:
        return _zero_score(fd.frame_path, house_index, timestamp_s, bb)

    h, w = img_bgr.shape[:2]

    # ---- Sharpness: Laplacian variance of the building ROI ----------------
    sharp = _sharpness(img_bgr, bb, w, h)

    # ---- Coverage: building bbox area / frame area -------------------------
    coverage = fd.building_coverage  # already computed in FrameDetections

    # ---- Frontality: how centred and upright the building appears ----------
    frontality = _frontality(bb, w, h)

    # ---- Exposure: mean pixel brightness of building ROI ------------------
    exposure = _exposure(img_bgr, bb, w, h)

    # ---- Occlusion: fraction of building bbox overlapped by occluders ------
    occ = _occlusion_fraction(bb, fd.occluders)

    total = (
        config.W_SHARPNESS  * sharp
        + config.W_COVERAGE   * coverage
        + config.W_FRONTALITY * frontality
        + config.W_EXPOSURE   * exposure
        - config.OCCLUSION_PENALTY_WEIGHT * occ
    )
    total = max(0.0, min(1.0, total))

    return ScoredFrame(
        frame_path=fd.frame_path,
        house_index=house_index,
        timestamp_s=timestamp_s,
        total_score=total,
        sharpness=sharp,
        coverage=coverage,
        frontality=frontality,
        exposure=exposure,
        occlusion_fraction=occ,
        building_bbox=bb,
    )


def top_candidates(
    scored: list[ScoredFrame],
    n: int = config.TOP_N_CANDIDATES,
) -> list[ScoredFrame]:
    return sorted(scored, key=lambda s: s.total_score, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Individual metric functions — all return values in [0, 1]
# ---------------------------------------------------------------------------

def _roi(img: np.ndarray, bb: BBox | None, w: int, h: int) -> np.ndarray:
    """Crop img to the building bbox, or return full image if no bbox."""
    if bb is None:
        return img
    x1 = max(0, int(bb.x1))
    y1 = max(0, int(bb.y1))
    x2 = min(w, int(bb.x2))
    y2 = min(h, int(bb.y2))
    roi = img[y1:y2, x1:x2]
    return roi if roi.size > 0 else img


def _sharpness(img: np.ndarray, bb: BBox | None, w: int, h: int) -> float:
    roi = _roi(img, bb, w, h)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Empirically, 500 is a well-focused house; cap at 1.0
    return float(min(lap_var / 500.0, 1.0))


def _frontality(bb: BBox | None, frame_w: int, frame_h: int) -> float:
    """
    Two components:
    1. Horizontal centrality: is the building bbox centred left-right?
    2. Aspect ratio plausibility: a straight-on house has width ≥ height.
    """
    if bb is None:
        return 0.0

    bbox_cx = (bb.x1 + bb.x2) / 2.0
    centrality = 1.0 - abs(bbox_cx / frame_w - 0.5) * 2.0  # 1 at centre, 0 at edge

    bbox_w = bb.x2 - bb.x1
    bbox_h = bb.y2 - bb.y1
    if bbox_h <= 0:
        aspect_score = 0.0
    else:
        ratio = bbox_w / bbox_h
        # Ideal ratio for a frontal house is 1.2–2.5.  Score peaks at ~1.8.
        aspect_score = float(np.clip(1.0 - abs(ratio - 1.8) / 1.8, 0.0, 1.0))

    return 0.6 * centrality + 0.4 * aspect_score


def _exposure(img: np.ndarray, bb: BBox | None, w: int, h: int) -> float:
    """
    Good exposure: mean brightness 80–180 on 0–255 scale, low clipping.
    Returns 1.0 for ideal, lower for dark/overexposed frames.
    """
    roi = _roi(img, bb, w, h)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = float(gray.mean())
    # Penalise deviation from ideal midtone brightness (128)
    brightness_score = 1.0 - abs(mean - 128.0) / 128.0

    # Clipping penalty: fraction of pixels that are blown (>250) or crushed (<5)
    total = gray.size
    clipped = float(np.sum((gray > 250) | (gray < 5)))
    clip_penalty = clipped / total if total > 0 else 0.0

    return float(max(0.0, brightness_score - clip_penalty))


def _occlusion_fraction(bb: BBox | None, occluders: list[BBox]) -> float:
    """Fraction of the building bbox area that is covered by occluder boxes."""
    if bb is None or not occluders:
        return 0.0
    building_area = bb.area
    if building_area <= 0:
        return 0.0
    # Simple union approximation: sum intersection areas (ignores overlap between
    # occluders themselves, but that's a minor edge case for street footage).
    total_occluded = sum(bb.intersection_area(occ) for occ in occluders)
    return float(min(total_occluded / building_area, 1.0))


def _zero_score(path: Path, house_index: int, ts: float, bb: BBox | None) -> ScoredFrame:
    return ScoredFrame(
        frame_path=path, house_index=house_index, timestamp_s=ts,
        total_score=0.0, sharpness=0.0, coverage=0.0, frontality=0.0,
        exposure=0.0, occlusion_fraction=1.0, building_bbox=bb,
    )
