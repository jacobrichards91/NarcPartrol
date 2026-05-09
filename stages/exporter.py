"""
Stage 8 — Crop and Export
Crops the selected frame to the building bounding box (plus padding), saves
as JPEG, and appends a row to the CSV log.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
from PIL import Image, ImageEnhance

import config
from stages.ocr import AddressResult
from stages.scorer import ScoredFrame


@dataclass
class ExportedHouse:
    house_index: int
    output_path: Path
    address: AddressResult
    scored_frame: ScoredFrame
    cloud_reviewed: bool


_CSV_FIELDS = [
    "house_id",
    "address_final",
    "address_ocr",
    "address_osm",
    "address_confidence",
    "timestamp_s",
    "lat",
    "lon",
    "source_file",
    "quality_score",
    "sharpness",
    "coverage",
    "frontality",
    "exposure",
    "occlusion_fraction",
    "cloud_reviewed",
    "output_filename",
]


def crop_and_save(
    sf: ScoredFrame,
    output_dir: Path,
    house_id: str,
) -> Path:
    """
    Crop the frame to the building bbox (+ padding) and write a JPEG.
    Falls back to the full frame if no bbox is available.
    Returns the output path.
    """
    img = Image.open(sf.frame_path)
    w, h = img.size

    if sf.building_bbox is not None:
        bb = sf.building_bbox
        pad_x = (bb.x2 - bb.x1) * config.CROP_PADDING
        pad_y = (bb.y2 - bb.y1) * config.CROP_PADDING
        x1 = max(0, int(bb.x1 - pad_x))
        y1 = max(0, int(bb.y1 - pad_y))
        x2 = min(w, int(bb.x2 + pad_x))
        y2 = min(h, int(bb.y2 + pad_y))
        cropped = img.crop((x1, y1, x2, y2))
    else:
        cropped = img

    out_path = output_dir / f"{house_id}.jpg"
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped.save(str(out_path), "JPEG", quality=config.JPEG_QUALITY)
    return out_path


def open_log(log_path: Path) -> tuple[io.TextIOWrapper, csv.DictWriter]:
    """Open the CSV log for appending, writing header if new."""
    is_new = not log_path.exists()
    fh = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
    if is_new:
        writer.writeheader()
    return fh, writer


def write_log_row(
    writer: csv.DictWriter,
    exported: ExportedHouse,
    seg_lat: float,
    seg_lon: float,
    source_file: str,
) -> None:
    sf = exported.scored_frame
    addr = exported.address
    writer.writerow({
        "house_id":           f"house_{exported.house_index:05d}",
        "address_final":      addr.address_final,
        "address_ocr":        addr.address_ocr,
        "address_osm":        addr.address_osm,
        "address_confidence": addr.confidence,
        "timestamp_s":        f"{sf.timestamp_s:.3f}",
        "lat":                f"{seg_lat:.7f}",
        "lon":                f"{seg_lon:.7f}",
        "source_file":        source_file,
        "quality_score":      f"{sf.total_score:.4f}",
        "sharpness":          f"{sf.sharpness:.4f}",
        "coverage":           f"{sf.coverage:.4f}",
        "frontality":         f"{sf.frontality:.4f}",
        "exposure":           f"{sf.exposure:.4f}",
        "occlusion_fraction": f"{sf.occlusion_fraction:.4f}",
        "cloud_reviewed":     str(exported.cloud_reviewed),
        "output_filename":    exported.output_path.name,
    })
