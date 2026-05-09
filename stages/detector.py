"""
Stage 4 — Building Detection
Runs YOLOv8x (Open Images V7 checkpoint) on sampled frames in batches.
Returns per-frame building and occluder bounding boxes.

The model is loaded once and reused across all calls within a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    label: str

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def intersection_area(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


@dataclass
class FrameDetections:
    frame_path: Path
    frame_w: int
    frame_h: int
    buildings: list[BBox] = field(default_factory=list)
    occluders: list[BBox] = field(default_factory=list)

    @property
    def best_building(self) -> BBox | None:
        """Largest confident building bbox, or None."""
        if not self.buildings:
            return None
        return max(self.buildings, key=lambda b: b.area)

    @property
    def building_coverage(self) -> float:
        bb = self.best_building
        if bb is None:
            return 0.0
        frame_area = self.frame_w * self.frame_h
        return bb.area / frame_area if frame_area > 0 else 0.0


_model = None  # module-level singleton so we only load weights once


def load_model() -> None:
    global _model
    if _model is not None:
        return
    from ultralytics import YOLO
    _model = YOLO(config.YOLO_MODEL)


def detect_batch(
    frame_paths: list[Path],
    batch_size: int = config.YOLO_BATCH_SIZE,
) -> list[FrameDetections]:
    """
    Run inference on a list of frame images.  Returns one FrameDetections per
    input path, in the same order.
    """
    load_model()

    results_out: list[FrameDetections] = []

    # Resolve class IDs for building and occluder labels from the loaded model.
    model_names: dict[int, str] = _model.names   # {id: label_name}
    building_ids = {
        cid for cid, name in model_names.items()
        if name in config.BUILDING_CLASS_NAMES
    }
    occluder_ids = {
        cid for cid, name in model_names.items()
        if name in config.OCCLUDER_CLASS_NAMES
    }

    for start in range(0, len(frame_paths), batch_size):
        batch = frame_paths[start: start + batch_size]
        # ultralytics accepts a list of paths directly
        results = _model(
            [str(p) for p in batch],
            conf=config.BUILDING_CONF_MIN,
            verbose=False,
        )

        for res, fpath in zip(results, batch):
            h, w = res.orig_shape[:2]
            fd = FrameDetections(frame_path=fpath, frame_w=w, frame_h=h)

            if res.boxes is None:
                results_out.append(fd)
                continue

            boxes = res.boxes
            for i in range(len(boxes)):
                cid = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy()
                bbox = BBox(
                    x1=float(xyxy[0]), y1=float(xyxy[1]),
                    x2=float(xyxy[2]), y2=float(xyxy[3]),
                    conf=conf,
                    label=model_names.get(cid, str(cid)),
                )
                if cid in building_ids:
                    fd.buildings.append(bbox)
                elif cid in occluder_ids:
                    fd.occluders.append(bbox)

            results_out.append(fd)

    return results_out


def filter_by_coverage(
    detections: list[FrameDetections],
) -> list[FrameDetections]:
    """Drop frames where no building meets the minimum coverage threshold."""
    return [
        fd for fd in detections
        if fd.building_coverage >= config.BUILDING_COVERAGE_MIN
    ]
