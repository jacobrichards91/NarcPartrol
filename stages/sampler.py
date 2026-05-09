"""
Stage 3 — Frame Sampling
Extracts frames from the video at a fixed FPS using a single ffmpeg pass.
Each frame is tagged with its wall-clock timestamp and assigned to the house
segment it falls within.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from stages.segment import HouseSegment, find_segment_for_time


@dataclass
class SampledFrame:
    house_index: int
    timestamp_s: float
    image_path: Path


def sample_frames(
    video_path: Path,
    segments: list[HouseSegment],
    output_dir: Path,
    fps: int,
) -> list[SampledFrame]:
    """
    Extract frames at `fps` frames-per-second for the full video duration in
    one ffmpeg invocation, then assign each frame to its house segment.

    Returns frames sorted by timestamp.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(output_dir / "frame_%08d.jpg")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",        # JPEG quality 2 = near-lossless for scoring accuracy
        "-y",
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed for {video_path}:\n{result.stderr}"
        )

    frames: list[SampledFrame] = []
    for frame_path in sorted(output_dir.glob("frame_????????.jpg")):
        # ffmpeg names frames 1-indexed: frame_00000001.jpg → index 1
        n = int(frame_path.stem.split("_")[1])
        ts = (n - 1) / fps
        house_idx = find_segment_for_time(ts, segments)
        frames.append(SampledFrame(
            house_index=house_idx,
            timestamp_s=ts,
            image_path=frame_path,
        ))

    return frames


def frames_by_house(frames: list[SampledFrame]) -> dict[int, list[SampledFrame]]:
    """Group a flat frame list into {house_index: [frames]} dict."""
    groups: dict[int, list[SampledFrame]] = {}
    for f in frames:
        groups.setdefault(f.house_index, []).append(f)
    return groups
