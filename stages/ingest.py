"""
Stage 1 — Metadata Extraction
Pulls video properties (fps, duration, resolution) and the iPhone GPS timed
track from a MOV/MP4 file.  exiftool is the primary extractor; ffprobe is the
fallback for GPS when exiftool isn't installed.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoMetadata:
    path: Path
    fps: float
    duration_s: float
    width: int
    height: int
    # List of (timestamp_seconds, latitude, longitude) sorted by time.
    # Empty when GPS data cannot be extracted.
    gps_track: list[tuple[float, float, float]] = field(default_factory=list)


def extract_metadata(video_path: Path) -> VideoMetadata:
    path = Path(video_path)
    probe = _ffprobe(path)

    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"), None
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    num, den = map(int, video_stream["r_frame_rate"].split("/"))
    fps = num / den

    # Duration: prefer stream-level, fall back to container-level.
    duration_s = float(
        video_stream.get("duration")
        or probe.get("format", {}).get("duration", 0)
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    gps_track = _extract_gps_track(path)

    return VideoMetadata(
        path=path,
        fps=fps,
        duration_s=duration_s,
        width=width,
        height=height,
        gps_track=gps_track,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _extract_gps_track(path: Path) -> list[tuple[float, float, float]]:
    try:
        track = _gps_via_exiftool(path)
        if track:
            return track
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return []


def _gps_via_exiftool(path: Path) -> list[tuple[float, float, float]]:
    """
    Use exiftool's -ee (extract embedded) mode to read the per-frame GPS track
    that iPhones embed as a QuickTime timed metadata track.

    Output format per line: "timestamp,lat,lon"
    The # suffix forces numeric (decimal-degree) output for coordinates.
    """
    cmd = [
        "exiftool", "-ee",
        "-p", "${SampleTime},$GPSLatitude#,$GPSLongitude#",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    track: list[tuple[float, float, float]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        t = _parse_sample_time(parts[0].strip())
        try:
            lat = float(parts[1].strip())
            lon = float(parts[2].strip())
        except ValueError:
            continue
        if t is None or (lat == 0.0 and lon == 0.0):
            continue
        track.append((t, lat, lon))

    return sorted(track)


def _parse_sample_time(s: str) -> float | None:
    """Convert exiftool SampleTime strings like '0:00:01.234' to seconds."""
    s = s.strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, IndexError):
        return None
