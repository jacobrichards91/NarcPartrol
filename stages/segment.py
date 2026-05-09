"""
Stage 2 — GPS-Based House Segmentation
Walks the GPS timed track, accumulates haversine distance, and starts a new
house segment every time the camera has traveled at least MIN_LOT_FRONTAGE_M
meters.  Falls back to a single whole-video segment when no GPS data exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HouseSegment:
    index: int
    start_time: float   # seconds into the video
    end_time: float
    lat: float          # GPS centroid of the segment (approximate house location)
    lon: float


def segment_by_gps(
    gps_track: list[tuple[float, float, float]],
    min_frontage_m: float,
    video_duration_s: float,
) -> list[HouseSegment]:
    """
    Split the video timeline into per-house windows.

    Args:
        gps_track:       [(timestamp_s, lat, lon), ...] sorted ascending.
        min_frontage_m:  Minimum meters traveled before starting a new segment.
        video_duration_s: Total video length; used to close the final segment.

    Returns:
        List of HouseSegment ordered by start_time.  Always returns at least
        one segment even when the GPS track is empty.
    """
    if not gps_track:
        return [HouseSegment(0, 0.0, video_duration_s, 0.0, 0.0)]

    segments: list[HouseSegment] = []
    seg_start_time = gps_track[0][0]
    seg_lat, seg_lon = gps_track[0][1], gps_track[0][2]
    accumulated_m = 0.0

    for i in range(1, len(gps_track)):
        t, lat, lon = gps_track[i]
        _, prev_lat, prev_lon = gps_track[i - 1]

        accumulated_m += _haversine_m(prev_lat, prev_lon, lat, lon)

        if accumulated_m >= min_frontage_m:
            segments.append(HouseSegment(
                index=len(segments),
                start_time=seg_start_time,
                end_time=t,
                lat=seg_lat,
                lon=seg_lon,
            ))
            seg_start_time = t
            seg_lat, seg_lon = lat, lon
            accumulated_m = 0.0

    # Close the final (possibly partial) segment.
    if seg_start_time < video_duration_s:
        segments.append(HouseSegment(
            index=len(segments),
            start_time=seg_start_time,
            end_time=video_duration_s,
            lat=seg_lat,
            lon=seg_lon,
        ))

    return segments


def find_segment_for_time(t: float, segments: list[HouseSegment]) -> int:
    """Return the index of the segment that contains timestamp t."""
    for seg in reversed(segments):
        if t >= seg.start_time:
            return seg.index
    return 0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))
