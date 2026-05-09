"""
NarcPartrol — HOA Compliance Snapshot Pipeline

Usage:
    python pipeline.py VIDEO [VIDEO ...] --output OUTPUT_DIR [options]

For each input video the pipeline:
    S1  Extract GPS track + video properties
    S2  Segment timeline into per-house windows (GPS-based)
    S3  Sample frames at 4 fps via ffmpeg
    S4  Detect buildings in sampled frames (YOLOv8x-oiv7, GTX 1080Ti)
    S5  Score each frame (sharpness, coverage, frontality, exposure, occlusion)
    S6  Identify addresses (full-frame OCR + OSM reverse geocode)
    S7  Cloud quality gate for low-confidence frames (Claude Haiku)
    S8  Crop to building bbox + padding, save JPEG, write log.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

import config
from stages.ingest import extract_metadata
from stages.segment import segment_by_gps
from stages.sampler import sample_frames, frames_by_house
from stages.detector import detect_batch, filter_by_coverage
from stages.scorer import score_frame, top_candidates
from stages.ocr import resolve_address
from stages.cloud import pick_best_frame
from stages.exporter import (
    ExportedHouse, crop_and_save, open_log, write_log_row,
)


def run_video(
    video_path: Path,
    output_dir: Path,
    log_writer,
    skip_cloud: bool = False,
    min_frontage: float = config.MIN_LOT_FRONTAGE_M,
    sample_fps: int = config.SAMPLE_FPS,
    house_id_offset: int = 0,
) -> int:
    """
    Process a single video file end-to-end.
    Returns the number of houses successfully exported.
    """
    print(f"\n{'='*60}")
    print(f"  Video: {video_path.name}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # S1 — Metadata
    # ------------------------------------------------------------------
    print("S1  Extracting metadata and GPS track...")
    meta = extract_metadata(video_path)
    print(f"    {meta.duration_s:.1f}s  {meta.fps:.2f}fps  "
          f"{meta.width}x{meta.height}  "
          f"GPS points: {len(meta.gps_track)}")

    # ------------------------------------------------------------------
    # S2 — Segmentation
    # ------------------------------------------------------------------
    print("S2  Segmenting into house windows...")
    segments = segment_by_gps(meta.gps_track, min_frontage, meta.duration_s)
    print(f"    {len(segments)} house segment(s) detected")
    if not segments:
        print("    Nothing to process.")
        return 0

    # ------------------------------------------------------------------
    # S3 — Frame sampling (one ffmpeg pass)
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="narcpartrol_frames_") as tmpdir:
        frames_dir = Path(tmpdir)
        print(f"S3  Sampling at {sample_fps} fps into {frames_dir} ...")
        frames = sample_frames(video_path, segments, frames_dir, fps=sample_fps)
        print(f"    {len(frames)} frames extracted")

        by_house = frames_by_house(frames)

        # ------------------------------------------------------------------
        # S4+S5 — Detection and scoring per house segment
        # ------------------------------------------------------------------
        print("S4  Running YOLOv8 building detection (batched)...")
        all_paths = [f.image_path for f in frames]
        detections_map: dict[Path, object] = {}

        # Detect in one big batch across all houses
        all_detections = detect_batch(all_paths, batch_size=16)
        for fd in all_detections:
            detections_map[fd.frame_path] = fd

        print("S5  Scoring frames...")
        # Map (frame_path → SampledFrame) for timestamp lookup
        ts_map = {f.image_path: f.timestamp_s for f in frames}

        exported_count = 0
        snapshots_dir = output_dir / config.OUTPUT_SUBDIR
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        for seg in tqdm(segments, desc="Houses", unit="house"):
            house_frames = by_house.get(seg.index, [])
            if not house_frames:
                continue

            # Score frames that have a building detection
            scored = []
            for sf in house_frames:
                fd = detections_map.get(sf.image_path)
                if fd is None or fd.building_coverage < config.BUILDING_COVERAGE_MIN:
                    continue
                s = score_frame(fd, sf.timestamp_s, seg.index)
                scored.append(s)

            if not scored:
                continue

            candidates = top_candidates(scored, n=config.TOP_N_CANDIDATES)
            best = candidates[0]
            cloud_used = False

            # ------------------------------------------------------------------
            # S7 — Cloud gate
            # ------------------------------------------------------------------
            if not skip_cloud and best.total_score < config.QUALITY_THRESHOLD:
                chosen_idx = pick_best_frame(candidates)
                best = candidates[chosen_idx]
                cloud_used = True

            # ------------------------------------------------------------------
            # S6 — Address resolution (run on best frame only)
            # ------------------------------------------------------------------
            addr = resolve_address(best.frame_path, seg.lat, seg.lon)

            # ------------------------------------------------------------------
            # S8 — Export
            # ------------------------------------------------------------------
            global_idx = house_id_offset + seg.index
            house_id = f"house_{global_idx:05d}"
            out_path = crop_and_save(best, snapshots_dir, house_id)

            exported = ExportedHouse(
                house_index=global_idx,
                output_path=out_path,
                address=addr,
                scored_frame=best,
                cloud_reviewed=cloud_used,
            )
            write_log_row(log_writer, exported, seg.lat, seg.lon, video_path.name)
            exported_count += 1

        print(f"    {exported_count} houses exported from this video")
        return exported_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one high-quality snapshot per house from iPhone street footage."
    )
    parser.add_argument(
        "videos",
        nargs="+",
        type=Path,
        metavar="VIDEO",
        help="Input MP4/MOV file(s)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        metavar="DIR",
        help="Output directory (will be created if absent)",
    )
    parser.add_argument(
        "--min-frontage",
        type=float,
        default=config.MIN_LOT_FRONTAGE_M,
        metavar="METERS",
        help=f"Minimum metres between houses (default: {config.MIN_LOT_FRONTAGE_M})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=config.SAMPLE_FPS,
        help=f"Frames per second to sample (default: {config.SAMPLE_FPS})",
    )
    parser.add_argument(
        "--skip-cloud",
        action="store_true",
        help="Disable the Claude Vision quality gate (useful for offline runs)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.LOG_FILENAME

    log_fh, log_writer = open_log(log_path)

    total_houses = 0
    house_id_offset = 0

    try:
        for video_path in args.videos:
            if not video_path.exists():
                print(f"ERROR: {video_path} not found — skipping", file=sys.stderr)
                continue
            count = run_video(
                video_path=video_path,
                output_dir=output_dir,
                log_writer=log_writer,
                skip_cloud=args.skip_cloud,
                min_frontage=args.min_frontage,
                sample_fps=args.fps,
                house_id_offset=house_id_offset,
            )
            house_id_offset += count
            total_houses += count
            log_fh.flush()
    finally:
        log_fh.close()

    print(f"\nDone.  {total_houses} house snapshot(s) saved to {output_dir}")
    print(f"Log:   {log_path}")


if __name__ == "__main__":
    main()
