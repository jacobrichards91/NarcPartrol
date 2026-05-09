"""
NarcPartrol — Streamlit UI

Launch with:
    streamlit run app.py
or via the convenience scripts:
    Windows:   launch_ui.bat
    Linux/Mac: launch_ui.sh
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).parent.resolve()
PIPELINE_PY = SCRIPT_DIR / "pipeline.py"
CONFIG_DIR  = Path.home() / "Documents" / "NarcPartrol"
CONFIG_FILE = CONFIG_DIR / "narcpartrol.toml"
ENV_FILE    = CONFIG_DIR / ".env"
RUNS_DIR    = CONFIG_DIR / "runs"

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NarcPartrol",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# Settings persistence — TOML config + .env file
# ===========================================================================
def load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def save_env_file(api_key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(f'ANTHROPIC_API_KEY="{api_key}"\n', encoding="utf-8")
    os.environ["ANTHROPIC_API_KEY"] = api_key


def load_toml_config() -> dict:
    """Load the user TOML config; return an empty dict if absent."""
    if not CONFIG_FILE.exists():
        return {}
    import tomllib
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def save_toml_config(updates: dict) -> None:
    """Merge updates into the user TOML config, preserving comments/formatting."""
    import tomlkit

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if CONFIG_FILE.exists():
        doc = tomlkit.parse(CONFIG_FILE.read_text(encoding="utf-8"))
    else:
        # Seed from the bundled template if user has no config yet
        template = SCRIPT_DIR / "narcpartrol.toml"
        if template.exists():
            doc = tomlkit.parse(template.read_text(encoding="utf-8"))
        else:
            doc = tomlkit.document()

    for section, values in updates.items():
        if section not in doc:
            doc[section] = tomlkit.table()
        for key, val in values.items():
            doc[section][key] = val

    CONFIG_FILE.write_text(tomlkit.dumps(doc), encoding="utf-8")


def get_setting(cfg: dict, section: str, key: str, default):
    return cfg.get(section, {}).get(key, default)


# ===========================================================================
# Subprocess streaming for the pipeline run
# ===========================================================================
def run_pipeline_streaming(
    args: list[str],
    log_queue: Queue,
    state: dict,
) -> None:
    """
    Launch pipeline.py as a subprocess and stream every stdout line into
    log_queue.  Sets state['rc'] when the process finishes.
    """
    cmd = [sys.executable, "-u", str(PIPELINE_PY)] + args
    log_queue.put(f"$ {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            cwd=str(SCRIPT_DIR),
            env={**os.environ},
        )
        state["pid"] = proc.pid
        for line in proc.stdout:
            log_queue.put(line.rstrip())
            if state.get("cancel"):
                proc.terminate()
                log_queue.put("[cancelled by user]")
                break
        proc.wait()
        state["rc"] = proc.returncode
    except Exception as e:
        log_queue.put(f"[error] {e!r}")
        state["rc"] = -1
    finally:
        log_queue.put(None)  # sentinel — drains the consumer loop


# ===========================================================================
# Initial setup
# ===========================================================================
load_env_file()
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

cfg = load_toml_config()

# Session state defaults
ss = st.session_state
ss.setdefault("running", False)
ss.setdefault("log_lines", [])
ss.setdefault("log_queue", None)
ss.setdefault("run_state", None)
ss.setdefault("output_dir_last", str(RUNS_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S")))


# ===========================================================================
# Sidebar — Settings
# ===========================================================================
with st.sidebar:
    st.markdown("### 🏠 NarcPartrol")
    st.caption("HOA Compliance Snapshot Pipeline")
    st.divider()

    st.markdown("#### Anthropic API")
    api_key = st.text_input(
        "API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Used by the cloud quality gate. Stored in ~/Documents/NarcPartrol/.env",
        placeholder="sk-ant-...",
    )

    st.divider()
    st.markdown("#### Detection")
    min_lot = st.number_input(
        "Min lot frontage (metres)",
        min_value=5.0, max_value=50.0, step=1.0,
        value=float(get_setting(cfg, "gps", "min_lot_frontage_m", 15.0)),
        help="Tighten for dense urban rows; loosen for wide suburban lots.",
    )
    sample_fps = st.number_input(
        "Sample FPS",
        min_value=1, max_value=15, step=1,
        value=int(get_setting(cfg, "sampling", "fps", 4)),
        help="How many frames per second of video to extract.",
    )
    coverage_min = st.slider(
        "Min building coverage",
        min_value=0.05, max_value=0.50, step=0.01,
        value=float(get_setting(cfg, "detection", "building_coverage_min", 0.15)),
        help="Building must occupy at least this fraction of the frame.",
    )

    st.divider()
    st.markdown("#### Cloud Quality Gate")
    use_cloud = st.checkbox(
        "Use Claude Vision for low-confidence frames",
        value=True,
        help="Disable to run fully offline (no API calls).",
    )
    quality_threshold = st.slider(
        "Quality threshold",
        min_value=0.0, max_value=1.0, step=0.05,
        value=float(get_setting(cfg, "cloud", "quality_threshold", 0.45)),
        disabled=not use_cloud,
        help="Frames below this score get sent to Claude.",
    )

    st.divider()
    st.markdown("#### Output")
    jpeg_quality = st.slider(
        "JPEG quality",
        min_value=70, max_value=100, step=1,
        value=int(get_setting(cfg, "export", "jpeg_quality", 95)),
    )

    st.divider()
    if st.button("💾 Save settings as defaults", use_container_width=True):
        save_toml_config({
            "gps":       {"min_lot_frontage_m": float(min_lot)},
            "sampling":  {"fps": int(sample_fps)},
            "detection": {"building_coverage_min": float(coverage_min)},
            "cloud":     {"quality_threshold": float(quality_threshold)},
            "export":    {"jpeg_quality": int(jpeg_quality)},
        })
        if api_key:
            save_env_file(api_key)
        st.success("Saved.")
        time.sleep(0.6)
        st.rerun()

    st.caption(f"Config: `{CONFIG_FILE}`")


# ===========================================================================
# Main — tabs
# ===========================================================================
st.markdown("# 🏠 NarcPartrol")
st.caption("Process iPhone street footage into one cropped snapshot per house.")

tab_run, tab_results, tab_about = st.tabs(["▶ Run", "📷 Results", "ℹ About"])


# ---------------------------------------------------------------------------
# Run tab
# ---------------------------------------------------------------------------
with tab_run:
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("#### Input video")
        video_path_str = st.text_input(
            "Path to video file (MP4 / MOV)",
            placeholder=r"C:\Users\you\Videos\drive.mov",
            key="video_path",
        )

        st.markdown("#### Output folder")
        output_dir_str = st.text_input(
            "Where snapshots and log.csv will be saved",
            value=ss.output_dir_last,
            key="output_dir",
        )

    with col_r:
        st.markdown("#### Run summary")
        st.write(f"**Min frontage:** {min_lot:.0f} m")
        st.write(f"**Sample FPS:** {sample_fps}")
        st.write(f"**Cloud gate:** {'enabled' if use_cloud else 'disabled'}")
        if use_cloud and not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("Cloud gate enabled but no API key set.")

    st.divider()

    can_run = bool(video_path_str) and bool(output_dir_str) and not ss.running
    btn_l, btn_r = st.columns([1, 5])
    with btn_l:
        start_clicked = st.button(
            "▶ Start Processing",
            type="primary",
            disabled=not can_run,
            use_container_width=True,
        )
    with btn_r:
        if ss.running and st.button("⏹ Cancel", use_container_width=False):
            if ss.run_state:
                ss.run_state["cancel"] = True

    if start_clicked:
        video_path = Path(video_path_str)
        if not video_path.exists():
            st.error(f"Video file not found: {video_path}")
        else:
            # Persist current settings to the TOML so pipeline.py picks them up
            save_toml_config({
                "gps":       {"min_lot_frontage_m": float(min_lot)},
                "sampling":  {"fps": int(sample_fps)},
                "detection": {"building_coverage_min": float(coverage_min)},
                "cloud":     {"quality_threshold": float(quality_threshold)},
                "export":    {"jpeg_quality": int(jpeg_quality)},
            })
            if api_key:
                save_env_file(api_key)

            args = [str(video_path), "--output", output_dir_str]
            if not use_cloud:
                args.append("--skip-cloud")
            args += ["--min-frontage", str(min_lot), "--fps", str(sample_fps)]

            ss.log_queue  = Queue()
            ss.run_state  = {"rc": None, "cancel": False, "pid": None}
            ss.log_lines  = []
            ss.running    = True
            ss.output_dir_last = output_dir_str
            ss.start_time = time.time()

            t = threading.Thread(
                target=run_pipeline_streaming,
                args=(args, ss.log_queue, ss.run_state),
                daemon=True,
            )
            t.start()
            st.rerun()

    # ----- live log + status -----
    if ss.running or ss.log_lines:
        st.divider()
        status_box = st.empty()
        log_box    = st.empty()

        # Drain any pending lines from the queue into our display buffer
        if ss.running and ss.log_queue is not None:
            done = False
            while True:
                try:
                    line = ss.log_queue.get_nowait()
                except Empty:
                    break
                if line is None:
                    done = True
                    break
                ss.log_lines.append(line)

            elapsed = int(time.time() - ss.start_time) if "start_time" in ss else 0
            status_box.info(
                f"⏱ Running …  elapsed {elapsed//60}m {elapsed%60}s   "
                f"({len(ss.log_lines)} log lines)"
            )

            if done or ss.run_state.get("rc") is not None:
                ss.running = False
                rc = ss.run_state.get("rc", -1)
                if rc == 0:
                    status_box.success(f"✅ Finished in {elapsed//60}m {elapsed%60}s.")
                elif rc == -1 or ss.run_state.get("cancel"):
                    status_box.warning("Cancelled.")
                else:
                    status_box.error(f"❌ Pipeline exited with code {rc}.")
            else:
                # Schedule a rerun so we keep draining the queue
                time.sleep(0.5)
                st.rerun()
        else:
            status_box.empty()

        # Show last 200 lines so the page doesn't get unwieldy
        tail = "\n".join(ss.log_lines[-200:]) or "(no output yet)"
        log_box.code(tail, language="text")


# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------
with tab_results:
    st.markdown("#### Browse a completed run")
    results_dir_str = st.text_input(
        "Output folder to view",
        value=ss.output_dir_last,
        key="results_dir",
    )

    results_dir = Path(results_dir_str) if results_dir_str else None

    if not results_dir or not results_dir.exists():
        st.info("Pick a folder that contains snapshots/ and log.csv")
    else:
        snap_dir = results_dir / "snapshots"
        log_csv  = results_dir / "log.csv"

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Snapshots", len(list(snap_dir.glob("*.jpg"))) if snap_dir.exists() else 0)
        col_b.metric("Log file", "yes" if log_csv.exists() else "no")
        col_c.metric("Folder", str(results_dir.name))

        # ----- Log table -----
        if log_csv.exists():
            st.markdown("#### Log")
            try:
                import pandas as pd
                df = pd.read_csv(log_csv)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇ Download log.csv",
                    data=log_csv.read_bytes(),
                    file_name="log.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Could not read log.csv: {e}")

        # ----- Snapshot gallery -----
        if snap_dir.exists():
            st.markdown("#### Snapshots")
            jpgs = sorted(snap_dir.glob("*.jpg"))
            if not jpgs:
                st.info("No snapshots in this folder yet.")
            else:
                cols_per_row = 4
                for i in range(0, len(jpgs), cols_per_row):
                    row = jpgs[i:i + cols_per_row]
                    columns = st.columns(cols_per_row)
                    for col, jpg in zip(columns, row):
                        with col:
                            st.image(str(jpg), caption=jpg.stem, use_container_width=True)


# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(f"""
### What this does

NarcPartrol takes hours of iPhone street footage and produces one cropped
JPEG per house, plus a CSV log with timestamps, GPS coordinates, and
identified addresses.

### Pipeline stages

1. **Metadata extraction** — pulls the GPS timed track from the MOV file
2. **GPS segmentation** — splits the timeline into per-house windows
3. **Frame sampling** — extracts frames at the configured FPS via ffmpeg
4. **Building detection** — YOLOv8x (Open Images V7) on your GPU
5. **Quality scoring** — sharpness, coverage, frontality, exposure, occlusion
6. **Address resolution** — full-frame OCR + OpenStreetMap reverse geocode
7. **Cloud quality gate** — Claude Haiku vision on borderline frames only
8. **Crop & export** — cropped JPEG + CSV log row

### Configuration files

- TOML config: `{CONFIG_FILE}`
- API key:     `{ENV_FILE}`
- Run outputs: `{RUNS_DIR}` (default location for new runs)

### Tips

- Run on a short clip first to verify house-segmentation looks right.
- If you get too many houses, raise the **Min lot frontage**.
- If houses get merged together, lower it.
- Disable the cloud gate to run fully offline (no API calls).
""")
