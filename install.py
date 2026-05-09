#!/usr/bin/env python3
"""
NarcPartrol Installer — works on Windows and Linux/macOS

Run with:  python install.py
           python3 install.py

What it does:
  1. Checks Python version (3.11+ required)
  2. Detects GPU and CUDA version via nvidia-smi
  3. Installs ffmpeg and exiftool (via winget/choco on Windows, apt on Linux)
  4. Installs PyTorch with the correct CUDA wheel
  5. Installs all Python packages from requirements.txt
  6. Pre-downloads the YOLOv8x-oiv7 model weights
  7. Places narcpartrol.toml in ~/Documents/NarcPartrol/
  8. Runs a final verification
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import platform
import re
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"
IS_MAC     = platform.system() == "Darwin"

CONFIG_DEST = Path.home() / "Documents" / "NarcPartrol"
CONFIG_FILE = CONFIG_DEST / "narcpartrol.toml"

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _supports_color() -> bool:
    if IS_WINDOWS:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_COLOR = _supports_color()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text

def ok(msg: str)     -> None: print(_c("32", f"  ✓  {msg}"))
def info(msg: str)   -> None: print(_c("36", f"  →  {msg}"))
def warn(msg: str)   -> None: print(_c("33", f"  ⚠  {msg}"))
def fail(msg: str)   -> None: print(_c("31", f"  ✗  {msg}"), file=sys.stderr); sys.exit(1)
def header(msg: str) -> None: print(f"\n{_c('1;36', f'━━━  {msg}  ━━━')}")
def step(n: int, total: int, msg: str) -> None:
    print(f"\n{_c('1', f'[{n}/{total}]')} {msg}")


def run(cmd: list[str], check: bool = False, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=capture, text=True,
        check=check,
        # On Windows, don't show a new console window for subprocesses
        creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
    )


def pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip"] + list(args), check=True)


# ---------------------------------------------------------------------------
# Step 1 — Python version
# ---------------------------------------------------------------------------
def check_python() -> None:
    step(1, 8, "Python version")
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 11):
        fail(
            f"Python 3.11+ is required but you have {v.major}.{v.minor}.\n"
            "  Download it from https://www.python.org/downloads/"
        )
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
    # PyTorch releases typically lag 3-6 months behind new Python versions.
    # If we're on a very new Python, wheel availability may be limited.
    if v.minor > 12:
        warn(f"Python 3.{v.minor} is newer than the PyTorch release cycle has "
             "been tested against.")
        warn("The installer will try multiple PyTorch wheel indexes automatically.")
        warn("If all fail, install Python 3.12 alongside this one from python.org")
        warn("and rerun:  py -3.12 install.py")


# ---------------------------------------------------------------------------
# Step 2 — GPU / CUDA
# ---------------------------------------------------------------------------

# Ordered lists of wheel tags to try per CUDA major version (newest first so
# we get the build most likely to carry wheels for the current Python version).
_CUDA_TAG_CANDIDATES: dict[int, list[str]] = {
    13: ["cu128", "cu126", "cu124", "cu121"],
    12: ["cu126", "cu124", "cu121"],
    11: ["cu118"],
}


def detect_gpu() -> list[str]:
    """
    Return an ordered list of PyTorch wheel tags to try, e.g.
    ['cu128', 'cu126', 'cu124', 'cu121'] for CUDA 13.x.
    Falls back to ['cpu'] when no GPU is detected.
    """
    step(2, 8, "GPU & CUDA detection")

    smi_path = shutil.which("nvidia-smi")
    if IS_WINDOWS and smi_path is None:
        # nvidia-smi lives in System32 on Windows — may not be in PATH
        candidate = Path(r"C:\Windows\System32\nvidia-smi.exe")
        if candidate.exists():
            smi_path = str(candidate)

    if smi_path is None:
        warn("nvidia-smi not found — assuming no NVIDIA GPU.")
        warn("The pipeline will run on CPU.  Processing will be slow.")
        return ["cpu"]

    result = run([smi_path])
    if result.returncode != 0:
        warn("nvidia-smi returned an error.  Falling back to CPU mode.")
        return ["cpu"]

    # Parse GPU name
    gpu_match = re.search(r"(?:GeForce|Quadro|Tesla|RTX|GTX|A\d)\s[\w\s]+", result.stdout)
    gpu_name  = gpu_match.group(0).strip() if gpu_match else "Unknown GPU"

    # Parse CUDA version
    cuda_match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
    if not cuda_match:
        warn(f"GPU found ({gpu_name}) but could not read CUDA version.")
        warn("Falling back to CPU PyTorch.  Install the CUDA toolkit and retry.")
        return ["cpu"]

    cuda_ver   = cuda_match.group(1)
    cuda_major = int(cuda_ver.split(".")[0])

    ok(f"GPU:  {gpu_name}")
    ok(f"CUDA: {cuda_ver}")

    tags = _CUDA_TAG_CANDIDATES.get(cuda_major)
    if tags is None:
        if cuda_major < 11:
            warn(f"CUDA {cuda_ver} is too old (need ≥ 11).  Falling back to CPU PyTorch.")
            return ["cpu"]
        # Unknown future major — try all known tags newest-first
        tags = ["cu128", "cu126", "cu124", "cu121"]

    ok(f"Will try wheel tags (in order): {', '.join(tags)}")
    return tags


# ---------------------------------------------------------------------------
# Step 3 — System packages (ffmpeg + exiftool)
# ---------------------------------------------------------------------------
def install_system_deps() -> None:
    step(3, 8, "System packages (ffmpeg, exiftool)")

    if IS_WINDOWS:
        _install_windows_tools()
    elif IS_LINUX:
        _install_linux_tools()
    elif IS_MAC:
        _install_mac_tools()
    else:
        warn("Unrecognised OS — skipping system package installation.")
        warn("Install ffmpeg and exiftool manually and make sure they are in PATH.")


def _install_windows_tools() -> None:
    tools = {
        "ffmpeg":    ("winget", ["winget", "install", "--id", "Gyan.FFmpeg",      "-e", "--silent", "--accept-source-agreements", "--accept-package-agreements"]),
        "exiftool":  ("winget", ["winget", "install", "--id", "OliverBetz.ExifTool", "-e", "--silent", "--accept-source-agreements", "--accept-package-agreements"]),
    }
    choco_tools = {
        "ffmpeg":   ["choco", "install", "ffmpeg",   "-y", "--no-progress"],
        "exiftool": ["choco", "install", "exiftool", "-y", "--no-progress"],
    }

    has_winget = shutil.which("winget") is not None
    has_choco  = shutil.which("choco")  is not None

    for name, (_, winget_cmd) in tools.items():
        if shutil.which(name):
            ok(f"{name} already in PATH")
            continue

        if has_winget:
            info(f"Installing {name} via winget ...")
            result = run(winget_cmd)
            if result.returncode == 0:
                ok(f"{name} installed via winget")
                continue
            warn(f"winget install of {name} failed (exit {result.returncode})")

        if has_choco:
            info(f"Installing {name} via Chocolatey ...")
            result = run(choco_tools[name])
            if result.returncode == 0:
                ok(f"{name} installed via Chocolatey")
                continue
            warn(f"choco install of {name} failed (exit {result.returncode})")

        # Neither worked — print manual instructions
        _manual_install_windows(name)

    # winget adds to PATH for new shells only — warn the user
    if has_winget and (not shutil.which("ffmpeg") or not shutil.which("exiftool")):
        warn("Newly installed tools may not be in PATH until you open a new terminal.")
        warn("If the pipeline says ffmpeg/exiftool not found, close and reopen your")
        warn("terminal and run the pipeline again.")


def _manual_install_windows(name: str) -> None:
    urls = {
        "ffmpeg":   "https://www.gyan.dev/ffmpeg/builds/  (download ffmpeg-release-essentials.zip)",
        "exiftool": "https://exiftool.org/  (download exiftool-XX.XX_64.zip)",
    }
    warn(f"{name} could not be installed automatically.")
    print(textwrap.dedent(f"""
        Install it manually:
          1. Download from: {urls.get(name, 'the official website')}
          2. Extract the zip to a folder, e.g. C:\\Tools\\{name}\\
          3. Add that folder to your system PATH:
             Windows key → "Edit the system environment variables"
             → Environment Variables → System variables → Path → Edit → New
          4. Open a new terminal and run this installer again.
    """))


def _install_linux_tools() -> None:
    if not shutil.which("apt-get"):
        warn("apt-get not available — install ffmpeg and exiftool manually.")
        return
    run(["sudo", "apt-get", "update", "-qq"], check=True)
    for pkg in ["ffmpeg", "libimage-exiftool-perl"]:
        result = run(["dpkg", "-s", pkg])
        if result.returncode == 0:
            ok(f"{pkg} already installed")
        else:
            info(f"Installing {pkg} ...")
            run(["sudo", "apt-get", "install", "-y", pkg], check=True, capture=False)
            ok(f"{pkg} installed")


def _install_mac_tools() -> None:
    if not shutil.which("brew"):
        warn("Homebrew not found.  Install it from https://brew.sh then rerun.")
        return
    for pkg in ["ffmpeg", "exiftool"]:
        result = run(["brew", "list", pkg])
        if result.returncode == 0:
            ok(f"{pkg} already installed")
        else:
            info(f"Installing {pkg} via brew ...")
            run(["brew", "install", pkg], check=True, capture=False)
            ok(f"{pkg} installed")


# ---------------------------------------------------------------------------
# Step 4 — PyTorch (CUDA-matched, with multi-tag fallback)
# ---------------------------------------------------------------------------
def install_pytorch(cuda_tags: list[str]) -> None:
    step(4, 8, "PyTorch")

    want_gpu = cuda_tags != ["cpu"]

    # Check if already installed with the right CUDA support
    try:
        import torch  # type: ignore
        cuda_ok = torch.cuda.is_available()
        if want_gpu and not cuda_ok:
            info("CPU-only torch detected but GPU is available — reinstalling.")
            raise ImportError
        ok(f"PyTorch {torch.__version__} already installed  (cuda={cuda_ok})")
        return
    except ImportError:
        pass

    def _try_install(tag: str) -> bool:
        """Attempt to pip-install torch with the given wheel tag. Returns True on success."""
        if tag == "cpu":
            url = "https://download.pytorch.org/whl/cpu"
        else:
            url = f"https://download.pytorch.org/whl/{tag}"
        info(f"Trying PyTorch wheel index: {tag} ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "torch", "torchvision", "--index-url", url],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
        )
        if result.returncode == 0:
            ok(f"PyTorch installed from {tag} index")
            return True
        # Surface the key error line so the user can see what went wrong
        for line in result.stderr.splitlines():
            if "ERROR" in line or "No matching" in line:
                info(f"  pip said: {line.strip()}")
                break
        return False

    # Try each preferred tag in order
    for tag in cuda_tags:
        if _try_install(tag):
            break
    else:
        # Last resort: default PyPI index (some Python versions only have wheels there)
        info("Trying default PyPI index as final fallback ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision"],
            capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
        )
        if result.returncode != 0:
            fail(
                "Could not install PyTorch for this Python version.\n\n"
                f"  Your Python: {sys.version.split()[0]}\n"
                "  PyTorch wheel availability lags new Python releases by a few months.\n\n"
                "  Fix options:\n"
                "    A) Install Python 3.12 from python.org and rerun:\n"
                "         py -3.12 install.py\n"
                "    B) Wait for an official PyTorch release that supports your Python.\n"
                "       Check https://pytorch.org/get-started/locally/ for current support."
            )

    import importlib
    torch = importlib.import_module("torch")
    ok(f"PyTorch {torch.__version__} installed  (cuda={torch.cuda.is_available()})")


# ---------------------------------------------------------------------------
# Step 5 — Python packages
# ---------------------------------------------------------------------------
def install_packages() -> None:
    step(5, 8, "Python packages")
    reqs = SCRIPT_DIR / "requirements.txt"
    if not reqs.exists():
        fail(f"requirements.txt not found at {reqs}")
    info("Running pip install -r requirements.txt ...")
    pip("install", "--upgrade", "pip", "--quiet")
    pip("install", "-r", str(reqs))
    ok("All packages installed")


# ---------------------------------------------------------------------------
# Step 6 — Pre-download YOLO model
# ---------------------------------------------------------------------------
def predownload_model() -> None:
    step(6, 8, "YOLO model weights")

    # Read model name from the user config if it exists, else use default
    model_name = "yolov8x-oiv7.pt"
    if CONFIG_FILE.exists():
        try:
            if sys.version_info >= (3, 11):
                import tomllib
                with open(CONFIG_FILE, "rb") as f:
                    data = tomllib.load(f)
            else:
                import tomli as tomllib  # type: ignore
                with open(CONFIG_FILE, "rb") as f:
                    data = tomllib.load(f)
            model_name = data.get("detection", {}).get("model", model_name)
        except Exception:
            pass

    info(f"Pre-downloading {model_name} (~140 MB — once only) ...")
    code = textwrap.dedent(f"""
        from ultralytics import YOLO
        import sys
        try:
            YOLO("{model_name}")
            print("Model ready.")
        except Exception as e:
            print(f"Warning: {{e}}", file=sys.stderr)
            print("Model will download automatically on first run.")
    """)
    subprocess.run([sys.executable, "-c", code], check=False)
    ok(f"{model_name} ready")


# ---------------------------------------------------------------------------
# Step 7 — Config file
# ---------------------------------------------------------------------------
def place_config() -> None:
    step(7, 8, "User config")
    CONFIG_DEST.mkdir(parents=True, exist_ok=True)

    src = SCRIPT_DIR / "narcpartrol.toml"
    if not src.exists():
        warn(f"narcpartrol.toml not found in {SCRIPT_DIR} — skipping config placement.")
        return

    if CONFIG_FILE.exists():
        ok(f"Config already exists — leaving unchanged:\n       {CONFIG_FILE}")
        info("Delete it and rerun the installer to reset to defaults.")
    else:
        import shutil as _shutil
        _shutil.copy2(src, CONFIG_FILE)
        ok(f"Config placed at:\n       {CONFIG_FILE}")


# ---------------------------------------------------------------------------
# Step 8 — Verification
# ---------------------------------------------------------------------------
def verify() -> None:
    step(8, 8, "Final verification")

    sys.path.insert(0, str(SCRIPT_DIR))
    errors: list[str] = []

    # Config
    try:
        import config  # type: ignore
        path = config.config_file_path()
        ok(f"config        loaded from {path or 'built-in defaults'}")
    except Exception as e:
        errors.append(f"config: {e}")

    # Stage imports
    for mod in ["stages.ingest", "stages.segment", "stages.sampler",
                "stages.scorer", "stages.exporter"]:
        try:
            __import__(mod)
            ok(f"{mod.split('.')[1]:<14}")
        except Exception as e:
            errors.append(f"{mod}: {e}")

    # PyTorch + CUDA
    try:
        import torch  # type: ignore
        cuda = torch.cuda.is_available()
        dev  = torch.cuda.get_device_name(0) if cuda else "CPU"
        ok(f"torch          {torch.__version__}  device={dev}")
    except Exception as e:
        errors.append(f"torch: {e}")

    # OpenCV
    try:
        import cv2  # type: ignore
        ok(f"opencv         {cv2.__version__}")
    except Exception as e:
        errors.append(f"opencv: {e}")

    # EasyOCR
    try:
        import easyocr  # type: ignore  # noqa
        ok("easyocr")
    except Exception as e:
        errors.append(f"easyocr: {e}")

    # ffmpeg
    if shutil.which("ffmpeg"):
        result = run(["ffmpeg", "-version"])
        ver = re.search(r"version ([\d.]+)", result.stdout or "")
        ok(f"ffmpeg         {ver.group(1) if ver else 'found'}")
    else:
        warn("ffmpeg not in PATH — open a new terminal after install and retry if needed.")

    # exiftool
    if shutil.which("exiftool"):
        result = run(["exiftool", "-ver"])
        ok(f"exiftool       {result.stdout.strip()}")
    else:
        warn("exiftool not in PATH — GPS extraction will fall back to ffprobe.")

    if errors:
        print()
        for e in errors:
            warn(f"Issue: {e}")
        print()
        warn("Some checks failed.  Review the messages above and fix before running the pipeline.")
    else:
        _print_success()


# ---------------------------------------------------------------------------
# Done banner
# ---------------------------------------------------------------------------
def _print_success() -> None:
    sep = "═" * 54
    print(f"\n{_c('1;32', sep)}")
    print(_c("1;32", "  NarcPartrol is ready."))
    print(_c("1;32", sep))
    print(f"""
  Config:  {CONFIG_FILE}
  Edit it to tune lot size, scoring weights, etc.

  Run the pipeline:
    python pipeline.py drive.mov --output ./results
    python pipeline.py *.mov    --output ./results --skip-cloud

  CLI flags:
    --min-frontage METERS   GPS lot-size threshold
    --fps N                 Frame sampling rate
    --skip-cloud            Disable Claude Vision gate
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(_c("1", "\n  NarcPartrol — Installer\n  " + "─" * 36))

    check_python()
    cuda_tags = detect_gpu()
    install_system_deps()
    install_pytorch(cuda_tags)
    install_packages()
    predownload_model()
    place_config()
    verify()


if __name__ == "__main__":
    main()
