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
    """Run a short command. Captures output by default for parsing version strings etc."""
    return subprocess.run(
        cmd, capture_output=capture, text=True,
        check=check,
    )


def run_streaming(cmd: list[str]) -> int:
    """Run a long command and let its stdout/stderr stream to the terminal."""
    print(_c("90", "  " + "─" * 60))
    rc = subprocess.run(cmd).returncode
    print(_c("90", "  " + "─" * 60))
    return rc


def pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip"] + list(args), check=True)


# ---------------------------------------------------------------------------
# Step 1 — Python version
# ---------------------------------------------------------------------------
ALLOW_UNTESTED = "--allow-untested-python" in sys.argv


def check_python() -> None:
    step(1, 8, "Python version")
    v = sys.version_info

    if v.major < 3 or (v.major == 3 and v.minor < 11):
        fail(
            f"Python 3.11+ is required but you have {v.major}.{v.minor}.\n"
            "  Download Python 3.12 from https://www.python.org/downloads/"
        )

    # Several pipeline dependencies (easyocr → python-bidi, etc.) don't yet
    # ship Windows wheels for Python 3.14, and source builds need Rust + MSVC
    # build tools.  Better to fail fast than half-way through pip.
    if v.minor >= 14 and not ALLOW_UNTESTED:
        msg = textwrap.dedent(f"""
            Python 3.{v.minor}.{v.micro} is too new for this pipeline.

            Some dependencies (easyocr → python-bidi) have no Windows wheel for
            Python 3.{v.minor} yet, and would need Visual Studio Build Tools + Rust
            to compile from source.

            ──  Recommended fix  ───────────────────────────────────────────
              1. Install Python 3.12 from python.org (keep 3.{v.minor} too):
                   https://www.python.org/downloads/release/python-31210/
                 During install, tick:
                   ☑ Add python.exe to PATH
                   ☑ Install py launcher

              2. Re-run this installer using the py launcher:
                   py -3.12 install.py

              3. Use launch_ui.bat as before — it will auto-detect the
                 Python where dependencies are installed.

              Or, to use Python 3.12 manually for everything:
                   py -3.12 pipeline.py footage.mov --output ./results

            ──  If you really want to try with this Python anyway  ─────────
              Install Visual Studio Build Tools with the "Desktop development
              with C++" workload, install Rust from https://rustup.rs/, then:
                   {Path(sys.executable).name} install.py --allow-untested-python
              (Expect long compile times and possibly more failures.)
        """).strip()
        fail(msg)

    ok(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.minor > 12:
        warn(f"Python 3.{v.minor} is past the well-tested range — proceeding anyway")
        warn("because --allow-untested-python was passed.  If pip build errors")
        warn("occur, install Python 3.12 and rerun: py -3.12 install.py")


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

# For older GPUs (compute capability < 7.5 — Pascal and before) we must use an
# older PyTorch release.  sm_61 support was silently dropped in PyTorch 2.5
# (both cu121 and cu118 builds).  PyTorch 2.4.x+cu118 is the newest release
# that still ships sm_60/sm_61 kernels.  Try newest-compatible first.
_LEGACY_GPU_CANDIDATES: list[tuple[str, str, str]] = [
    # (torch_version, torchvision_version, wheel_tag)
    ("2.4.0", "0.19.0", "cu118"),
    ("2.3.1", "0.18.1", "cu118"),
    ("2.2.2", "0.17.2", "cu118"),
]


def _detect_compute_cap(smi_path: str) -> float | None:
    """Return the GPU's compute capability as a float (e.g. 6.1, 7.5, 8.6)."""
    # Newer drivers expose this directly:
    result = run([smi_path, "--query-gpu=compute_cap", "--format=csv,noheader"])
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                return float(line.strip())
            except ValueError:
                continue
    return None


def detect_gpu() -> tuple[list[str], float | None]:
    """
    Return (wheel_tags, compute_capability).
    wheel_tags is the ordered list of pip --index-url tags to try.
    compute_capability is e.g. 6.1 for a 1080Ti, 8.6 for a 3090.
    Both come from the local nvidia-smi.
    """
    step(2, 8, "GPU & CUDA detection")

    smi_path = shutil.which("nvidia-smi")
    if IS_WINDOWS and smi_path is None:
        candidate = Path(r"C:\Windows\System32\nvidia-smi.exe")
        if candidate.exists():
            smi_path = str(candidate)

    if smi_path is None:
        warn("nvidia-smi not found — assuming no NVIDIA GPU.")
        warn("The pipeline will run on CPU.  Processing will be slow.")
        return ["cpu"], None

    result = run([smi_path])
    if result.returncode != 0:
        warn("nvidia-smi returned an error.  Falling back to CPU mode.")
        return ["cpu"], None

    gpu_match = re.search(r"(?:GeForce|Quadro|Tesla|RTX|GTX|A\d)\s[\w\s]+", result.stdout)
    gpu_name  = gpu_match.group(0).strip() if gpu_match else "Unknown GPU"

    cuda_match = re.search(r"CUDA Version:\s*([\d.]+)", result.stdout)
    if not cuda_match:
        warn(f"GPU found ({gpu_name}) but could not read CUDA version.")
        return ["cpu"], None

    cuda_ver   = cuda_match.group(1)
    cuda_major = int(cuda_ver.split(".")[0])

    cc = _detect_compute_cap(smi_path)

    ok(f"GPU:  {gpu_name}")
    ok(f"CUDA: {cuda_ver}")
    ok(f"Compute capability: {cc if cc is not None else 'unknown'}")

    # Old GPU (pre-Turing): force the legacy wheel set
    if cc is not None and cc < 7.5:
        warn(f"GPU compute capability {cc} is older than Turing (sm_75).")
        warn("PyTorch 2.5+ dropped sm_60/sm_61 kernels.  Will try PyTorch 2.4.0→2.2.2")
        warn("on cu118 wheels, which still ship sm_60/sm_61 support.")
        # Return a sentinel; install_pytorch handles the actual candidate loop
        return ["__legacy__"], cc

    tags = _CUDA_TAG_CANDIDATES.get(cuda_major)
    if tags is None:
        if cuda_major < 11:
            warn(f"CUDA {cuda_ver} is too old (need ≥ 11).  Falling back to CPU PyTorch.")
            return ["cpu"], cc
        tags = ["cu128", "cu126", "cu124", "cu121"]

    ok(f"Will try wheel tags (in order): {', '.join(tags)}")
    return tags, cc


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
def _torch_works_on_gpu(gpu_cc: float | None = None) -> bool:
    """
    Returns True only if the installed torch:
      (a) reports CUDA available,
      (b) was compiled with kernels for this GPU's compute capability
          (checked deterministically via torch.cuda.get_arch_list()), and
      (c) can actually run a conv2d on the GPU — which uses cuDNN, the same
          path YOLO takes.  Element-wise ops like `x + 1` go through JIT-PTX
          and may pass even when cuDNN doesn't ship the relevant arch, so
          we test conv2d specifically.
    """
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return False

        # Deterministic check: is the host GPU's arch in the wheel's arch list?
        if gpu_cc is None:
            gpu_cc = float(f"{torch.cuda.get_device_capability(0)[0]}."
                           f"{torch.cuda.get_device_capability(0)[1]}")
        host_arch = f"sm_{int(gpu_cc * 10)}"
        arch_list = torch.cuda.get_arch_list()  # e.g. ['sm_75','sm_80', ...]
        if host_arch not in arch_list:
            return False

        # Runtime check: actually launch a conv2d (the kind YOLO uses)
        x   = torch.randn(1, 3, 16, 16, device="cuda")
        net = torch.nn.Conv2d(3, 4, 3).cuda()
        _   = net(x).sum().cpu().item()
        return True
    except Exception:
        return False


def install_pytorch(cuda_tags: list[str], compute_cap: float | None) -> None:
    step(4, 8, "PyTorch")

    want_gpu   = cuda_tags != ["cpu"]
    legacy_gpu = compute_cap is not None and compute_cap < 7.5

    def _purge_torch() -> None:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision"],
            capture_output=True,
        )

    def _reload_and_check() -> bool:
        """Force-reload torch modules and confirm the GPU actually works."""
        if not want_gpu:
            return True
        for mod in list(sys.modules):
            if mod == "torch" or mod.startswith("torch."):
                del sys.modules[mod]
        return _torch_works_on_gpu(compute_cap)

    def _pip_install(torch_spec: str, vision_spec: str, url: str) -> bool:
        """pip-install the given specs from url. Returns True on exit 0."""
        info("  (PyTorch wheels are 2+ GB — download may take several minutes)")
        info("  pip output streaming below:")
        print(_c("90", "  " + "─" * 60))
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "--progress-bar", "on",
             torch_spec, vision_spec, "--index-url", url],
        )
        print(_c("90", "  " + "─" * 60))
        return result.returncode == 0

    # Check if a working install is already in place
    try:
        import torch  # type: ignore
        if want_gpu:
            if _torch_works_on_gpu(compute_cap):
                ok(f"PyTorch {torch.__version__} already installed and runs on GPU")
                return
            arch_list = []
            try:
                arch_list = torch.cuda.get_arch_list()
            except Exception:
                pass
            info(f"PyTorch {torch.__version__} is installed but does not run on this GPU.")
            info(f"  GPU arch needed: sm_{int((compute_cap or 0) * 10)}")
            info(f"  Wheel arch list: {', '.join(arch_list) or 'unknown'}")
            info("Uninstalling so we can install a compatible build ...")
            _purge_torch()
        else:
            ok(f"PyTorch {torch.__version__} already installed (CPU mode)")
            return
    except ImportError:
        pass

    # ── Legacy GPU path (Pascal / CC < 7.5) ─────────────────────────────────
    # sm_61 was dropped in PyTorch 2.5 (both cu121 and cu118 builds).
    # Try each candidate from newest to oldest until one actually works.
    if legacy_gpu:
        for tv, vv, tag in _LEGACY_GPU_CANDIDATES:
            info(f"Trying PyTorch {tv}+{tag} / torchvision {vv}  (legacy GPU build)")
            url = f"https://download.pytorch.org/whl/{tag}"
            if not _pip_install(f"torch=={tv}", f"torchvision=={vv}", url):
                warn(f"  pip failed for torch=={tv}+{tag} — trying next candidate")
                continue
            ok(f"Installed torch=={tv}+{tag} — verifying on GPU ...")
            if _reload_and_check():
                ok(f"PyTorch {tv}+{tag} runs on this GPU")
                break
            warn(f"  torch=={tv}+{tag} installed but does not run on this GPU (sm_61 missing).")
            warn("  Trying an older release ...")
            _purge_torch()
        else:
            fail(
                "None of the legacy PyTorch builds ran on this GPU.\n\n"
                f"  Python:          {sys.version.split()[0]}\n"
                f"  GPU compute cap: {compute_cap}\n\n"
                "  The GTX 1080 Ti (sm_61) needs torch ≤ 2.4.x built for cu118.\n"
                "  If you are on Python 3.12 and this still fails, try:\n"
                "    pip install torch==2.4.0 torchvision==0.19.0 \\\n"
                "      --index-url https://download.pytorch.org/whl/cu118\n"
                "  and check https://pytorch.org/get-started/previous-versions/"
            )

    # ── Modern GPU path ──────────────────────────────────────────────────────
    else:
        for tag in cuda_tags:
            if tag == "cpu":
                url = "https://download.pytorch.org/whl/cpu"
                info("Installing CPU-only PyTorch ...")
            else:
                url = f"https://download.pytorch.org/whl/{tag}"
                info(f"Trying PyTorch wheel index: {tag}")
            if not _pip_install("torch", "torchvision", url):
                warn(f"  {tag} did not yield a usable wheel — trying next tag")
                continue
            ok(f"PyTorch installed from {tag} — verifying ...")
            if _reload_and_check():
                ok(f"PyTorch from {tag} runs successfully on this GPU")
                break
            warn(f"  {tag} wheel does not run on this GPU. Trying next tag.")
            _purge_torch()
        else:
            fail(
                "Could not install a PyTorch build that works on this system.\n\n"
                f"  Python:          {sys.version.split()[0]}\n"
                f"  GPU compute cap: {compute_cap}\n\n"
                "  Fix options:\n"
                "    A) Install Python 3.12 from python.org and rerun:\n"
                "         py -3.12 install.py\n"
                "    B) Check https://pytorch.org/get-started/locally/ for the\n"
                "       right wheel for your GPU and Python."
            )

    import torch  # re-import after install (modules were cleared above)
    ok(f"PyTorch {torch.__version__} installed and verified")


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
def verify(compute_cap: float | None = None) -> None:
    step(8, 8, "Final verification")

    sys.path.insert(0, str(SCRIPT_DIR))
    errors: list[str] = []

    # ---- Config & stage imports --------------------------------------------
    try:
        import config  # type: ignore
        path = config.config_file_path()
        ok(f"config        loaded from {path or 'built-in defaults'}")
    except Exception as e:
        errors.append(f"config: {e}")

    for mod in ["stages.ingest", "stages.segment", "stages.sampler",
                "stages.scorer", "stages.exporter"]:
        try:
            __import__(mod)
            ok(f"{mod.split('.')[1]:<14}")
        except Exception as e:
            errors.append(f"{mod}: {e}")

    # ---- PyTorch: CUDA available, arch list covers GPU, conv2d runs --------
    try:
        # Force a fresh import in case torch was just (un)installed
        for m in list(sys.modules):
            if m == "torch" or m.startswith("torch."):
                del sys.modules[m]
        import torch  # type: ignore

        cuda = torch.cuda.is_available()
        dev  = torch.cuda.get_device_name(0) if cuda else "CPU"
        ok(f"torch          {torch.__version__}  device={dev}")

        if cuda:
            arch_list = torch.cuda.get_arch_list()
            ok(f"torch arch     {', '.join(arch_list)}")
            try:
                x   = torch.randn(1, 3, 16, 16, device="cuda")
                net = torch.nn.Conv2d(3, 4, 3).cuda()
                _   = net(x).sum().cpu().item()
                ok("conv2d         runs on GPU (cuDNN OK)")
            except RuntimeError as e:
                errors.append(
                    "conv2d failed on GPU — wheel architecture does not "
                    "cover this GPU's compute capability.\n"
                    f"             {e}"
                )
    except Exception as e:
        errors.append(f"torch: {e}")

    # ---- OpenCV: import + decode a 1px JPEG ---------------------------------
    try:
        import cv2  # type: ignore
        import numpy as np
        # Build a 1x1 red JPEG in-memory and decode it back
        ok_jpeg, buf = cv2.imencode(".jpg", np.zeros((4, 4, 3), dtype=np.uint8))
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if not ok_jpeg or decoded is None:
            raise RuntimeError("cv2 round-trip jpeg encode/decode failed")
        ok(f"opencv         {cv2.__version__}  (encode+decode OK)")
    except Exception as e:
        errors.append(f"opencv: {e}")

    # ---- EasyOCR: import works (full model load deferred to first use) -----
    try:
        import easyocr  # type: ignore  # noqa
        ok("easyocr        import OK")
    except Exception as e:
        errors.append(f"easyocr: {e}")

    # ---- Streamlit + Anthropic SDK + tomlkit -------------------------------
    for name in ("streamlit", "anthropic", "tomlkit", "pandas"):
        try:
            __import__(name)
            ok(f"{name:<14} import OK")
        except Exception as e:
            errors.append(f"{name}: {e}")

    # ---- ffmpeg can run --------------------------------------------------
    if shutil.which("ffmpeg"):
        result = run(["ffmpeg", "-version"])
        if result.returncode == 0:
            ver = re.search(r"version ([\d.]+)", result.stdout or "")
            ok(f"ffmpeg         {ver.group(1) if ver else 'found'}")
        else:
            errors.append(f"ffmpeg exited {result.returncode}")
    else:
        warn("ffmpeg not in PATH — open a new terminal after install and retry if needed.")

    # ---- exiftool can run -----------------------------------------------
    if shutil.which("exiftool"):
        result = run(["exiftool", "-ver"])
        if result.returncode == 0:
            ok(f"exiftool       {result.stdout.strip()}")
        else:
            errors.append(f"exiftool exited {result.returncode}")
    else:
        warn("exiftool not in PATH — GPS extraction will fall back to ffprobe.")

    # ---- YOLO: load weights and run inference on a synthetic frame -------
    if not errors:
        info("Running YOLO smoke test (loads model, runs on a 640×640 black frame)...")
        try:
            from ultralytics import YOLO  # type: ignore
            import numpy as np
            model = YOLO(_yolo_model_name())
            blank = np.zeros((640, 640, 3), dtype=np.uint8)
            results = model(blank, verbose=False)
            ok(f"yolo smoke     ran inference on {len(results)} image(s)")
        except Exception as e:
            errors.append(f"yolo smoke test: {e}")

    # ---- Result -----------------------------------------------------------
    if errors:
        print()
        for e in errors:
            warn(f"Issue: {e}")
        print()
        warn("Some checks failed.  Pipeline runs may not work until these are resolved.")
        warn("Re-run install.py after addressing the issues above.")
    else:
        _print_success()


def _yolo_model_name() -> str:
    """Read the model name from the user TOML config, or use the default."""
    if CONFIG_FILE.exists():
        try:
            import tomllib
            with open(CONFIG_FILE, "rb") as f:
                data = tomllib.load(f)
            return data.get("detection", {}).get("model", "yolov8x-oiv7.pt")
        except Exception:
            pass
    return "yolov8x-oiv7.pt"


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

  ─────────────────────────────────────────────────────
  Recommended:  start the UI
  ─────────────────────────────────────────────────────
    Windows:    launch_ui.bat   (double-click in Explorer)
    Linux/Mac:  ./launch_ui.sh
    Or:         python -m streamlit run app.py

  The UI lets you set your API key, pick a video, see live
  processing logs, and browse results — all in your browser.

  ─────────────────────────────────────────────────────
  Or run from the command line
  ─────────────────────────────────────────────────────
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
    cuda_tags, compute_cap = detect_gpu()
    install_system_deps()
    install_pytorch(cuda_tags, compute_cap)
    install_packages()
    predownload_model()
    place_config()
    verify(compute_cap)


if __name__ == "__main__":
    main()
