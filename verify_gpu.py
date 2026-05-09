#!/usr/bin/env python3
"""
NarcPartrol GPU verification — proves which hard stages run on the GPU.

Run after install.py:
    python verify_gpu.py
    py -3.14 verify_gpu.py

Reports, for each compute-heavy stage:
  * what backend is loaded (ORT CUDA EP, ORT CPU EP, torch CUDA, torch CPU)
  * whether inference actually lands on the GPU
  * a wall-clock timing on a 16-frame batch (the pipeline's batch_size)

Read-only; makes no installs and no edits.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import config  # noqa: E402

SEP = "─" * 64


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

def ok(m):   print(_c("32", f"  ✓  {m}"))
def warn(m): print(_c("33", f"  ⚠  {m}"))
def info(m): print(_c("36", f"  →  {m}"))
def bad(m):  print(_c("31", f"  ✗  {m}"))
def hdr(m):  print(f"\n{SEP}\n  {m}\n{SEP}")


# ---------------------------------------------------------------------------
# 1. nvidia-smi sanity
# ---------------------------------------------------------------------------
def check_nvidia_smi() -> None:
    hdr("1. NVIDIA driver / GPU")
    import shutil, subprocess
    smi = shutil.which("nvidia-smi")
    if not smi and sys.platform == "win32":
        cand = Path(r"C:\Windows\System32\nvidia-smi.exe")
        if cand.exists():
            smi = str(cand)
    if not smi:
        bad("nvidia-smi not found — no NVIDIA driver visible.")
        return
    try:
        r = subprocess.run(
            [smi, "--query-gpu=name,compute_cap,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        for line in (r.stdout or "").strip().splitlines():
            ok(f"GPU: {line}")
    except Exception as e:
        bad(f"nvidia-smi failed: {e}")


# ---------------------------------------------------------------------------
# 2. PyTorch (informational — Pascal/Windows runs torch CPU by design)
# ---------------------------------------------------------------------------
def check_torch() -> None:
    hdr("2. PyTorch")
    try:
        import torch
        info(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            ok(f"CUDA available — device: {torch.cuda.get_device_name(0)}")
            ok(f"arch list: {', '.join(torch.cuda.get_arch_list())}")
        else:
            warn("CUDA not available in torch (expected on Pascal+Windows).")
            warn("  YOLO will go through onnxruntime-gpu instead — verified below.")
            warn("  EasyOCR uses torch, so it will run on CPU. That's fine: OCR")
            warn("  runs once per house, not per frame, so it isn't the bottleneck.")
    except ImportError:
        bad("torch is not installed.")


# ---------------------------------------------------------------------------
# 3. ONNX Runtime — the actual GPU path for YOLO
# ---------------------------------------------------------------------------
def check_onnxruntime() -> bool:
    """Returns True if CUDAExecutionProvider is available."""
    hdr("3. ONNX Runtime")
    try:
        import onnxruntime as ort
    except ImportError:
        bad("onnxruntime not installed.")
        return False
    info(f"onnxruntime {ort.__version__}")
    providers = ort.get_available_providers()
    info(f"available providers: {', '.join(providers)}")
    if "CUDAExecutionProvider" in providers:
        ok("CUDAExecutionProvider is available.")
        return True
    bad("CUDAExecutionProvider NOT available — GPU inference will not work.")
    bad("  Fix:  pip install --force-reinstall onnxruntime-gpu")
    return False


# ---------------------------------------------------------------------------
# 4. Direct ORT GPU smoke test on the YOLO ONNX model
# ---------------------------------------------------------------------------
def check_yolo_onnx_direct(have_cuda_ep: bool) -> None:
    hdr("4. YOLO ONNX direct inference (bypassing ultralytics)")
    model_name = config.YOLO_MODEL
    if not model_name.lower().endswith(".onnx"):
        warn(f"config.YOLO_MODEL = {model_name!r} — not an ONNX model.")
        warn("  On Pascal+Windows the installer should have set this to a .onnx file.")
        warn("  Re-run install.py if the ORT fallback didn't activate.")
        return

    onnx_path = Path(model_name)
    if not onnx_path.is_absolute():
        onnx_path = SCRIPT_DIR / onnx_path
    if not onnx_path.exists():
        bad(f"ONNX model not found at {onnx_path}")
        return
    ok(f"model: {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    import numpy as np

    if not have_cuda_ep:
        warn("Skipping GPU session creation — CUDAExecutionProvider missing.")
        return

    # Build a session that REQUIRES CUDA EP first, with CPU fallback.
    try:
        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    except Exception as e:
        bad(f"Failed to create ORT session: {e}")
        return

    active = sess.get_providers()
    info(f"session providers (in priority order): {', '.join(active)}")
    if active and active[0] == "CUDAExecutionProvider":
        ok("YOLO ONNX session is using CUDAExecutionProvider.")
    else:
        bad(f"Session is NOT on CUDA — first provider is {active[0] if active else '?'}")
        return

    # Pull input shape from the model. Most YOLOv8 ONNX exports are
    # (1, 3, 640, 640); we batch up to 16 to match the pipeline.
    inp = sess.get_inputs()[0]
    name = inp.name
    shape = list(inp.shape)
    # Replace dynamic dims (strings or None) with concrete numbers
    shape = [16 if (i == 0) else (s if isinstance(s, int) and s > 0 else 640)
             for i, s in enumerate(shape)]
    info(f"input '{name}' shape: {shape}")

    x = np.random.rand(*shape).astype(np.float32)

    # Warm-up (compiles kernels / loads weights into VRAM)
    sess.run(None, {name: x[:1]})

    # Time a 16-frame batch
    t0 = time.perf_counter()
    sess.run(None, {name: x})
    dt = time.perf_counter() - t0
    ok(f"batch-of-{shape[0]} inference: {dt*1000:.1f} ms  "
       f"({shape[0] / dt:.1f} frames/sec)")

    # Quick CPU comparison so it's obvious the GPU is actually doing work
    try:
        cpu_sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        cpu_sess.run(None, {name: x[:1]})  # warm-up
        t0 = time.perf_counter()
        cpu_sess.run(None, {name: x[:4]})  # 4-frame CPU batch (full 16 is slow)
        cpu_dt = time.perf_counter() - t0
        info(f"CPU-only baseline (4 frames): {cpu_dt*1000:.1f} ms  "
             f"({4 / cpu_dt:.1f} frames/sec)")
        speedup = (4 / cpu_dt) / (shape[0] / dt) if (4 / cpu_dt) > 0 else 0
        if speedup and speedup < 1:
            ok(f"GPU is ~{(1/speedup):.1f}× faster than CPU on this model.")
    except Exception as e:
        warn(f"CPU baseline skipped: {e}")


# ---------------------------------------------------------------------------
# 5. Ultralytics path (the actual one the pipeline uses)
# ---------------------------------------------------------------------------
def check_ultralytics_path() -> None:
    hdr("5. Ultralytics → ONNX path used by the pipeline")
    try:
        from ultralytics import YOLO
    except ImportError:
        bad("ultralytics not installed.")
        return
    import numpy as np

    model_name = config.YOLO_MODEL
    if not model_name.lower().endswith(".onnx"):
        warn("YOLO_MODEL is a .pt file — pipeline will need PyTorch CUDA, which")
        warn("is not available on Pascal+Windows. Re-run install.py to switch")
        warn("to ONNX/ORT mode.")
        return

    try:
        model = YOLO(model_name)
    except Exception as e:
        bad(f"YOLO load failed: {e}")
        return

    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    # Warm-up
    try:
        model([blank], verbose=False, device=0)
    except Exception as e:
        bad(f"YOLO inference failed with device=0: {e}")
        return

    # Time the same 16-frame batch the pipeline uses
    batch = [blank] * config.YOLO_BATCH_SIZE
    t0 = time.perf_counter()
    results = model(batch, verbose=False, device=0)
    dt = time.perf_counter() - t0
    ok(f"ultralytics(device=0) batch-of-{len(batch)}: {dt*1000:.1f} ms  "
       f"({len(batch)/dt:.1f} frames/sec)  → {len(results)} result(s)")

    # Reflect into ultralytics' AutoBackend to confirm the active ORT providers.
    backend = getattr(model, "predictor", None)
    backend = getattr(backend, "model", None) if backend else None
    sess = getattr(backend, "session", None)
    if sess is not None and hasattr(sess, "get_providers"):
        active = sess.get_providers()
        info(f"ultralytics ORT session providers: {', '.join(active)}")
        if active and active[0] == "CUDAExecutionProvider":
            ok("Ultralytics is using CUDAExecutionProvider — GPU confirmed.")
        else:
            bad("Ultralytics is NOT on CUDA — providers: " + ", ".join(active))
            bad("  This means YOLO is silently running on CPU.")
            bad("  Check that onnxruntime-gpu is installed and that")
            bad("  CUDAExecutionProvider appears in section 3 above.")
    else:
        warn("Could not reflect into ultralytics ORT session to confirm provider.")
        warn("Compare the timing above with section 4's direct ORT timing —")
        warn("they should be roughly similar if both are on the GPU.")


# ---------------------------------------------------------------------------
# 6. EasyOCR — informational
# ---------------------------------------------------------------------------
def check_easyocr() -> None:
    hdr("6. EasyOCR (runs once per house — CPU is acceptable)")
    try:
        import easyocr  # noqa
        import torch
    except ImportError as e:
        bad(f"missing import: {e}")
        return
    if torch.cuda.is_available():
        ok("torch CUDA is available — EasyOCR will use GPU.")
    else:
        warn("torch CUDA unavailable — EasyOCR will run on CPU.")
        warn("  Acceptable: OCR only runs on the single best frame per house.")


def main() -> None:
    print(_c("1", "\n  NarcPartrol — GPU verification\n"))
    check_nvidia_smi()
    check_torch()
    have_cuda_ep = check_onnxruntime()
    check_yolo_onnx_direct(have_cuda_ep)
    check_ultralytics_path()
    check_easyocr()
    print()
    print(SEP)
    print("  Done.  If section 4 and section 5 both show CUDAExecutionProvider")
    print("  and similar throughput, GPU is doing the heavy lifting.")
    print(SEP)


if __name__ == "__main__":
    main()
