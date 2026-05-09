#!/usr/bin/env python3
"""
NarcPartrol diagnostic — run this and paste the output to the dev.
Reads only; makes no changes to your system.
"""
import sys, platform, subprocess, shutil, os, json
from pathlib import Path

SEP = "─" * 60

def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERROR: {e}"

def header(t): print(f"\n{SEP}\n  {t}\n{SEP}")

# ── Python ─────────────────────────────────────────────────────────────────
header("Python interpreters on PATH / py launcher")
print(f"This interpreter: {sys.executable}")
print(f"Version:          {sys.version}")
print(f"Platform:         {platform.platform()}")
print(f"Architecture:     {platform.machine()}")

# Windows py launcher — list all installed versions
if platform.system() == "Windows":
    print("\npy --list output:")
    print(run(["py", "--list"]))

# Other pythons on PATH
for cmd in ["python3.14","python3.13","python3.12","python3.11","python3.10","python3","python"]:
    path = shutil.which(cmd)
    if path:
        ver = run([path, "--version"])
        print(f"  {cmd:<16} {path}  →  {ver}")

# ── GPU / CUDA ──────────────────────────────────────────────────────────────
header("NVIDIA GPU & CUDA")
smi = shutil.which("nvidia-smi")
if not smi and platform.system() == "Windows":
    candidate = Path(r"C:\Windows\System32\nvidia-smi.exe")
    if candidate.exists():
        smi = str(candidate)

if smi:
    print(run([smi]))
    print("\nnvidia-smi --query-gpu:")
    print(run([smi,
               "--query-gpu=name,compute_cap,memory.total,driver_version",
               "--format=csv,noheader"]))
else:
    print("nvidia-smi NOT FOUND")

# ── PyTorch (if installed) ──────────────────────────────────────────────────
header("PyTorch")
try:
    import torch  # type: ignore
    print(f"torch version:  {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version:   {torch.version.cuda}")
        print(f"Device:         {torch.cuda.get_device_name(0)}")
        cc = torch.cuda.get_device_capability(0)
        print(f"Compute cap:    {cc[0]}.{cc[1]}")
        print(f"Arch list:      {torch.cuda.get_arch_list()}")
except ImportError:
    print("torch NOT installed in this interpreter")
except Exception as e:
    print(f"torch error: {e}")

# ── Key packages ────────────────────────────────────────────────────────────
header("Key pip packages")
pkgs = ["ultralytics","easyocr","opencv-python","onnxruntime","onnxruntime-gpu",
        "streamlit","anthropic","tomlkit","ffmpeg-python","geopy","tqdm","numpy",
        "pandas","Pillow","requests"]
out = run([sys.executable, "-m", "pip", "show"] + pkgs)
# Compact: just Name + Version lines
for line in out.splitlines():
    if line.startswith("Name:") or line.startswith("Version:"):
        print(line)

# ── System tools ────────────────────────────────────────────────────────────
header("System tools")
for tool in ["ffmpeg","exiftool","winget","choco","git"]:
    path = shutil.which(tool)
    if path:
        ver = run([tool, "-version" if tool == "ffmpeg" else "--version"])
        first = ver.splitlines()[0] if ver else ""
        print(f"  {tool:<12} {path}")
        print(f"             {first[:80]}")
    else:
        print(f"  {tool:<12} NOT FOUND")

# ── NarcPartrol config ──────────────────────────────────────────────────────
header("NarcPartrol config")
for p in [
    Path.home() / "Documents" / "NarcPartrol" / "narcpartrol.toml",
    Path.home() / ".config" / "narcpartrol" / "narcpartrol.toml",
]:
    if p.exists():
        print(f"Found: {p}")
        print(p.read_text(encoding="utf-8")[:800])
    else:
        print(f"Not found: {p}")

header("END OF DIAGNOSTIC")
