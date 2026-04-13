"""System information collection for agent context injection.

Collects OS, CPU, RAM, GPU, and installed Python packages so that
the Operation Agent can generate code adapted to the actual environment.
"""

import os
import platform
import subprocess
import sys
from typing import Optional


# ML-relevant packages to highlight in the short report
_KEY_PACKAGES = [
    "numpy", "pandas", "polars", "scipy", "scikit-learn",
    "xgboost", "lightgbm", "catboost", "optuna", "hyperopt",
    "torch", "torchvision", "torchaudio",
    "tensorflow", "keras",
    "transformers", "datasets", "huggingface-hub",
    "shap", "lime",
    "matplotlib", "seaborn", "plotly",
    "joblib", "dask", "ray",
    "fastapi", "gradio", "streamlit",
    "kaggle", "boto3",
]


def collect_system_info() -> dict:
    """Return a dict with OS, hardware, and installed-package info."""
    info: dict = {}

    # ── OS & Python ──────────────────────────────────────────────────────────
    info["os"] = platform.platform()
    info["python"] = sys.version.split()[0]

    # ── CPU & RAM ────────────────────────────────────────────────────────────
    try:
        import psutil
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["cpu_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    info["cpu_name"] = platform.processor() or platform.machine() or "unknown"

    # ── GPU ──────────────────────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["gpu"] = result.stdout.strip()
        else:
            info["gpu"] = "none"
    except Exception:
        info["gpu"] = "none"

    # ── Installed packages ───────────────────────────────────────────────────
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            packages: dict[str, str] = {}
            for line in result.stdout.strip().splitlines():
                if "==" in line:
                    name, ver = line.split("==", 1)
                    packages[name.lower().replace("-", "_")] = ver
            info["packages"] = packages
    except Exception:
        info["packages"] = {}

    return info


def format_for_agent(info: Optional[dict] = None) -> str:
    """Return a human-readable string suitable for injection into an agent prompt."""
    if info is None:
        info = collect_system_info()

    lines = [
        "## Execution Environment",
        f"- OS: {info.get('os', 'unknown')}",
        f"- Python: {info.get('python', 'unknown')}",
    ]

    cpu = info.get("cpu_name", "")
    phys = info.get("cpu_physical", "?")
    logi = info.get("cpu_logical", "?")
    if cpu or phys != "?":
        lines.append(f"- CPU: {cpu} — {phys} physical cores / {logi} threads")

    if "ram_gb" in info:
        lines.append(f"- RAM: {info['ram_gb']} GB")

    gpu = info.get("gpu", "none")
    lines.append(f"- GPU: {gpu}")

    pkgs = info.get("packages", {})
    if pkgs:
        # Highlight key ML packages
        key_installed = {p: pkgs[p.replace("-", "_")] for p in _KEY_PACKAGES if p.replace("-", "_") in pkgs}
        if key_installed:
            pkg_str = ", ".join(f"{k}=={v}" for k, v in sorted(key_installed.items()))
            lines.append(f"- Key packages: {pkg_str}")
        # Full list (for completeness / LLM context)
        all_str = ", ".join(f"{k}=={v}" for k, v in sorted(pkgs.items()))
        lines.append(f"- All installed packages: {all_str}")

    return "\n".join(lines)
