"""
Hardware Detection Subsystem
Detects CPU, RAM, disk, GPU (CUDA/Torch), Python version, current-process memory
and available model storage so the model manager can make resource-aware decisions.

 Detection is best-effort and must never crash JARVIS startup:
 every field falls back to a safe default.
"""

import os
import sys
import shutil
import platform
import logging

log = logging.getLogger("jarvis.hardware")

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class HardwareInfo(dict):
    """dict subclass so hardware info is JSON-serializable and easy to log."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def _detect_cpu_name() -> str:
    name = ""
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=15,
        )
        name = out.stdout.strip()
    except Exception:
        name = platform.processor() or ""
    return name or "Unknown CPU"


def _detect_cpu_metrics() -> dict:
    cores = logical = freq = 0
    if psutil:
        try:
            cores = psutil.cpu_count(logical=False) or 0
            logical = psutil.cpu_count(logical=True) or 0
            freq = int(psutil.cpu_freq().current) if psutil.cpu_freq() else 0
        except Exception:
            pass
    return {"physical_cores": cores, "logical_cores": logical, "current_freq_mhz": freq}


def _detect_ram() -> dict:
    total = available = 0
    if psutil:
        try:
            vm = psutil.virtual_memory()
            total = int(vm.total // (2 ** 20))
            available = int(vm.available // (2 ** 20))
        except Exception:
            pass
    return {"total_ram_mb": total, "available_ram_mb": available}


def _detect_disks(paths) -> dict:
    free_total = 0
    details = {}
    for base in paths:
        try:
            free = shutil.disk_usage(base).free  # bytes
            free_total += free
            details[base] = {"drive": os.path.splitdrive(base)[0], "free_gb": round(free / (2 ** 30), 2)}
        except Exception:
            continue
    return {"free_disk_bytes": int(free_total), "model_roots": details}


def _detect_gpu():
    """Detects GPU/CUDA without importing torch (import cost ~8s at startup).

    Uses nvidia-smi for CUDA capability; torch itself detects the device at model
    load time. Falls back to CPU on any AMD/Intel integrated GPU (our target laptop).
    """
    info = {
        "gpu_name": None,
        "gpu_mem_mb": 0,
        "cuda_available": False,
        "directml_available": False,
        "torch_device": "cpu",
        "torch_version": None,
    }
    cuda = False
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=8)
        if out.returncode == 0 and out.stdout.strip():
            cuda = bool(__import__("importlib").util.find_spec("torch"))
            if cuda:
                info["cuda_available"] = True
                info["torch_device"] = "cuda"
    except Exception:
        pass
    info["gpu_name"] = _wmi_gpu_name()
    if cuda:
        info["gpu_mem_mb"] = _nvidia_memory_mb()
    # DirectML is optional and only advertised if the package is importable (cheap check)
    try:
        if __import__("importlib").util.find_spec("torch_directml"):
            info["directml_available"] = True
    except Exception:
        pass
    try:
        import importlib.metadata as md
        info["torch_version"] = md.version("torch")
    except Exception:
        pass
    return info


def _nvidia_memory_mb() -> int:
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
        first = out.stdout.strip().splitlines()[0]
        return int(first.strip())
    except Exception:
        return 0


def _wmi_gpu_name():
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController | Select-Object -First 1).Name"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _process_memory_mb() -> int:
    if not psutil:
        return 0
    try:
        return int(psutil.Process().memory_info().rss // (2 ** 20))
    except Exception:
        return 0


def detect_hardware(model_roots=None) -> HardwareInfo:
    """One-shot hardware snapshot returned as a JSON-friendly dict-like object."""
    if model_roots is None:
        try:
            from config.settings import get_settings
            model_roots = [get_settings().model_dir]
        except Exception:
            model_roots = [os.path.join(os.getcwd(), "models")]

    cpu_name = _detect_cpu_name()
    cpu_metrics = _detect_cpu_metrics()
    ram = _detect_ram()
    disks = _detect_disks(model_roots)
    gpu = _detect_gpu()

    info = HardwareInfo(
        cpu_name=cpu_name,
        physical_cores=cpu_metrics["physical_cores"],
        logical_cores=cpu_metrics["logical_cores"],
        current_freq_mhz=cpu_metrics["current_freq_mhz"],
        total_ram_mb=ram["total_ram_mb"],
        available_ram_mb=ram["available_ram_mb"],
        free_disk_bytes=disks["free_disk_bytes"],
        model_roots=disks["model_roots"],
        gpu_name=gpu.get("gpu_name"),
        gpu_mem_mb=gpu.get("gpu_mem_mb", 0),
        cuda_available=gpu.get("cuda_available", False),
        directml_available=gpu.get("directml_available", False),
        torch_device=gpu.get("torch_device", "cpu"),
        torch_version=gpu.get("torch_version"),
        python_version=platform.python_version(),
        process_memory_mb=_process_memory_mb(),
        platform=platform.platform()
    )
    log.info("[HARDWARE] %s", dict(info))
    return info


def model_policy(info: HardwareInfo) -> dict:
    """
    Maps hardware to a conservative model-load policy.

    Rules:
      - torch device = cuda if available else cpu
      - Qwen3-0.6B only loaded if >= 1000 MB available RAM
      - whisper only loaded if >= 250 MB available RAM
      - SmolVLM only if >= 1500 MB available RAM
      - ECAPA speaker model (security-critical, ~90 MB) loads unless avail < 250 MB
    """
    avail = info.get("available_ram_mb", 0) if hasattr(info, "get") else 0
    policy = {
        "torch_device": info.get("torch_device", "cpu") if hasattr(info, "get") else "cpu",
        # CPU inference: cap to physical cores; SMT threads add contention on Ryzen 3.
        "threads": max(2, (info.get("physical_cores") or max(1, (info.get("logical_cores") or 2) // 2))
                       if hasattr(info, "get") else 2),
        "allow_slm": True,
        "allow_whisper": True,
        "allow_vlm": True,
        "allow_ecapa": True,
        "reason": "base policy",
    }
    # fp32 Qwen3-0.6B needs ~2.5 GB RSS; require headroom to avoid Windows paging.
    if avail < 2200:
        policy["allow_slm"] = False
        policy["reason"] = f"SLM needs ~2.5GB RAM headroom (available {avail} MB)"
    if avail < 250:
        policy["allow_ecapa"] = False
        policy["reason"] = f"critical RAM ({avail} MB) -> voice auth fail-closed, use PIN recovery"
    if avail < 700:
        policy["allow_whisper"] = False
        policy["allow_vlm"] = False
        policy["reason"] = f"low available RAM ({avail} MB) -> only lightweight engines"
    elif avail < 1500:
        policy["allow_vlm"] = False
        if policy["reason"] == "base policy":
            policy["reason"] = "moderate RAM -> skip VLM by default"
    return policy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    hw = detect_hardware()
    print("\n=== JARVIS Hardware Report ===")
    for key, value in hw.items():
        if key != "model_roots":
            print(f"{key}: {value}")
    import json
    print("\nmodel_policy:", json.dumps(model_policy(hw), indent=2))
