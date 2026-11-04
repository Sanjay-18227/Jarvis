# Hardware Compatibility

## 1. This machine (verified)

| Component | Value |
|---|---|
| Laptop | Acer (owner's machine) |
| CPU | AMD Ryzen 3 7320U, 4 physical cores / 8 logical threads |
| RAM | 8 GB installed (7.3 GB usable) |
| GPU | AMD Radeon integrated (no CUDA) |
| OS | Windows 11 Home, 10.0.26300, 64-bit |
| Free disk (project drive) | ~68 GB |
| Python | 3.13.14 in project `venv/` |
| Torch | 2.15.0+cpu |
| Transformers | 5.22.1 |

**Inference mode:** CPU only. No CUDA, no ROCm, no DirectML requirement.
PyTorch's CPU build is used for Qwen3, ECAPA and optional vision/TTS models.
Faster-Whisper runs through CTranslate2 with int8 weights on CPU.

---

## 2. Minimum / recommended configurations

| Tier | CPU | RAM | Disk | Experience |
|---|---|---|---|---|
| **Minimum** | 4 cores, AVX2 (2015+) | 8 GB | 10 GB free | Voice + skills + Whisper-tiny + ECAPA; SLM may be skipped under pressure |
| **Recommended** | 4–8 cores | 16 GB | 20 GB free | SLM stays resident longer; vision/TTS options become practical |
| **GPU (optional)** | CUDA-capable NVIDIA | 8 GB+ VRAM | 20 GB | CUDA fp16/bf16 for Qwen3/VLM; not required |
| **Not supported** | 2 cores / 4 GB RAM | < 8 GB | < 5 GB | Model loads fail or Windows pages heavily |

Windows 10/11 64-bit is required. Python 3.13 is the tested interpreter.

---

## 3. Actual RAM budget on this laptop

| Component | Resident estimate | Notes |
|---|---|---|
| Python + core JARVIS | 250–450 MB | voice, skills, security |
| Silero VAD | ~60 MB | tiny, always available |
| Faster-Whisper tiny (int8) | ~200–320 MB | loaded on first voice command |
| Faster-Whisper base (int8) | ~400–600 MB | on demand only |
| ECAPA speaker model | ~180 MB | loaded for auth, kept hot |
| Qwen3-0.6B (fp32) | **~2.5 GB** | on demand; unloaded after idle |
| SmolVLM-256M (optional) | ~0.9–1.2 GB | disabled by default |
| indic-parler TTS (optional) | ~1.8 GB | disabled by default |

The model manager allows **one heavy model** at a time. Closing browsers/IDEs
before asking for SLM reasoning is recommended on 8 GB.

### Measured SLM gate

Qwen3-0.6B requires **≥ 2200 MB free RAM** before loading. Below that, the
model manager returns `None` and JARVIS continues with deterministic rules
(no crash, no swap storm).

---

## 4. CPU threading

`core/system/hardware_detector.py` sets `threads = physical_cores` (4 here).
Oversubscribing the 8 logical threads made generation slower in practice.
Torch is capped with `torch.set_num_threads(4)`; CTranslate2 uses
`cpu_threads=4`.

---

## 5. GPU notes

- CUDA is auto-detected without importing torch at startup (via `nvidia-smi`),
  keeping startup fast on CPU-only machines.
- If a CUDA GPU is present, Qwen3 loads in bf16 on CUDA and the RAM gates are
  relaxed by the model manager. `torch_device` will show `cuda`.
- AMD integrated GPUs are **not** used for torch inference on this build.
  DirectML support is optional and not part of the tested path.
- **ECAPA is security-critical:** a GPU failure must not disable voice auth
  permanently. The CPU path remains the fallback.

---

## 6. Disk and model sizes

| Item | Measured / expected |
|---|---|
| Qwen3-0.6B snapshot | ~1449 MB |
| Faster-Whisper tiny (CT2) | ~75 MB |
| Faster-Whisper base (CT2) | ~145 MB |
| ECAPA VoxCeleb | ~85 MB |
| SmolVLM-256M (optional) | ~600 MB |
| indic-parler TTS (optional) | ~2.2 GB |
| Project + venv | ~3–4 GB |

`download()` requires at least **2×** the model size in free disk before starting.

---

## 7. Known hardware limitations

- 8 GB RAM cannot hold Qwen3 + a browser + an IDE comfortably; the SLM is
  on-demand for this reason.
- Local vision is disabled by default; structural UI inspection is the
  fallback and is always available.
- Background jobs only run while the laptop is powered on and JARVIS is running.
- Wake-word accuracy depends on the microphone and the installed wake-word
  model; Silero VAD + RMS gating is the first line of noise rejection.

---

## 8. Verification commands

```powershell
# Hardware snapshot + model policy
.\venv\Scripts\python.exe doctor.py

# Model load/generation benchmark
.\venv\Scripts\python.exe benchmark_models.py --vad --asr --speaker
.\venv\Scripts\python.exe benchmark_models.py --slm

# Current policy JSON
.\venv\Scripts\python.exe -m core.models.model_manager
```

See `docs/TEST_RESULTS.md` for the latest measured numbers on this machine.
