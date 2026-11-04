# Model Catalog

All models are free/open-source and fetched from Hugging Face. Each registry entry
pins a verified revision (commit SHA). Nothing here requires Ollama or a paid API.

Registry: `models/model_registry.yaml` · Manager: `core/models/model_manager.py`

---

## 1. Active models (default)

| Model ID | Hugging Face repo | Task | License | Pinned revision | Disk | RAM budget | Load policy | On this laptop |
|---|---|---|---|---|---|---|---|---|
| `silero-vad` | pip `silero-vad` 6.2.3 (bundled VAD v5) | Voice activity detection | MIT | 6.2.3 / model v5.1.2 | ~5 MB | ~60 MB | startup (tiny) | ✅ |
| `whisper-tiny-local` | `Systran/faster-whisper-tiny` | Speech-to-text (en/ta/mixed) | MIT | `d90ca5fe260221311c53c58e660288d3deb8d356` | ~75 MB measured | ~320 MB | on demand | ✅ downloaded |
| `whisper-base-local` | `Systran/faster-whisper-base` | Higher-accuracy ASR | MIT | `ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66` | ~141 MB measured | ~600 MB | on demand | ✅ downloaded |
| `ecapa-voxceleb-speaker` | `speechbrain/spkrec-ecapa-voxceleb` | Speaker verification (192-d) | Apache-2.0 | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` | ~85 MB measured | ~180 MB | on demand, kept hot for auth | ✅ downloaded |
| `qwen3-0.6b-instruct` | `Qwen/Qwen3-0.6B` | Intent JSON / local NLU | Apache-2.0 | `c1899de289a04d12100db370d81485cdf75e47ca` | ~1449 MB measured | ~2500 MB | on demand, idle unload | ✅ downloaded |

> Qwen3-0.6B is loaded in **fp32 on CPU**. Measured: ~5.8 s per short JSON generation,
> ~11 s load when RAM is healthy; int8 dynamic quantization was slower (~13.7 s) and is
> deliberately not used. The RAM gate requires ≥ 2200 MB free.

---

## 2. Optional models (disabled by default)

| Model ID | Hugging Face repo | Task | License | Pinned revision | Disk | RAM budget | Why disabled |
|---|---|---|---|---|---|---|---|
| `smolvlm-256m-instruct` | `HuggingFaceTB/SmolVLM-256M-Instruct` | Local screen description | Apache-2.0 | `7e3e67edbbed1bf9888184d9df282b700a323964` | ~600 MB | ~900 MB | 8 GB laptop; English-oriented, no Tamil OCR guarantee |
| `multilingual-e5-small` | `intfloat/multilingual-e5-small` | Semantic memory embeddings | MIT | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | ~480 MB | ~350 MB | SQLite FTS5 is the default and sufficient |
| `indic-parler-tts-mini` | `ai4bharat/indic-parler-tts` | Tamil/English TTS | Apache-2.0 | `660bc53c64ae83150613264111708a0465211237` | ~2200 MB | ~1800 MB | Not benchmarked on 8 GB CPU; Edge TTS/pyttsx3 remain default |

---

## 3. Runtime mapping (explicit loaders)

| `loader_method` | Backend | Notes |
|---|---|---|
| `load_whisper_tiny` / `load_whisper_base` | Faster-Whisper (CTranslate2) | int8 on CPU; resolved from `models/Systran__faster-whisper-*` |
| `load_ecapa_speaker` | SpeechBrain + PyTorch | 16 kHz mono, 192-d normalized embeddings |
| `load_qwen3_slm` | Transformers + PyTorch | fp32 CPU, non-thinking chat template |
| `load_smolvlm_vlm` | Transformers + PyTorch | Opt-in only; vision encoder + LLM |
| `load_e5_embedder` | Transformers + PyTorch | Manual opt-in |
| `load_indic_tts` | Transformers + PyTorch | Manual opt-in, benchmark first |
| Silero VAD | `silero-vad` pip package | ONNX/JIT weights shipped by package |

There is **no model-id string replacement** anywhere in the loader path. Each entry
must name a method that exists on `ModelManager` (enforced by a test).

---

## 4. Commands

```powershell
# Show environment, hardware, model presence
.\venv\Scripts\python.exe doctor.py

# Download only the models needed for the default local stack
.\venv\Scripts\python.exe download_models.py --only whisper-tiny-local ecapa-voxceleb-speaker qwen3-0.6b-instruct

# Download everything (large; not recommended on 8 GB)
.\venv\Scripts\python.exe download_models.py --all

# Measure real load/generation times on this machine
.\venv\Scripts\python.exe benchmark_models.py --vad --asr --speaker
.\venv\Scripts\python.exe benchmark_models.py --slm
```

Downloaded snapshots live under `models/<org>__<repo>/` and are pinned to the
registry revision; re-running the downloader is safe and resumable.

---

## 5. Model policy

- Only **one heavy model** (estimated RAM ≥ 500 MB) may be resident.
- Heavy models are unloaded after `MODEL_UNLOAD_IDLE_SECONDS` (default 300 s).
- Missing models are **not** silently replaced by a different model.
- Under `LOCAL_ONLY=true`, remote models/providers are not called even if keys exist.
- If ECAPA cannot load while an ECAPA profile is enrolled, voice auth fails closed and
  the owner must use the recovery PIN (`python main.py --auth-pin`).

See `docs/TEST_RESULTS.md` for current measured numbers and
`docs/HARDWARE_COMPATIBILITY.md` for machine requirements.
