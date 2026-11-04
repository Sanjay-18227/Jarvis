"""
Model Manager Subsystem
Single owner of the Hugging Face model set. Loads registry, gates models by
hardware policy, downloads lazily with disk checks + resume, loads/unloads
in-memory models with bound concurrency (only ONE heavy model resident).
"""

import os
import shutil
import logging
import threading
import time
import importlib

log = logging.getLogger("jarvis.model_manager")

REGISTRY_PATH = os.path.join("models", "model_registry.yaml")

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class ModelManager:
    """
    Resource-aware model lifecycle controller.

    Contract:
      get_runtime(entry_id) -> object | None    (loaded inference wrapper or None if policy denies / not downloaded)
      download(entry_id) -> local path or None  (safe download with disk + resume)
      unload(entry_id) -> frees memory, respects concurrency
      statuses() -> {entry_id: state}

    States: missing | downloading | loading | ready | disabled | corrupted

    Loader binding is EXPLICIT: each registry entry carries `loader_method`
    (a method name on this class). No fragile model-id string mangling.
    """

    STATES = {"missing", "downloading", "loading", "ready", "unavailable", "corrupted", "disabled"}

    def __init__(self, settings=None):
        from config.settings import get_settings
        self.settings = settings or get_settings()
        self.registry_path = os.path.join(self.settings.base_dir, REGISTRY_PATH)
        self.cache_root = self.settings.model_dir or os.path.join(self.settings.base_dir, "models")
        os.makedirs(self.cache_root, exist_ok=True)
        self.entries: dict[str, dict] = {}
        self._state: dict[str, str] = {}
        self._loaded: dict[str, object] = {}
        self._last_used: dict[str, float] = {}
        self._lock = threading.RLock()
        if yaml:
            self._load_registry()
        # hardware + policy
        try:
            from core.system.hardware_detector import detect_hardware, model_policy
            self.hardware = detect_hardware([self.cache_root])
            self.policy = model_policy(self.hardware)
        except Exception as e:
            log.error("[MODEL] hardware detect failed: %s", e)
            self.hardware = {"available_ram_mb": 0, "torch_device": "cpu"}
            self.policy = {"torch_device": "cpu", "threads": 2, "allow_slm": False,
                           "allow_whisper": False, "allow_vlm": False, "allow_ecapa": False,
                           "reason": "hardware detect failed"}
        # network avoid hammering: last failed download timestamps
        self._failures: dict[str, float] = {}

    # ------------------------------------------------------------------ registry
    def _load_registry(self):
        with open(self.registry_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for entry in data.get("models", []):
            mid = entry.get("model_id")
            if not mid:
                continue
            self.entries[mid] = dict(entry)
            if entry.get("enabled") is False:
                self._state[mid] = "disabled"
            else:
                self._state[mid] = "missing" if not self.is_downloaded(mid) else "ready"
        log.info("[MODEL] registry: %d entries enabled: %s",
                 len(self.entries), [m for m, e in self.entries.items() if e.get("enabled")])

    def entry(self, model_id: str) -> dict | None:
        return self.entries.get(model_id)

    # ------------------------------------------------------------------ disk gating
    def _common_cache_dir(self, producer: str) -> str:
        # huggingface stores snapshots under models/<basename>/...
        return os.path.join(self.cache_root, producer.replace("/", "__"))

    def is_downloaded(self, model_id: str) -> bool:
        """Model presence check per backend / cache layout."""
        e = self.entry(model_id)
        if not e:
            return False
        if e["task"] == "voice_activity_detection":
            # silero-vad ships onnx + jit weights when the pip package installed them
            try:
                import silero_vad  # noqa: F401
                return True
            except ImportError:
                return False
        repo = e.get("hf_repo") or ""
        base = os.path.join(self.cache_root, repo.replace("/", "__"))
        # A downloaded snapshot must contain a config/model marker
        for marker in ("config.json", "engine.pt", "*.ckpt", "*.ct2"):
            path = marker if "*" not in marker else None
            if path and os.path.exists(os.path.join(base, path)):
                return True
        # speechbrain/hf variants craft slightly different layouts; presence of any dir suffices
        if os.path.isdir(base) and os.listdir(base):
            return True
        return False

    def _disk_guard(self, needed_mb: int) -> bool:
        try:
            free_bytes = shutil.disk_usage(self.cache_root).free
            return free_bytes > needed_mb * (2 ** 20) * 2  # 2x margin
        except Exception:
            return False

    # ------------------------------------------------------------------ download
    def download(self, model_id: str, allow_patterns=None) -> str | None:
        e = self.entry(model_id)
        if not e or not e.get("hf_repo"):
            log.warning("[MODEL] %s: no hf_repo; skip download", model_id)
            return None
        if self._state.get(model_id) == "downloading":
            return None
        # Do not retry within 5 min after a failure (offline loop protection)
        if time.time() - self._failures.get(model_id, 0) < 300:
            log.warning("[MODEL] %s: skipping download (recent failure, likely offline)", model_id)
            return None
        disk_mb = int(e.get("expected_disk_size_mb", 1000))
        if not self._disk_guard(disk_mb):
            log.error("[MODEL] insufficient disk for %s (%d MB needed)", model_id, disk_mb)
            self._state[model_id] = "unavailable"
            return None
        repo = e["hf_repo"]
        local = self._common_cache_dir(repo)
        ignore = e.get("ignore_patterns", [])
        self._state[model_id] = "downloading"
        log.info("[MODEL] downloading %s (%s rev %s) -> %s", model_id, repo, e.get("model_revision"), local)
        try:
            from huggingface_hub import snapshot_download
            kwargs = dict(
                repo_id=repo,
                revision=str(e.get("model_revision", "main")),
                local_dir=local,
                ignore_patterns=ignore,
            )
            if allow_patterns:
                kwargs["allow_patterns"] = allow_patterns
            snapshot_download(**kwargs)
            self._state[model_id] = "ready"
            log.info("[MODEL] %s downloaded OK", model_id)
            return local
        except Exception as ex:
            self._failures[model_id] = time.time()
            self._state[model_id] = "missing"
            log.error("[MODEL] download failed %s: %s", model_id, ex)
            return None

    # ------------------------------------------------------------------ load/unload
    def _sel_allowed(self, model_id: str) -> str | None:
        e = self.entries.get(model_id)
        if not e:
            return "not registered"
        if e.get("enabled") is False:
            return "disabled in registry"
        allow = self.policy
        if "whisper" in model_id and not allow.get("allow_whisper"):
            return "RAM policy: whisper disabled"
        if "qwen3" in model_id and not allow.get("allow_slm"):
            return "RAM policy: SLM disabled"
        if "smolvlm" in model_id and not allow.get("allow_vlm"):
            return "RAM policy: VLM disabled"
        if ("ecapa" in model_id) and not allow.get("allow_ecapa"):
            return "RAM policy: ECAPA disabled"
        # Only one heavy model at a time
        heavy_in_ram = [m for m in self._loaded
                        if self._heavy(m) and m != model_id]
        if heavy_in_ram and self._heavy(model_id):
            return f"heavy model concurrency: {heavy_in_ram[0]} loaded; unload first"
        return None

    def _heavy(self, model_id: str) -> bool:
        e = self.entries.get(model_id) or {}
        return int(e.get("estimated_ram_mb") or 0) >= 500

    def get_runtime(self, model_id: str):
        """Returns loaded runtime or None. Never instantiates two heavy models."""
        with self._lock:
            if model_id in self._loaded:
                self._last_used[model_id] = time.time()
                return self._loaded[model_id]
            # refresh RAM-driven policy with current memory snapshot
            try:
                from core.system.hardware_detector import detect_hardware, model_policy
                self.hardware = detect_hardware([self.cache_root])
                self.policy = model_policy(self.hardware)
            except Exception:
                pass
            reason = self._sel_allowed(model_id)
            if reason:
                log.info("[MODEL] %s not loaded: %s", model_id, reason)
                self._state[model_id] = "unavailable" if not self.is_downloaded(model_id) else self._state.get(model_id, "missing")
                return None
            if not self.is_downloaded(model_id):
                # try downloading (allowed for networked startup) then still verify
                ok = self.download(model_id)
                if ok is None:
                    return None
            loader = self._loader_for(model_id)
            if not loader:
                self._state[model_id] = "corrupted"
                return None
            self._state[model_id] = "loading"
            t0 = time.perf_counter()
            try:
                runtime = loader(self.entry(model_id))
                self._loaded[model_id] = runtime
                self._last_used[model_id] = time.time()
                ms = int((time.perf_counter() - t0) * 1000)
                print(f"[MODEL] {model_id} loaded in {ms} ms ({self.policy['torch_device']})")
                self._state[model_id] = "ready"
                return runtime
            except Exception as ex:
                log.error("[MODEL] %s load failed: %s", model_id, ex)
                self._state[model_id] = "corrupted"
                self._failures[model_id] = time.time()
                return None

    def unload(self, model_id: str) -> bool:
        """Free loaded runtime. Frees torch cache and gc."""
        with self._lock:
            runtime = self._loaded.pop(model_id, None)
            self._last_used.pop(model_id, None)
            self._state.setdefault(model_id, "missing")
            if runtime:
                try:
                    del runtime
                    import gc
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    print(f"[MODEL] {model_id} unloaded (memory freed)")
                    self._state[model_id] = "missing"
                    return True
                except Exception as ex:
                    log.error("[MODEL] unload failed %s: %s", model_id, ex)
            return False

    def unload_idle(self, idle_seconds=None):
        idle = idle_seconds if idle_seconds is not None else self.settings.model_unload_idle_seconds
        now = time.time()
        for mid in list(self._loaded):
            if not self._heavy(mid):
                continue
            if now - self._last_used.get(mid, 0) > idle:
                self.unload(mid)

    def statuses(self) -> dict:
        return dict(self._state)

    # ------------------------------------------------------------------ loaders
    def _loader_for(self, model_id: str):
        """EXPLICIT binding from the registry `loader_method` field (no string munging)."""
        e = self.entries.get(model_id)
        if not e:
            return None
        method_name = e.get("loader_method")
        if not method_name:
            return None
        loader = getattr(self, method_name, None)
        if loader is None:
            log.error("[MODEL] registry loader_method '%s' for %s does not exist on ModelManager",
                      method_name, model_id)
        return loader

    def load_whisper_tiny(self, entry):
        return self._load_ct2_whisper(entry, "tiny")

    def load_whisper_base(self, entry):
        return self._load_ct2_whisper(entry, "base")

    def _load_ct2_whisper(self, entry, size: str):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return None
        local = self._common_cache_dir(entry["hf_repo"])
        backend = getattr(self.settings, "asr_local_backend", "faster-whisper")
        compute = "int8" if "faster" in backend else "float32"
        model = WhisperModel(local, device="cpu", compute_type=compute,
                             cpu_threads=int(self.policy["threads"]))
        print(f"[MODEL] whisper-{size} CT2 ready (int8, {self.policy['threads']} threads)")
        return model

    def load_ecapa_speaker(self, entry):
        try:
            import torch  # noqa: F401
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception as e:
            log.error("[MODEL] speechbrain import failed: %s", e)
            return None
        local = self._common_cache_dir(entry["hf_repo"])
        cls = EncoderClassifier.from_hparams(
            source=str(local) if os.path.isdir(local) and os.listdir(local) else entry["hf_repo"],
            savedir=local,
        )
        return cls

    def load_smolvlm_vlm(self, entry):
        """Optional SmolVLM-256M loader (heavy, on-demand only)."""
        try:
            import torch
            from transformers import AutoProcessor
        except Exception as e:
            log.error("[MODEL] transformers import failed: %s", e)
            return None
        local = self._common_cache_dir(entry["hf_repo"])
        src = local if os.path.isdir(local) and os.listdir(local) else entry["hf_repo"]
        processor = AutoProcessor.from_pretrained(src)
        model = None
        for cls_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
            try:
                import transformers
                cls = getattr(transformers, cls_name, None)
                if cls is None:
                    continue
                model = cls.from_pretrained(src, torch_dtype=torch.float32, low_cpu_mem_usage=True)
                break
            except Exception as e:
                log.warning("[MODEL] %s load attempt failed: %s", cls_name, e)
        if model is None:
            return None
        model.eval()
        return {"processor": processor, "model": model}

    def load_e5_embedder(self, entry):
        """Optional multilingual-e5-small semantic embedder (manual opt-in)."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            log.error("[MODEL] transformers import failed: %s", e)
            return None
        local = self._common_cache_dir(entry["hf_repo"])
        src = local if os.path.isdir(local) and os.listdir(local) else entry["hf_repo"]
        tok = AutoTokenizer.from_pretrained(src)
        mdl = AutoModel.from_pretrained(src, low_cpu_mem_usage=True).eval()
        return {"tokenizer": tok, "model": mdl, "kind": "e5"}

    def load_indic_tts(self, entry):
        """Optional ai4bharat/indic-parler-tts loader (manual opt-in; benchmark before use)."""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForTextToWaveform
        except Exception as e:
            log.error("[MODEL] indic-parler import failed: %s", e)
            return None
        local = self._common_cache_dir(entry["hf_repo"])
        src = local if os.path.isdir(local) and os.listdir(local) else entry["hf_repo"]
        processor = AutoProcessor.from_pretrained(src)
        model = AutoModelForTextToWaveform.from_pretrained(src, low_cpu_mem_usage=True).eval()
        return {"processor": processor, "model": model}

    def load_qwen3_slm(self, entry):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            log.error("[MODEL] transformers import failed: %s", e)
            return None
        local = self._common_cache_dir(entry["hf_repo"])
        src = local if os.path.isdir(local) and os.listdir(local) else entry["hf_repo"]
        # CPU-first: fp32 is measurably faster than int8 dynamic quantization on this
        # laptop (5.8s vs 13.7s per short generation), at ~2.5 GB RSS. Loaded strictly
        # on demand and unloaded when idle (see ModelManager.evict_idle).
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tok = AutoTokenizer.from_pretrained(src)
        mdl = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=dtype,
            low_cpu_mem_usage=True, trust_remote_code=False,
        )
        mdl.eval()
        torch.set_num_threads(max(2, self.policy["threads"]))
        return {"tokenizer": tok, "model": mdl}

    def generate_text(self, model_id: str, prompt: str, system: str = None,
                      max_new_tokens: int = 128, enforce_non_thinking: bool = True) -> str:
        """Generate text with the local SLM (Qwen3 chat template, CPU-first)."""
        rt = self.get_runtime(model_id)
        if not rt:
            return ""
        import re
        import torch
        tok = rt["tokenizer"]
        model = rt["model"]
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        # Qwen3 supports native non-thinking mode; fall back for older templates.
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enforce_non_thinking:
            kwargs["enable_thinking"] = False
        try:
            text = tok.apply_chat_template(msgs, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            text = tok.apply_chat_template(msgs, **kwargs)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Strip any residual reasoning block even in non-thinking mode.
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
        return reply.strip()


_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


if __name__ == "__main__":
    import json
    mm = get_model_manager()
    print(json.dumps({"statuses": mm.statuses(), "policy": mm.policy}, indent=2, default=str))
