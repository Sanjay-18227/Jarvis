"""
Optional local screen-vision provider using HuggingFaceTB/SmolVLM-256M-Instruct.
CPU-only inference (no CUDA on target laptop). DISABLED by default because
the model + vision encoder needs ~900 MB RAM; users must opt in:

    python download_models.py --only smolvlm-256m-instruct
    # then set in .env:  VISION_PROVIDER=local    (and registry enabled: true)

LIMITATIONS (documented): English-oriented; Tamil UI text recognition is not
guaranteed. Local VLM output is treated as an observation, never as commands.
"""

import io
import logging
from typing import Optional, Dict, Any

log = logging.getLogger("jarvis.vision.local")

_MODEL_ID = "smolvlm-256m-instruct"


class LocalSmolVLM:
    """Lazy wrapper around transformers + PyTorch for SmolVLM-256M."""

    def __init__(self):
        self._runtime = None
        self._failed = False

    def available(self) -> bool:
        if self._runtime is not None:
            return True
        if self._failed:
            return False
        try:
            from core.models.model_manager import get_model_manager
            mm = get_model_manager()
            entry = mm.entry(_MODEL_ID)
            if not entry or entry.get("enabled") is False:
                return False
            if not mm.is_downloaded(_MODEL_ID):
                log.info("[VISION] local VLM not downloaded; run download_models.py --only smolvlm-256m-instruct")
                return False
            rt = mm.get_runtime(_MODEL_ID)
            if rt is None:
                return False
            self._runtime = rt
            return True
        except Exception as e:
            log.warning("[VISION] local VLM unavailable: %s", e)
            self._failed = True
            return False

    def describe(self, image, query: str) -> Optional[str]:
        """Returns a natural-language observation or None (fail-safe)."""
        if not self.available():
            return None
        try:
            import torch
            rt = self._runtime
            processor = rt["processor"]
            model = rt["model"]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": (
                        f"Answer briefly for a Windows assistant. Question: {query}. "
                        "Describe only what is visible. If unsure, say so.")},
                ],
            }]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
            text = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            return text.strip() or None
        except Exception as e:
            log.error("[VISION] local VLM inference failed: %s", e)
            return None


_provider: LocalSmolVLM | None = None


def get_local_vlm() -> LocalSmolVLM:
    global _provider
    if _provider is None:
        _provider = LocalSmolVLM()
    return _provider
