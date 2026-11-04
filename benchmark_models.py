#!/usr/bin/env python
"""
benchmark_models.py - measure ACTUAL hardware performance (never invent numbers).

 Measures:
  1. SLM   qwen3-0.6b    (load time, tok/s, resident RAM)   on demand: --slm
  2. ASR   whisper-tiny / base  (load time, RTF on a synthetic tone file, RAM)  --asr
  3. VAD   silero-vad    (chunk latency)                    --vad
  4. Speaker ECAPA        (embedding latency)                --speaker

 Usage:
  python benchmark_models.py --all            (may download missing models first)
  python benchmark_models.py --asr            (ASR only)
 Results are printed and stored in docs/BENCHMARK_RESULTS.md template line format
 so TEST_RESULTS.md keeps real numbers only.
"""

import os
import sys
import time
import json
import argparse
import statistics

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _ram_mb():
    import psutil
    return psutil.Process().memory_info().rss // (2 ** 20)


def fmt_peak(before, after, delta_hint=""):
    return f"peak RAM +{after - before} MB {delta_hint}"


def bench_slm(mm, out):
    e = mm.entry("qwen3-0.6b-instruct")
    if not e or e.get("enabled") is False:
        out["slm"] = {"status": "skipped (disabled by registry/policy)"}; return
    t0 = time.perf_counter(); before = _ram_mb()
    mm.unload("qwen3-0.6b-instruct"); rt = mm.get_runtime("qwen3-0.6b-instruct")
    if not rt:
        out["slm"] = {"status": "unavailable (not downloaded / policy denied)"}; return
    load_ms = int((time.perf_counter() - t0) * 1000)
    after = _ram_mb()
    tokencounts = []
    prompts = ["/listen hi jarvis what time is it",
               "/listen open folder Kumar in vscode",
               "Convert to JSON intent: tell the team meeting at 8pm"]
    gen_ms, gen_tokens = 0.0, 0
    for text in prompts:
        t0 = time.perf_counter()
        reply = mm.generate_text("qwen3-0.6b-instruct", text, max_new_tokens=64,
                                 enforce_non_thinking=True)
        dt = time.perf_counter() - t0
        out.setdefault("slm_samples", []).append({"prompt": text[:38], "reply": reply[:60], "seconds": round(dt, 3)})
        n = max(len(reply.split()), 1)
        gen_ms += dt * 1000; gen_tokens += n
    out["slm"] = {
        "status": "ok", "load_ms": load_ms, "resident_ram_mb": int(after - before),
        "avg_seconds_per_sample": round(gen_ms / len(prompts) / 1000.0, 3),
        "approx_words_per_s": round(gen_tokens / (gen_ms / 1000.0), 1),
    }
    mm.unload("qwen3-0.6b-instruct")


def bench_asr(mm, out):
    import numpy as np
    # make a small spoken-like tone demo; real voices verified separately (LIMITATION)
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
    audio = (0.35 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    audio += (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    results = {}
    for mid in ("whisper-tiny-local", "whisper-base-local"):
        e = mm.entry(mid)
        if not e or e.get("enabled") is False:
            results[mid] = {"status": "disabled"}; continue
        before = _ram_mb(); t0 = time.perf_counter()
        rt = mm.get_runtime(mid)
        load_ms = int((time.perf_counter() - t0) * 1000)
        if not rt:
            results[mid] = {"status": "unavailable"}; continue
        after = _ram_mb()
        run_ms = []
        for _ in range(2):
            t0 = time.perf_counter()
            segments, info = rt.transcribe(audio, language="en")
            list(segments)  # pull generator
            run_ms.append((time.perf_counter() - t0) * 1000)
        results[mid] = {
            "status": "ok", "load_ms": load_ms, "resident_ram_mb": int(after - before),
            "transcribe_ms_audio2s": round(statistics.mean(run_ms), 1),
            "estimated_rtf_on_speech": "needs real voice recording (MANUAL TEST)",
        }
        mm.unload(mid)
    out["asr"] = results


def bench_vad(mm, out):
    try:
        from silero_vad import load_silero_vad
        import numpy as np, torch
        t0 = time.perf_counter()
        model = load_silero_vad()
        load_ms = int((time.perf_counter() - t0) * 1000)
        chunk = torch.from_numpy(np.zeros((512,), dtype=np.float32))
        lat = []
        for _ in range(50):
            t0 = time.perf_counter()
            model(chunk, sr=16000)
            lat.append((time.perf_counter() - t0) * 1000)
        out["vad"] = {"status": "ok", "load_ms": load_ms,
                      "mean_chunk_ms": round(statistics.mean(lat), 3),
                      "max_chunk_ms": round(max(lat), 3)}
    except Exception as e:
        out["vad"] = {"status": f"error: {e}"}


def bench_speaker(mm, out):
    try:
        import io, wave
        import numpy as np
        # Measure the production path (decode -> resample -> encode -> L2 normalize).
        rng = np.random.RandomState(7)
        pcm = (rng.randn(16000 * 2) * 0.05 * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm.tobytes())
        from voice.ecapa_engine import get_ecapa_engine
        engine = get_ecapa_engine()
        t0 = time.perf_counter()
        vec = engine.embedding(buf.getvalue())
        dt = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        engine.embedding(buf.getvalue())
        warm = (time.perf_counter() - t0) * 1000
        if vec is None:
            out["speaker"] = {"status": "error: engine returned no embedding"}
            return
        # Embeddings are biometric data: log only aggregate facts, never raw vectors.
        out["speaker"] = {"status": "ok", "ms_first_embedding": round(dt, 1),
                          "ms_warm_embedding": round(warm, 1),
                          "embed_dim": int(vec.shape[0]),
                          "embedding_l2_norm": round(float((vec ** 2).sum() ** 0.5), 4)}
    except Exception as e:
        out["speaker"] = {"status": f"error: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--slm", action="store_true")
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--speaker", action="store_true")
    args = parser.parse_args()
    if not any([args.all, args.slm, args.asr, args.vad, args.speaker]):
        args.all = True

    from core.models.model_manager import get_model_manager
    mm = get_model_manager()
    out = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": {k: v for k, v in mm.policy.items()}}
    if args.all or args.vad:
        bench_vad(mm, out)
    if args.all or args.asr:
        bench_asr(mm, out)
    if args.all or args.speaker:
        bench_speaker(mm, out)
    if args.all or args.slm:
        bench_slm(mm, out)

    print(json.dumps(out, indent=2, default=str))
    with open(os.path.join("logs", "benchmark_latest.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\nSaved -> logs/benchmark_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
