#!/usr/bin/env python
"""
download_models.py - fetch only registry-approved models from Hugging Face.

Usage:
  python download_models.py                    # default: every enabled model
  python download_models.py --selected-only    # startup/on-demand critical set (vad, whisper-tiny, ecapa)
  python download_models.py --only mid1 mid2   # specific registry model_ids
  python download_models.py --check            # no download, just status

Downloads are gated by:
 - disk headroom (2x expected size)
 - verified model_revision pinned in models/model_registry.yaml
 - ignore *.bin/*.h5/*msgpack patterns (prefer safetensors)
 - failure cooldown (5 min) so an offline laptop never retry-spams the network
"""

import os
import sys
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SELECTED_ONLY: set[str] = {
    "whisper-tiny-local",
    "ecapa-voxceleb-speaker",
}


def status_line(state: str) -> str:
    return {
        "missing": "not downloaded", "downloading": "downloading", "loading": "loading",
        "ready": "ready", "unavailable": "unavailable", "corrupted": "corrupted", "disabled": "disabled",
    }.get(state, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-only", action="store_true", help="download only core startup set (whisper-tiny, ecapa, silero-vad)")
    parser.add_argument("--check", action="store_true", help="report status only, no downloads")
    parser.add_argument("--only", nargs="*", default=None, help="specific model ids from the registry")
    args = parser.parse_args()

    from core.models.model_manager import get_model_manager
    mm = get_model_manager()
    targets = []
    if args.only:
        targets = args.only
    elif args.selected_only:
        targets = [m for m in SELECTED_ONLY if m in mm.entries]
    else:
        targets = [m for m in mm.entries if mm.entry(m).get("enabled")]

    print("=" * 62)
    print(" JARVIS Model Downloader (Hugging Face, registry-gated)")
    print("=" * 62)
    print(" policy:", mm.policy)

    ok = removed_skipped = failed = 0
    for mid in targets:
        e = mm.entry(mid)
        state = mm.statuses().get(mid, "unknown")
        print(f"  {mid}: {status_line(state)} | disk~{e.get('expected_disk_size_mb')}MB | "
              f"ram~{e.get('estimated_ram_mb')}MB | rev {e.get('model_revision')}")
        if args.check:
            continue
        if e.get("enabled") is False:
            print("    -> skipped (disabled in registry)")
            removed_skipped += 1
            continue
        if state == "ready":
            print("    -> skipped (already downloaded)")
            ok += 1
            continue
        t0 = time.perf_counter()
        res = mm.download(mid)
        if res:
            ms = int((time.perf_counter() - t0) * 1000)
            print(f"    -> OK at {res} ({ms} ms)")
            ok += 1
        else:
            print("    -> FAILED (disk/network gates; see logs)")
            failed += 1

    print("-" * 62)
    if not args.check:
        print(f" done: {ok} ok, {failed} failed, {removed_skipped} skipped")
    else:
        print(f" status check complete: {len(targets)} entries")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
