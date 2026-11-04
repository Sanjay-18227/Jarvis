"""
microphone diagnostics module.
python main.py --mic-test   ->   measures input device, sample rate, channels,
noise floor, speech RMS, peak, and suggests recorder VAD thresholds.
"""

import time
import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None


def _collect(duration_s: float) -> np.ndarray:
    rate = 16000
    print(f"   [MIC] listening {duration_s:.1f}s (whisper: measure NOISE; then talk: measure SPEECH)...")
    rec = sd.rec(int(rate * duration_s), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    return rec.flatten().astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def run_mic_test(sample_rate: int = 16000):
    if not sd:
        print("[MIC-TEST] sounddevice not installed")
        return 1
    print("=" * 55)
    print(" JARVIS Microphone Test")
    print("=" * 55)
    try:
        default_in = sd.default.device[0]
        info = sd.query_devices(default_in)
        print(f" device      : {info['name']}")
        print(f" in-channels : {info['max_input_channels']}")
        print(f" sample rate : default {info['default_samplerate']:.0f} Hz (test uses {sample_rate})")
    except Exception as e:
        print(f"[MIC-TEST] device enumeration failed: {e}")
        return 1

    # noise window ~3s
    noise = _collect(3.0)
    nf = _rms(noise)
    peak_n = float(np.max(np.abs(noise))) if len(noise) else 0.0
    print(f" noise floor  : rms={nf:.0f} peak={peak_n:.0f}")
    print(" Say a short word like 'JARVIS' now...")
    speech = _collect(5.0)
    sr = _rms(speech)
    peak_s = float(np.max(np.abs(speech))) if len(speech) else 0.0
    print(f" speech RMS   : rms={sr:.0f} peak={peak_s:.0f}")
    if sr < 200:
        print(" [WARN] very low speech RMS - microphone may be muted")
    elif sr < 400:
        print(" [WARN] speech not clearly louder than noise; try moving closer")
    else:
        print(" [OK] speech detected clearly")
    # Suggested thresholds follow the adaptive strategy used by recorder.py
    suggested_min = max(nf * 1.8, 250)
    suggested_max = min(max(nf * 3.5, 3200), 3200)
    suggested_speech_min = max(nf * 2.2, 300)
    print(f" suggested VAD thresholds:")
    print(f"   energy_threshold        = int(max(nf*1.8, 250)) -> {int(suggested_min)}")
    print(f"   vad_min_threshold       = {int(suggested_min)} (for CONFIG)")
    print(f"   vad_max_threshold       = {int(suggested_max)}")
    print(f"   min speech rms accepted = {int(suggested_speech_min)}")
    snr_like = 0.0 if nf <= 0 else sr / nf
    print(f" rough speech/noise ratio : {snr_like:.2f} (expect > 3)")
    return 0
