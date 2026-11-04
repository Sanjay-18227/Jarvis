"""
Owner Recovery Authentication (fail-safe when the voice biometric model is
unavailable or borderline). Owner presents a PIN they enrolled locally.
The PIN is stored ONLY as salted SHA-256 hash in data/owner_recovery.json.
Never spoken aloud by JARVIS; recommended over typed console (--auth-pin),
voice phrase fallback is the last resort because STT sees the digits.

Owner-controlled: only the owner can enroll (python main.py --set-pin).
"""

import os
import json
import hashlib
import hmac
import binascii
import secrets
import argparse
import sys
import getpass
from typing import Optional, Tuple

_RECOVERY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "owner_recovery.json")
_MIN_PIN_LEN = 4
_MAX_ATTEMPTS_BEFORE_LOCKOUT = 5
_LOCKOUT_SECONDS = 300.0
_attempts: list = []


def _hash_pin(pin: str, salt: bytes) -> str:
    # PBKDF2-style iterated scrypt-ish: use sha256_100k for stdlib-only portability
    data = salt + pin.encode("utf-8")
    for _ in range(100_000):
        data = hashlib.sha256(data).digest()
    return binascii.hexlify(data).decode()


def recovery_path() -> str:
    return _RECOVERY_PATH


def has_recovery_pin() -> bool:
    return os.path.exists(_RECOVERY_PATH)


def set_owner_pin(pin: str) -> bool:
    if not pin or not pin.strip().isdigit() or len(pin.strip()) < _MIN_PIN_LEN:
        print(f"[RECOVERY] <= '{_MIN_PIN_LEN}' digit numeric PIN required; pin not stored")
        return False
    os.makedirs(os.path.dirname(_RECOVERY_PATH), exist_ok=True)
    salt = secrets.token_bytes(16)
    payload = {"salt": salt.hex(), "pin": _hash_pin(pin.strip(), salt)}
    try:
        with open(_RECOVERY_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.chmod(_RECOVERY_PATH, 0o600)
        print("[RECOVERY] owner recovery PIN stored (salted hash only)")
        return True
    except Exception as e:
        print(f"[RECOVERY] PIN store failed: {e}")
        return False


def verify_owner_pin(pin: str) -> Tuple[bool, str]:
    """Returns (ok, message). Applies fail-closed lockout."""
    global _attempts
    # Lockout window
    now = __import__("time").time()
    _attempts = [t for t in _attempts if now - t < _LOCKOUT_SECONDS]
    if len(_attempts) >= _MAX_ATTEMPTS_BEFORE_LOCKOUT:
        return False, "locked"
    if not has_recovery_pin():
        return False, "not_enrolled"
    try:
        with open(_RECOVERY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        salt = binascii.unhexlify(data["salt"])
        if hmac.compare_digest(_hash_pin(pin.strip(), salt), data["pin"]):
            _attempts.clear()
            return True, "ok"
        _attempts.append(now)
        return False, "denied"
    except Exception as e:
        return False, f"error: {e}"


def authenticate_with_pin(pin: str) -> bool:
    """Convenience: verifies PIN and, on success, opens the owner speaker session."""
    ok, msg = verify_owner_pin(pin)
    if ok:
        try:
            from security.authorization_gate import get_speaker_session
            get_speaker_session().authenticate_owner()
        except Exception:
            pass
    return ok


def _interactive_set_pin():
    print("JARVIS Owner Recovery PIN enrollment (local only; 4-12 digits)")
    for attempt in range(3):
        p1 = getpass.getpass(" new pin: ").strip()
        p2 = getpass.getpass(" confirm : ").strip()
        if p1 != p2:
            print("  PINs do not match")
            continue
        if set_owner_pin(p1):
            sys.exit(0)
    sys.exit(1)


def _interactive_auth():
    if not has_recovery_pin():
        print("No recovery PIN enrolled. Owner can enroll: python main.py --set-pin")
        sys.exit(1)
    pin = getpass.getpass(" owner pin: ").strip()
    ok, msg = verify_owner_pin(pin)
    if ok:
        print("[SECURITY] owner session authenticated via recovery PIN")
        print("NOTE: this verifies the PIN in a separate process. To unlock the live")
        print("      voice session, start JARVIS and say 'pin <digits>' (or re-enroll")
        print("      the voice profile).")
        sys.exit(0)
    print(f"[SECURITY] PIN failed ({msg})")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", action="store_true", help="enroll recovery PIN")
    args = parser.parse_args()
    if args.set:
        _interactive_set_pin()
    else:
        _interactive_auth()
