# Security Model

JARVIS is voice-first, so the security boundary starts at **who is speaking**
and ends at **what the computer is allowed to do**. This document describes the
enforced controls and their limits.

---

## 1. Trust boundaries

| Asset | Boundary |
|---|---|
| Microphone audio | Local process only; no upload with `LOCAL_ONLY=true` |
| Owner biometrics | Local files under `data/speaker_profiles/`; never logged |
| Recovery PIN | Local salted hash; never logged or spoken by JARVIS |
| Cloud reasoning | Disabled unless `ALLOW_REMOTE_LLM=true` |
| Cloud vision | Disabled unless `ALLOW_REMOTE_VISION=true` |
| Remote STT | Only if `STT_ENGINE=groq_whisper`/`google` (default is local) |
| Computer actions | Risk-classified, confirmation-gated, audited |

`LOCAL_ONLY=true` is the default. The provider factory returns the offline
provider before any cloud call is attempted; the vision service uses the same
policy before sending a screenshot.

---

## 2. Speaker classes

| Class | How it is produced | Privileges |
|---|---|---|
| `OWNER` | Similarity ≥ calibrated owner threshold | Allowed, subject to action risk |
| `BORDERLINE` | Score between uncertain and owner thresholds | Limited; high-risk actions require fresh owner confirmation |
| `UNKNOWN` | Score below uncertain threshold, missing profile, or failed verification | **Zero computer control** |

A failed verification invalidates any previous owner session. The owner session
has a short TTL (30 s) and is re-checked for privileged actions.

### Fail-closed ECAPA rule

If an ECAPA profile is enrolled and the ECAPA model cannot run (missing files,
load failure, or < 250 MB free RAM):

- verification returns `None` / `UNKNOWN`;
- JARVIS does **not** silently fall back to the weaker MFCC space;
- the owner must use the local recovery PIN.

Normal RAM pressure does not disable ECAPA; it is only denied below a critical
threshold where reliable inference is impossible.

---

## 3. Action risk levels

| Level | Examples | Default handling |
|---|---|---|
| 0 — Informational | questions, screen description | Allowed for owner |
| 1 — Safe | open calculator, volume, read battery | Allowed for owner |
| 2 — Moderate | create file, edit code, normal download | Allowed with normal checks |
| 3 — Sensitive | send email, upload personal file, account settings | Explicit owner confirmation |
| 4 — Critical | delete data, credentials, financial/destructive commands | Explicit owner confirmation; strongest checks |

High-risk confirmation is single-use and ephemeral (`security/confirmation.py`).
UNKNOWN speakers cannot trigger any computer action, including level 1.

---

## 4. Confirmation and audit

- Confirmation prompts are bound to the specific action and expire.
- The audit log (`logs/jarvis_audit.jsonl`) records action, risk, status and
  evidence with secrets redacted.
- Embedding vectors, PINs, passwords, API keys and tokens are never written to
  the audit log.
- `security/privacy_guard.py` blocks/redacts credential-like strings and
  private-data requests from non-owner speakers.

---

## 5. Secrets handling

- `.env` contains provider keys and must never be committed (`.gitignore`).
- `Settings.sanitized_dict()` masks key/secret/token fields for logging.
- Memory (`memory/store.py`) rejects content that looks like passwords, API
  keys, tokens or card numbers; it is not a secrets vault.
- The recovery PIN is stored as a **salted, 100,000-round iterated SHA-256**
  hash in `data/owner_recovery.json`; comparison uses `hmac.compare_digest`.
- Five failed PIN attempts trigger a 300-second lockout. A successful attempt
  clears the failure window.

> The PIN hash is iterated SHA-256 for stdlib portability. It is not PBKDF2,
> scrypt or Argon2; the PIN is a local fallback, not a replacement for a strong
> biometric profile.

---

## 6. Local-only enforcement

| Control | Enforced in |
|---|---|
| Cloud reasoning blocked | `llm/provider_factory.py` returns offline provider when `LOCAL_ONLY=true` and `ALLOW_REMOTE_LLM=false` |
| Remote screenshots blocked | `vision/screen_vision_service.py` checks `LOCAL_ONLY` / `ALLOW_REMOTE_VISION` |
| Cloud voice transport blocked | `voice/voice_engine.py` forces local microphone transport when `LOCAL_ONLY=true` |
| Local STT fail-fast | `voice/stt.py` with `STT_ENGINE=local` never calls a remote fallback |
| No hidden model substitution | `models/model_registry.yaml` pins revisions and explicit loaders |
| No cloud memory | Memory and jobs are SQLite files on this laptop |

Opting in is deliberate and visible:

```ini
LOCAL_ONLY=false
# or keep LOCAL_ONLY=true and set:
ALLOW_REMOTE_LLM=true
ALLOW_REMOTE_VISION=true
```

With `ALLOW_REMOTE_LLM=true`, the configured provider key is used; with no key,
JARVIS stays in offline mode.

---

## 7. Biometric and privacy rules

- Enrollment audio is temporary and deleted by default.
- Profiles are stored only on this machine; never uploaded.
- Custom (test) profiles keep ECAPA data in the test directory, so tests cannot
  overwrite the production biometric profile.
- Do not commit or share:
  - `data/speaker_profiles/**`
  - `data/owner_recovery.json`
  - `memory/jarvis.db`
  - `data/jobs.db`
  - `.env`, `logs/`

---

## 8. Threat model (honest limits)

**Covered:**

- Another person speaking to JARVIS → `UNKNOWN`, no computer control.
- Replay of a recorded owner phrase → ECAPA embedding comparison makes this
  harder than simple MFCC, but **replay is not cryptographically prevented**;
  do not rely on voice alone for high-value secrets.
- Accidental credential storage → memory rejection + audit redaction.
- Cloud leakage → local-only defaults and explicit opt-ins.
- Model substitution → pinned revisions + explicit loader methods.

**Not covered / limitations:**

- A determined attacker with a high-quality voice clone may pass speaker
  verification. High-risk actions still require confirmation, but this is not
  a replacement for OS-level authentication.
- The recovery PIN is only as strong as its length and the local machine's
  security. Use 6+ digits and keep the machine locked.
- Malware running as the same Windows user can read local profiles and memory;
  OS account security is the outer boundary.
- Background jobs are not encrypted at rest beyond the OS user boundary.
- `--auth-pin` verifies the PIN in a separate process; it does not unlock an
  already-running JARVIS process. Use the spoken `pin <digits>` fallback in the
  live session.

---

## 9. Security checklist

- [ ] `LOCAL_ONLY=true`, `ALLOW_REMOTE_LLM=false`, `ALLOW_REMOTE_VISION=false`
- [ ] Real owner re-enrolled (not `--synth`)
- [ ] Recovery PIN set with 6+ digits
- [ ] `.env` not committed; keys rotated if exposed
- [ ] `data/speaker_profiles/` backups kept private
- [ ] Second-speaker test → `UNKNOWN`
- [ ] ECAPA-unavailable test → fail closed + PIN recovery
- [ ] Audit log inspected for unexpected actions
