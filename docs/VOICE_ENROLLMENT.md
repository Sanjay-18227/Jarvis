# Owner Voice Enrollment

This document describes how Sanjay enrolls his voice, what is stored, how
verification behaves, and how to recover access if the biometric model is
unavailable.

---

## 1. What enrollment creates

Enrollment captures multiple clean recordings and writes **two local profiles**:

| File | Contents | Used when |
|---|---|---|
| `data/speaker_profiles/owner_profile.dat` | 42-d MFCC centroid + representative embeddings, calibrated threshold | legacy fallback / no ECAPA profile |
| `data/speaker_profiles/owner_profile_ecapa.json` | 192-d ECAPA centroid + representatives | primary biometric gate |
| `data/speaker_profiles/backups/owner_profile_backup_*.dat` | timestamped backup before replacement | rollback on failed enrollment |

Nothing is uploaded. Raw temporary audio is deleted unless
`OWNER_VOICE_KEEP_SAMPLES=true` is set for debugging.

---

## 2. Before you enroll

1. Use the same microphone JARVIS uses.
2. Sit in a quiet room; avoid fans, music and notifications.
3. Speak naturally at a normal volume, 1–3 seconds per prompt.
4. Make sure the ECAPA model is present:
   ```powershell
   .\venv\Scripts\python.exe download_models.py --only ecapa-voxceleb-speaker
   ```
5. Optional microphone check:
   ```powershell
   .\venv\Scripts\python.exe main.py --mic-test
   # or
   .\venv\Scripts\python.exe enroll_owner.py --mic-test
   ```

---

## 3. Enroll (recommended)

```powershell
.\venv\Scripts\python.exe enroll_owner.py --re-enroll --name Sanjay
```

- Target: 10 samples; minimum: 8 accepted samples.
- Each sample is quality-checked: duration, clipping, energy, speech ratio.
- Inconsistent outliers are filtered by pairwise cosine similarity.
- The MFCC profile is saved first, then the ECAPA profile is built from the
  same samples.
- ECAPA embeddings trim leading/trailing silence and keep the strongest natural
  speech cluster, so idle room noise cannot dilute the voiceprint.
- Near-orthogonal ECAPA outliers (partial or noisy recordings) are removed
  before the centroid is computed.
- ECAPA uses its **own calibrated threshold** derived from same-owner scores;
  the 0.78 MFCC threshold is never applied to the ECAPA scale. The ECAPA floor
  is 0.60 and re-enrollment can only tighten it, never lower it. Calibration is
  capped at 0.72 because enrollment self-scores are optimistically biased;
  genuine held-out owner speech typically scores 0.77–0.85.
- The previous `.dat` and ECAPA profiles are backed up before replacement; if
  enrollment fails, the old profiles are preserved.

Other entry points:

```powershell
.\venv\Scripts\python.exe main.py --enroll          # launch enrollment from main
.\venv\Scripts\python.exe enroll_owner.py           # standard enrollment
.\venv\Scripts\python.exe enroll_owner.py --test-verification
```

> `--synth` generates samples with Edge TTS. It is only for automated tests and
> writes to `data/speaker_profiles/synthetic_test/`; it cannot overwrite the real
> owner profile or ECAPA biometrics.

---

## 4. Verify the result

```powershell
.\venv\Scripts\python.exe enroll_owner.py --test-verification
```

Expected output: `[SECURITY] Owner session authenticated`.

For a read-only live check that prints both MFCC and ECAPA scores without
changing the profile:

```powershell
.\venv\Scripts\python.exe main.py --speaker-test
```

Then start JARVIS and speak a command:

```powershell
.\venv\Scripts\python.exe main.py
```

If verification returns `UNKNOWN` for the real owner, do **not** lower the
threshold. Re-run `--speaker-test`, move closer to the microphone, and re-enroll
with cleaner samples, or use the recovery PIN.

---

## 5. How verification decides

| Result | Meaning | Computer control |
|---|---|---|
| `OWNER` | Score ≥ calibrated owner threshold | Allowed, subject to risk/confirmation rules |
| `BORDERLINE` | Score between uncertain and owner thresholds | Limited; high-risk actions require fresh owner confirmation |
| `UNKNOWN` | Score below uncertain threshold, or no profile | **Zero computer control** |

Rules that are enforced in code:

- Thresholds are **never lowered automatically**. Calibration only proposes a
  threshold within the allowed band; the owner can raise it in `.env`
  (`OWNER_SIMILARITY_THRESHOLD`, `UNCERTAIN_SIMILARITY_THRESHOLD`).
- ECAPA and MFCC are separate similarity scales. ECAPA uses
  `OWNER_ECAPA_THRESHOLD` (floor 0.60, calibrated at enrollment), while MFCC
  uses `OWNER_SIMILARITY_THRESHOLD`. Applying the MFCC threshold to ECAPA is a
  bug and is guarded by tests.
- Live voice mode calibrates the ambient microphone threshold at startup.
  Otherwise fan/keyboard noise can create 30–60 s turns dominated by silence,
  which collapses ECAPA owner scores.
- If an ECAPA profile exists but the ECAPA model cannot load (missing files or
  < 250 MB free RAM), verification **fails closed**. It does not silently fall
  back to the weaker MFCC space.
- A failed or missing profile means `UNKNOWN`, never `OWNER`.
- Custom/isolated profiles (used by tests) keep their ECAPA data next to the
  custom profile, so tests never overwrite the real owner's biometrics.

---

## 6. Re-enrollment and rollback

- Every enrollment backs up the previous `.dat` profile under
  `data/speaker_profiles/backups/` and the previous ECAPA profile as
  `owner_profile_ecapa_backup_*.json`.
- A failed enrollment restores/preserves the previous `.dat` and keeps the old
  ECAPA profile.
- To re-enroll, run `enroll_owner.py --re-enroll`; no need to delete files.
- Keep backups private; they are biometric data.
- To deliberately remove biometrics, delete both profile files and re-enroll.
  JARVIS will then treat all speakers as `UNKNOWN` until a profile exists.

---

## 7. Recovery PIN (when voice cannot authenticate)

The PIN is a local, owner-controlled fallback for cases such as:

- ECAPA model unavailable or RAM too low (fail-closed path)
- persistent microphone failure
- voice lost/hoarse, or enrollment no longer matches

### Set the PIN (owner, once)

```powershell
.\venv\Scripts\python.exe main.py --set-pin
```

- 4–12 digits.
- Stored only as a salted, iterated SHA-256 hash in `data/owner_recovery.json`.
- Never spoken or logged.

### Authenticate with the PIN

```powershell
.\venv\Scripts\python.exe main.py --auth-pin
```

Typing is preferred. The voice fallback (`"pin 123456"` spoken) exists as a
last resort, but STT sees the digits; use it only when necessary.

Security controls:

- 5 failed attempts trigger a 300-second lockout.
- PIN verification fails closed if no PIN is enrolled.
- The PIN file is created with owner-only permissions where supported.

---

## 8. Privacy and sensitive-data rules

- Biometric vectors are stored locally only.
- Raw enrollment audio is temporary and deleted by default.
- The audit log records verification outcomes, never embedding vectors or PINs.
- `owner_profile*.dat`, `owner_profile_ecapa.json`, `owner_recovery.json` and
  `data/speaker_profiles/backups/` must never be committed to git or shared.

---

## 9. Manual tests (must be performed by the owner)

These cannot be automated honestly:

- [ ] Enroll with the real voice and confirm `OWNER` on 3 different phrases.
- [ ] Have a second person speak; confirm `UNKNOWN` and zero computer control.
- [ ] Temporarily rename the ECAPA model directory; confirm fail-closed and PIN
      recovery works.
- [ ] Test Tamil and Tamil-English code-mixed commands with real speech.
- [ ] Test in a noisy room; confirm noise does not authenticate as owner.
