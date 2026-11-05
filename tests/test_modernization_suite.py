"""
Modernization verification suite (Stages 3-10).

Covers:
 - hardware detector output contract + model policy gating
 - model registry integrity (licenses, revisions, explicit loader methods)
 - model manager disk/lifecycle helpers (no forced downloads)
 - ASR vocabulary correction
 - SQLite memory store (remember / search / forget / privacy rejection / temp clear)
 - durable job store (create / recurse / retry)
 - scheduler dispatch through the SkillRegistry
 - owner recovery PIN (hash only, lockout)
 - capability/skill registry integration of memory + scheduler
"""

import os
import sys
import time
import uuid
import tempfile

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------- hardware
def test_hardware_detector_contract():
    from core.system.hardware_detector import detect_hardware, model_policy
    hw = detect_hardware()
    for key in ("cpu_name", "physical_cores", "logical_cores", "total_ram_mb",
                "available_ram_mb", "free_disk_bytes", "cuda_available",
                "torch_device", "python_version", "process_memory_mb"):
        assert key in hw, f"missing hardware field {key}"
    assert hw["physical_cores"] >= 1
    assert hw["total_ram_mb"] > 0
    assert hw["torch_device"] in ("cpu", "cuda")
    policy = model_policy(hw)
    assert set(policy) >= {"torch_device", "threads", "allow_slm", "allow_whisper",
                           "allow_vlm", "allow_ecapa"}
    assert policy["threads"] >= 2


def test_policy_never_disables_ecapa_under_normal_ram():
    """Requirement: normal RAM pressure must NOT permanently disable speaker verification."""
    from core.system.hardware_detector import model_policy
    for avail in (400, 600, 900, 1500, 8000):
        p = model_policy({"available_ram_mb": avail, "torch_device": "cpu",
                          "logical_cores": 4})
        assert p["allow_ecapa"] is True, f"ECAPA disabled at {avail} MB (must stay available)"


# ---------------------------------------------------------------- registry
def test_model_registry_integrity():
    import yaml
    from config.settings import get_settings
    path = os.path.join(get_settings().base_dir, "models", "model_registry.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    models = data["models"]
    assert len(models) >= 8
    ids = {m["model_id"] for m in models}
    assert {"qwen3-0.6b-instruct", "whisper-tiny-local", "ecapa-voxceleb-speaker"} <= ids
    for m in models:
        assert m.get("license"), f"{m['model_id']} missing license"
        assert m.get("model_revision"), f"{m['model_id']} missing revision"
        assert m.get("task")
        if m.get("hf_repo"):
            assert m.get("loader_method"), f"{m['model_id']} missing explicit loader_method"
    # Vision/TTS stay opt-in on 8GB hardware
    by_id = {m["model_id"]: m for m in models}
    assert by_id["smolvlm-256m-instruct"]["enabled"] is False
    assert by_id["indic-parler-tts-mini"]["enabled"] is False


def test_loader_methods_exist_on_manager():
    from core.models.model_manager import get_model_manager
    mm = get_model_manager()
    for mid, e in mm.entries.items():
        if e.get("loader_method"):
            assert hasattr(mm, e["loader_method"]), f"loader {e['loader_method']} missing"


# ---------------------------------------------------------------- ASR vocab
def test_asr_vocab_correction():
    from voice.stt import apply_vocab_fixes
    assert "vscode" in apply_vocab_fixes("open vs code").lower()
    assert apply_vocab_fixes("my craft edge").lower().startswith("microsoft")
    assert apply_vocab_fixes("go to my not bad").lower().find("notepad") >= 0
    assert apply_vocab_fixes("") is None


# ---------------------------------------------------------------- memory
@pytest.fixture()
def memory_store(tmp_path):
    from memory.store import MemoryStore
    return MemoryStore(db_path=str(tmp_path / "mem.db"))


def test_memory_roundtrip(memory_store):
    mid = memory_store.remember("Owner prefers Edge as default browser", kind="preference")
    assert mid is not None
    rows = memory_store.search("browser")
    assert any(r[0] == mid for r in rows)
    listed = memory_store.list()
    assert any(item["content"].startswith("Owner prefers") for item in listed)
    assert memory_store.forget(memory_id=mid) == 1
    assert all(item["id"] != mid for item in memory_store.list())


def test_memory_rejects_secrets_and_never_logs_them(memory_store):
    assert memory_store.remember("my password is hunter2") is None
    assert memory_store.remember("api key sk-live-123") is None
    assert memory_store.remember("") is None
    assert memory_store.list() == []


def test_memory_ambiguous_forget_is_refused(memory_store):
    memory_store.remember("project alpha runs on tuesdays")
    memory_store.remember("project alpha has three members")
    assert memory_store.forget(content_matches="project alpha") == 0  # ambiguous
    assert len(memory_store.list()) == 2


def test_memory_clear_only_temp(memory_store):
    memory_store.remember("long term fact", kind="fact")
    memory_store.remember("temporary focus", kind="task_context")
    removed = memory_store.clear_temp()
    assert removed == 1
    remain = memory_store.list()
    assert len(remain) == 1 and remain[0]["kind"] == "fact"


# ---------------------------------------------------------------- jobs
@pytest.fixture()
def job_store(tmp_path):
    from core.automation.job_store import JobStore
    return JobStore(db_path=str(tmp_path / "jobs.db"))


def test_job_store_roundtrip_and_recurrence(job_store):
    jid = job_store.add_job("daily report", {"skill": "system", "action": "get_battery"},
                            recurrence="daily")
    assert job_store.due_jobs(), "job should be due immediately"
    job_store.mark_running(jid)
    assert job_store.mark_done(jid, "ok")
    jobs = job_store.list_jobs()
    assert jobs[0]["status"] == "pending"  # recurrent rescheduling
    assert jobs[0]["next_run_at"] > time.time()


def test_job_store_retry_then_fail(job_store):
    jid = job_store.add_job("flakey", {"skill": "system", "action": "get_battery"},
                            next_run_at=time.time() - 1, max_retries=1)
    job_store.mark_failed(jid, "boom")
    pending = job_store.list_jobs()
    assert pending[0]["status"] == "pending" and pending[0]["retries"] == 1
    job_store.mark_failed(jid, "boom again")
    assert job_store.list_jobs()[0]["status"] == "failed"


def test_job_store_cancel(job_store):
    jid = job_store.add_job("cancel me", {"skill": "system", "action": "get_battery"})
    assert job_store.cancel(jid)
    assert job_store.list_jobs()[0]["status"] == "cancelled"


# ---------------------------------------------------------------- scheduler
class _FakeResult:
    success = True
    message = "done"
    evidence = "ev"


class _FakeRegistry:
    def execute(self, skill, action, params):
        assert skill == "system" and action == "get_battery"
        return _FakeResult()


def test_scheduler_dispatches_via_registry(monkeypatch):
    """Scheduler must dispatch jobs through the explicit skill registry contract."""
    from core.automation import task_scheduler
    monkeypatch.setattr("skills.registry.get_skill_registry", lambda: _FakeRegistry())
    job = {"id": "x", "payload": {"skill": "system", "action": "get_battery", "params": {}}}
    assert task_scheduler.TaskScheduler._default_handler(job) == "done"


def test_scheduler_refuses_unknown_payload():
    from core.automation.task_scheduler import TaskScheduler
    with pytest.raises(ValueError):
        TaskScheduler._default_handler({"id": "x", "payload": {}})


def test_scheduler_skips_unapproved_confirmation_jobs(monkeypatch):
    """A confirmation_required job must not run until the owner approves it."""
    from core.automation.task_scheduler import TaskScheduler

    class _Store:
        def __init__(self):
            self.jobs = [
                {"id": "needs_approval", "payload": {"skill": "system",
                                                     "action": "get_battery"},
                 "confirmation_required": 1, "owner_approved": 0},
                {"id": "safe_job", "payload": {"skill": "system",
                                               "action": "get_battery"},
                 "confirmation_required": 0, "owner_approved": 0},
            ]
            self.started = []

        def due_jobs(self):
            return list(self.jobs)

        def mark_running(self, jid):
            self.started.append(jid)

        def mark_done(self, jid, result):
            pass

        def mark_failed(self, jid, error):
            pass

    store = _Store()
    monkeypatch.setattr("core.automation.task_scheduler.get_job_store", lambda: store)
    scheduler = TaskScheduler(poll_seconds=0.01, handler=lambda job: "done")
    scheduler._tick()

    assert "needs_approval" not in store.started, "unapproved high-risk job must be skipped"
    assert "safe_job" in store.started, "ordinary jobs still dispatch"


# ---------------------------------------------------------------- recovery
def test_recovery_pin_hash_and_lockout(tmp_path, monkeypatch):
    import security.recovery as rec
    monkeypatch.setattr(rec, "_RECOVERY_PATH", str(tmp_path / "recovery.json"))
    monkeypatch.setattr(rec, "_attempts", [])
    assert rec.set_owner_pin("482913")
    raw = open(rec._RECOVERY_PATH, encoding="utf-8").read()
    assert "482913" not in raw, "PIN must never be stored in plaintext"
    ok, msg = rec.verify_owner_pin("482913")
    assert ok and msg == "ok"
    for _ in range(rec._MAX_ATTEMPTS_BEFORE_LOCKOUT):
        ok, _ = rec.verify_owner_pin("000000")
        assert not ok
    ok, msg = rec.verify_owner_pin("482913")
    assert not ok and msg == "locked"


def test_recovery_rejects_weak_pin(tmp_path, monkeypatch):
    import security.recovery as rec
    monkeypatch.setattr(rec, "_RECOVERY_PATH", str(tmp_path / "recovery.json"))
    assert not rec.set_owner_pin("12")
    assert not rec.set_owner_pin("abcd")


def test_local_only_blocks_cloud_reasoning():
    """LOCAL_ONLY=true must return the offline provider even when keys exist."""
    from config.settings import Settings
    from llm.mock_provider import MockProvider
    from llm.openrouter_provider import OpenRouterProvider
    from llm.provider_factory import get_llm_provider

    blocked = Settings(llm_provider="openrouter", openrouter_api_key="sk-or-test",
                       local_only=True, allow_remote_llm=False)
    assert isinstance(get_llm_provider(settings=blocked), MockProvider)

    allowed = Settings(llm_provider="openrouter", openrouter_api_key="sk-or-test",
                       local_only=True, allow_remote_llm=True)
    assert isinstance(get_llm_provider(settings=allowed), OpenRouterProvider)


def test_local_only_forces_local_voice_transport(monkeypatch):
    """LOCAL_ONLY must not send microphone audio through the LiveKit cloud relay."""
    import voice.voice_engine as ve
    from config.settings import Settings

    seen = {}

    class _FakeLegacy:
        def __init__(self, **kwargs):
            seen["legacy"] = True

    class _FakeLiveKit:
        def __init__(self, **kwargs):
            seen["livekit"] = True

    monkeypatch.setattr(ve, "LegacyVoiceEngine", _FakeLegacy)
    monkeypatch.setattr(ve, "LiveKitVoiceEngine", _FakeLiveKit)

    local = Settings(local_only=True, voice_provider="livekit")
    assert isinstance(ve.get_voice_engine(settings=local), _FakeLegacy)
    assert seen.get("legacy") is True and "livekit" not in seen

    remote = Settings(local_only=False, voice_provider="livekit")
    assert isinstance(ve.get_voice_engine(settings=remote), _FakeLiveKit)


def test_custom_profile_ecapa_stays_isolated(tmp_path):
    """Test/custom profiles must never write to the production biometric file."""
    from voice.speaker_verification import SpeakerVerifier

    custom = tmp_path / "owner_profile.dat"
    verifier = SpeakerVerifier(profile_path=str(custom))
    assert verifier._ecapa_path == str(tmp_path / "owner_profile_ecapa.json")
    assert "data" not in verifier._ecapa_path


def test_ecapa_trim_removes_silence_padding():
    """30-60 s VAD recordings must not dilute the ECAPA embedding with silence."""
    import numpy as np
    from voice.ecapa_engine import EcapaEmbeddingEngine

    sr = 16000
    t = np.arange(int(sr * 3), dtype=np.float32) / sr
    speech = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    padded = np.concatenate([
        np.zeros(sr * 20, dtype=np.float32),
        speech,
        np.zeros(sr * 20, dtype=np.float32),
    ])
    trimmed = EcapaEmbeddingEngine._trim_speech(padded, sr, max_seconds=6.0)
    assert trimmed is not None
    assert len(trimmed) <= int(sr * 6.1)
    assert len(trimmed) >= int(sr * 2.0), "speech must survive trimming"


def test_ecapa_threshold_calibration_has_security_floor():
    """ECAPA threshold is calibrated per space and never drops below 0.60."""
    from voice.speaker_verification import SpeakerVerifier

    assert SpeakerVerifier.calibrate_ecapa_threshold([]) >= 0.60
    scores = [0.199, 0.655, 0.710, 0.712, 0.787, 0.806, 0.833, 0.841, 0.860]
    calibrated = SpeakerVerifier.calibrate_ecapa_threshold(scores)
    assert 0.60 <= calibrated <= 0.72
    # Enrollment self-scores are biased high; never exceed the lockout cap.
    assert SpeakerVerifier.calibrate_ecapa_threshold([0.95, 0.97]) <= 0.72
    assert SpeakerVerifier.calibrate_ecapa_threshold([0.99] * 10) <= 0.72


def test_ecapa_only_profile_can_verify():
    """An ECAPA profile must work even if the legacy MFCC profile is absent."""
    import numpy as np
    from voice.speaker_verification import SpeakerVerifier
    from config.constants import SpeakerClassification

    verifier = SpeakerVerifier.__new__(SpeakerVerifier)
    centroid = np.ones(192, dtype="float32") / np.sqrt(192)
    verifier.owner_embedding = None
    verifier.representative_embeddings = []
    verifier.ecapa_centroid = centroid
    verifier.ecapa_representatives = []
    verifier.ecapa_threshold = 0.62
    verifier.ecapa_uncertain_threshold = 0.52
    verifier.similarity_threshold_owner = 0.78
    verifier.similarity_threshold_uncertain = 0.70
    verifier._last_embedding_space = "ecapa"
    verifier.extract_embedding = lambda audio_data: centroid

    result, confidence = verifier.verify_speaker(b"RIFF-fake-audio")
    assert result == SpeakerClassification.OWNER
    assert confidence > 0.99


# ---------------------------------------------------------------- integration
def test_skills_include_memory_and_scheduler():
    from skills.registry import get_skill_registry
    names = get_skill_registry().list_skills()
    assert "memory" in names
    assert "scheduler" in names


def test_verify_speaker_fails_closed_when_ecapa_unavailable(monkeypatch):
    """With an ECAPA profile present, memory/model failure must NOT fall back to weaker auth."""
    import numpy as np
    from voice.speaker_verification import SpeakerVerifier
    from config.constants import SpeakerClassification

    verifier = SpeakerVerifier.__new__(SpeakerVerifier)  # bypass disk profile requirements
    verifier.ecapa_centroid = np.ones(192, dtype="float32") / np.sqrt(192)
    verifier.ecapa_representatives = []
    verifier.owner_embedding = np.ones(42, dtype="float32") / np.sqrt(42)
    verifier.representative_embeddings = []
    verifier.similarity_threshold_owner = 0.81
    verifier.similarity_threshold_uncertain = 0.70
    verifier._last_embedding_space = "mfcc"

    class _Unavailable:
        def available(self):
            return False

    verifier._ecapa = _Unavailable()
    result = verifier.extract_embedding(b"RIFF-fake-audio")
    assert result is None, "must fail closed, not silently use MFCC"
