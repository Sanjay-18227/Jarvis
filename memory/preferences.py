"""
User Preferences & Long-term Settings Management
"""

import os
import json
from typing import Any, Dict, Optional
from config.settings import get_settings


class PreferencesManager:
    """
    Persists and loads owner personal preferences locally in JSON format.
    """

    def __init__(self, filepath: Optional[str] = None):
        if filepath:
            self.filepath = filepath
        else:
            base_dir = get_settings().base_dir
            self.filepath = os.path.join(base_dir, "memory", "user_preferences.json")

        self._preferences: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Loads preferences from disk or initializes defaults."""
        defaults = {
            "owner_name": "Sanjay",
            "preferred_title": "Sir",
            "language": "en-US",
            "tts_voice": "en-US-ChristopherNeural",
            "voice_speed": 1.0,
            "favorite_apps": {
                "editor": "VS Code",
                "browser": "Chrome",
                "terminal": "PowerShell",
            },
        }

        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    defaults.update(data)
            except Exception as e:
                print(f"[Preferences] Error loading preferences: {e}")

        return defaults

    def save(self):
        """Save preferences to disk."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self._preferences, f, indent=2)
        except Exception as e:
            print(f"[Preferences] Error saving preferences: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._preferences.get(key, default)

    def set(self, key: str, value: Any):
        self._preferences[key] = value
        self.save()
