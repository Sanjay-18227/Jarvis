"""
Screen Understanding (ScreenVision) Subsystem
Performs on-demand visual analysis using Gemini 3.8 Flash multimodal vision.
"""

from typing import Optional, Dict, Any, Tuple
from vision.screen_capture import ScreenCapture
from config.constants import SpeakerClassification
from config.settings import get_settings


class ScreenVision:
    """
    Analyzes desktop screen and active window context on user request.
    """

    def __init__(self, capture: Optional[ScreenCapture] = None):
        self.capture = capture or ScreenCapture()

    def inspect_screen(
        self,
        user_query: str,
        speaker: SpeakerClassification = SpeakerClassification.OWNER,
    ) -> Tuple[bool, str]:
        """
        Captures the screen and asks Gemini multimodal model to analyze the UI/error.

        Args:
            user_query: Question about the screen (e.g. "What error is showing?")
            speaker: Speaker requesting the screen analysis.

        Returns:
            (success: bool, explanation_or_error: str)
        """
        # Step 1: Detect active window
        win_info = self.capture.get_active_window_info()

        # Step 2: Privacy check
        if win_info.get("is_sensitive", False):
            if speaker != SpeakerClassification.OWNER:
                return False, "Access denied: The active window contains sensitive private credentials."

        # Step 3: Capture display
        img_bytes = self.capture.capture_bytes()
        if not img_bytes:
            return False, "Could not capture display."

        # Step 4: Ask Gemini Vision
        cfg = get_settings()
        if cfg.gemini_api_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=cfg.gemini_api_key)
                prompt = (
                    f"Active Window: {win_info.get('title', 'Unknown')}\n"
                    f"User Request: {user_query}\n"
                    "Please analyze the screen and answer concisely (1-3 sentences) suitable to be spoken aloud."
                )

                response = client.models.generate_content(
                    model=cfg.gemini_model or "gemini-3.8-flash",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                )
                return True, response.text or "I inspected the screen, Sir."
            except Exception as e:
                return False, f"Vision model error: {str(e)}"

        # Offline fallback
        return True, f"Active window is '{win_info.get('title')}'. No error visible in offline simulation mode."
