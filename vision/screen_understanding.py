"""
Screen Understanding (ScreenVision) Subsystem
Performs on-demand visual analysis using Gemini multimodal vision.
Fulfills Requirement G: Never claims to see the screen unless a real vision tool executed successfully.
"""

from typing import Optional, Dict, Any, Tuple
from core.tool_result import ToolResult
from vision.screen_capture import ScreenCapture
from config.constants import SpeakerClassification
from config.settings import get_settings


class ScreenVision:
    """
    Analyzes desktop screen and active window context on user request.
    Strictly truthful: reports failure if vision capability is not currently active.
    """

    def __init__(self, capture: Optional[ScreenCapture] = None, enabled: Optional[bool] = None):
        self.capture = capture or ScreenCapture()
        self.enabled = enabled

    def inspect_screen(
        self,
        user_query: str,
        speaker: SpeakerClassification = SpeakerClassification.OWNER,
    ) -> ToolResult:
        """
        Captures the screen and asks multimodal model to analyze the UI/error.
        Returns ToolResult. Never claims to see the screen if vision is not working.
        """
        print(f"[ROUTER] intent=SCREEN_VISION query='{user_query}'")
        print(f"[TOOL] screen_capture display")

        cfg = get_settings()
        is_enabled = self.enabled if self.enabled is not None else getattr(cfg, "screen_vision_enabled", False)
        if not is_enabled or not cfg.gemini_api_key:
            print(f"[RESULT] success=False reason='Screen vision not enabled'")
            return ToolResult.fail(
                action="inspect_screen",
                message="Sir, screen inspection isn't enabled yet, so I can't visually verify it.",
                error="Vision API not active",
            )

        # Step 1: Detect active window
        win_info = self.capture.get_active_window_info()

        # Step 2: Privacy check
        if win_info.get("is_sensitive", False):
            if speaker != SpeakerClassification.OWNER:
                print(f"[RESULT] success=False error='Privacy Block'")
                return ToolResult.fail(
                    action="inspect_screen",
                    message="Access denied: The active window contains sensitive private credentials.",
                    error="Sensitive window in foreground",
                )

        # Step 3: Capture display
        img_bytes = self.capture.capture_bytes()
        if not img_bytes:
            print(f"[VERIFY] screen_captured=False")
            print(f"[RESULT] success=False")
            return ToolResult.fail(
                action="inspect_screen",
                message="Sir, I was unable to capture your display.",
                error="Display capture returned empty bytes",
            )

        print(f"[VERIFY] screen_captured=True bytes={len(img_bytes)}")

        # Step 4: Ask Gemini Vision
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
            analysis_text = (response.text or "").strip()
            if analysis_text:
                print(f"[RESULT] success=True")
                return ToolResult.ok(
                    action="inspect_screen",
                    message=analysis_text,
                    evidence=f"Active window: {win_info.get('title')}",
                )
        except Exception as e:
            print(f"[RESULT] success=False error='{str(e)}'")
            return ToolResult.fail(
                action="inspect_screen",
                message="Sir, screen inspection isn't enabled yet, so I can't visually verify it.",
                error=f"Vision model error: {str(e)}",
            )

        print(f"[RESULT] success=False reason='Screen vision returned empty analysis'")
        return ToolResult.fail(
            action="inspect_screen",
            message="Sir, screen inspection isn't enabled yet, so I can't visually verify it.",
            error="Vision API not active",
        )

