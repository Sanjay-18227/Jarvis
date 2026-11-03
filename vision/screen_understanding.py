"""
Screen Understanding (ScreenVision) Subsystem (Phase 4)
Performs on-demand visual analysis using Gemini multimodal vision via ScreenVisionService.
Fulfills Requirement G & Section 27:
- Never claims to see the screen unless a real vision tool executed successfully.
- Conforms to structured visual result contract.
"""

from typing import Optional, Dict, Any, Tuple
from core.tool_result import ToolResult
from vision.screen_capture import ScreenCapture
from vision.screen_vision_service import ScreenVisionService, get_screen_vision_service
from config.constants import SpeakerClassification
from config.settings import get_settings


class ScreenVision:
    """
    Analyzes desktop screen and active window context on user request.
    Strictly truthful: reports failure if vision capability is not currently active.
    Wraps ScreenVisionService for full Phase 4 multimodal capabilities.
    """

    def __init__(
        self,
        capture: Optional[ScreenCapture] = None,
        service: Optional[ScreenVisionService] = None,
        enabled: Optional[bool] = None,
    ):
        self.capture = capture or ScreenCapture()
        self.service = service or ScreenVisionService(capture=self.capture, enabled=enabled)
        self.enabled = enabled

    def inspect_screen(
        self,
        user_query: str,
        speaker: SpeakerClassification = SpeakerClassification.OWNER,
        mode: Optional[str] = None,
    ) -> ToolResult:
        """
        Captures the screen and asks multimodal model to analyze the UI/error.
        Returns ToolResult. Never claims to see the screen if vision did not run.
        """
        print(f"[ROUTER] intent=SCREEN_VISION query='{user_query}'")
        print(f"[TOOL] screen_capture display")

        q_low = user_query.lower()

        # Determine optimal capture mode (ACTIVE_WINDOW vs FULL_SCREEN)
        if mode:
            chosen_mode = mode
        elif any(k in q_low for k in ["error", "dialog", "button", "terminal", "this window", "which application am i looking at"]):
            chosen_mode = "ACTIVE_WINDOW"
        elif any(k in q_low for k in ["all apps", "what apps", "entire screen", "what is on my screen", "what's on my screen"]):
            chosen_mode = "FULL_SCREEN"
        else:
            chosen_mode = "ACTIVE_WINDOW"

        # Execute analysis via ScreenVisionService
        res = self.service.analyze_screen(
            mode=chosen_mode,
            user_query=user_query,
            speaker=speaker,
        )

        if res.get("success", False):
            print("[VERIFY] screen_captured=True")
            print("[RESULT] success=True")
            return ToolResult.ok(
                action="inspect_screen",
                message=res.get("summary", "Analysis complete."),
                evidence=str(res.get("evidence", {})),
            )
        else:
            print("[VERIFY] screen_captured=False")
            print(f"[RESULT] success=False error='{res.get('error')}'")
            return ToolResult.fail(
                action="inspect_screen",
                message=res.get("summary", "Sir, screen inspection isn't available right now."),
                error=res.get("error", "Vision unavailable"),
            )
