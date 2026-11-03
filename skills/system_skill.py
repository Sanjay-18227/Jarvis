"""
JARVIS System Skill
Handles:
- System vitals (battery percentage, power status)
- Audio volume and media controls
- Workstation lock screen
- Date and time query
- Performance diagnostics
- Advertises: get_battery, get_volume, set_volume, lock_workstation, get_diagnostics, get_datetime
"""

import datetime
from typing import Dict, Any, List, Optional
from skills.base import BaseSkill
from core.tool_result import ToolResult
from computer.system_monitor import SystemMonitor
from computer.windows_controller import WindowsController
from core.context_memory import get_context_memory


class SystemSkill(BaseSkill):
    """System vitals, hardware, and desktop controls skill."""

    def __init__(
        self,
        sys_monitor: Optional[SystemMonitor] = None,
        win_ctrl: Optional[WindowsController] = None,
    ):
        self.sys_monitor = sys_monitor or SystemMonitor()
        self.win_ctrl = win_ctrl or WindowsController()

    @property
    def name(self) -> str:
        return "system"

    @property
    def description(self) -> str:
        return "System hardware status, battery percentage, volume control, and workstation lock."

    @property
    def supported_actions(self) -> List[str]:
        return ["get_battery", "get_volume", "set_volume", "lock_workstation", "get_diagnostics", "get_datetime"]

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        int_clean = intent.lower().strip()
        return int_clean in [
            "battery", "get_battery", "diagnostics", "get_diagnostics",
            "datetime", "get_datetime", "media_control", "volume", "lock_screen", "lock_workstation"
        ]

    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        ctx = context or get_context_memory()
        act = action.lower().strip()

        if act in ["battery", "get_battery"]:
            vitals = self.sys_monitor.get_system_vitals()
            pct = vitals["battery"]["percent"]
            plugged = vitals["battery"]["plugged"]
            status_desc = "charging" if plugged else "discharging"
            evidence = f"percent={pct}, plugged={plugged}, status={status_desc}"
            msg = f"Your battery is at {pct}%, currently {status_desc}, Sir."
            ctx.update(last_tool="battery", last_result=f"{pct}% {status_desc}")
            return ToolResult.ok(action="get_battery", target="battery", evidence=evidence, message=msg)

        elif act in ["media_control", "volume", "set_volume"]:
            sub_act = params.get("action", "volume_up")
            self.win_ctrl.control_media(sub_act)
            if sub_act == "volume_up":
                msg = "Volume increased, Sir."
            elif sub_act == "volume_down":
                msg = "Volume decreased, Sir."
            elif sub_act == "mute":
                msg = "Mute toggled, Sir."
            else:
                msg = "Media playback toggled, Sir."
            ctx.update(last_tool="media_control", last_result=sub_act)
            return ToolResult.ok(action="volume", target="system_audio", evidence=f"action={sub_act}", message=msg)

        elif act in ["lock_screen", "lock_workstation", "lock"]:
            self.win_ctrl.lock_workstation()
            ctx.update(last_tool="lock_screen", last_result="Workstation locked")
            return ToolResult.ok(
                action="lock_screen",
                target="workstation",
                evidence="LockWorkStation called",
                message="Workstation locked, Sir.",
            )

        elif act in ["datetime", "get_datetime"]:
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p")
            date_str = now.strftime("%A, %B %d")
            msg = f"It's {time_str} on {date_str}, Sir."
            ctx.update(last_tool="datetime", last_result=msg)
            return ToolResult.ok(action="get_datetime", target="clock", evidence=f"time={time_str}, date={date_str}", message=msg)

        elif act in ["diagnostics", "get_diagnostics"]:
            _, diagnosis = self.sys_monitor.diagnose_performance()
            ctx.update(last_tool="diagnostics", last_result=diagnosis)
            return ToolResult.ok(action="diagnostics", target="system", evidence=diagnosis, message=diagnosis)

        return ToolResult.fail(action=act, error=f"Unsupported system action: '{action}'")
