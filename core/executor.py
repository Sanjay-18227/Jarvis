"""
Trusted Action Executor Subsystem
Validates, permissions-checks, risk-scores, and executes structured computer actions.
Never executes raw or unvalidated commands directly.
Enforces defense-in-depth with mandatory confirmation on Level 3/4 sensitive actions.
"""

from typing import Dict, Any, Tuple, Optional
from config.constants import RiskLevel, SpeakerClassification, ActionStatus
from core.permissions import PermissionManager
from core.risk_engine import RiskEngine
from security.confirmation import ConfirmationManager
from security.audit_log import AuditLogger
from security.secrets_manager import SecretsManager
from computer.app_controller import AppController
from computer.file_manager import FileManager
from computer.windows_controller import WindowsController
from computer.mouse_controller import MouseController
from computer.keyboard_controller import KeyboardController
from computer.browser_controller import BrowserController
from computer.terminal_controller import TerminalController
from computer.system_monitor import SystemMonitor
from ai.research_agent import ResearchAgent
from core.project_detector import ProjectDetector


class ActionExecutor:
    """
    Central verified execution gatekeeper for Windows autonomous computer control.
    """

    def __init__(
        self,
        permission_mgr: Optional[PermissionManager] = None,
        risk_engine: Optional[RiskEngine] = None,
        confirmation_mgr: Optional[ConfirmationManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        secrets_mgr: Optional[SecretsManager] = None,
    ):
        self.permission_mgr = permission_mgr or PermissionManager()
        self.risk_engine = risk_engine or RiskEngine()
        self.confirmation_mgr = confirmation_mgr or ConfirmationManager()
        self.audit_logger = audit_logger or AuditLogger()
        self.secrets_mgr = secrets_mgr or SecretsManager()

        # Native Controllers
        self.app_ctrl = AppController()
        self.file_mgr = FileManager()
        self.win_ctrl = WindowsController()
        self.mouse_ctrl = MouseController()
        self.keyboard_ctrl = KeyboardController()
        self.browser_ctrl = BrowserController(keyboard=self.keyboard_ctrl)
        self.terminal_ctrl = TerminalController()

        # Performance, Research, & Project Automation Subsystems
        self.sys_monitor = SystemMonitor()
        self.research_agent = ResearchAgent()
        self.project_detector = ProjectDetector()

    def execute_action(
        self,
        action: str,
        parameters: Optional[Dict[str, Any]] = None,
        speaker: SpeakerClassification = SpeakerClassification.OWNER,
        confirmed: bool = False,
    ) -> Tuple[bool, str]:
        """
        Executes a computer action following the full validation pipeline:
        Risk Check -> Permission Check -> Confirmation Check -> Safe Execution -> Verify -> Audit.

        Returns:
            (success: bool, response_message: str)
        """
        params = parameters or {}
        act_clean = action.lower().strip()

        # Step 1: Calculate Risk Level
        risk_level, rationale = self.risk_engine.classify_action(act_clean, params)

        # Step 2: Permission Check
        is_allowed, reason = self.permission_mgr.check_permission(speaker, act_clean, risk_level)
        if not is_allowed:
            self.audit_logger.log_action(
                task=f"{action} {self.secrets_mgr.sanitize(str(params))}",
                speaker=speaker,
                action=act_clean,
                risk_level=risk_level,
                status=ActionStatus.DENIED,
                details={"reason": reason},
            )
            return False, reason

        # Step 3: Mandatory Confirmation Check for High Risk (Level 3/4)
        if risk_level >= RiskLevel.LEVEL_3_SENSITIVE and not confirmed:
            prompt = self.confirmation_mgr.request_confirmation(
                action_name=act_clean,
                parameters=params,
                risk_level=risk_level,
                description=f"Action '{act_clean}' involves sensitive modifications.",
            )
            self.audit_logger.log_action(
                task=f"{action} {self.secrets_mgr.sanitize(str(params))}",
                speaker=speaker,
                action=act_clean,
                risk_level=risk_level,
                status=ActionStatus.AWAITING_CONFIRMATION,
            )
            return False, prompt

        # Step 4: Dispatch to Safe Native Controllers
        success = False
        msg = ""

        try:
            # 1. Applications
            if act_clean in ["open_application", "open_app"]:
                target = params.get("target") or params.get("app") or params.get("name", "")
                success, msg = self.app_ctrl.open_application(target)

            elif act_clean in ["close_application", "close_app"]:
                target = params.get("target") or params.get("app") or params.get("name", "")
                success, msg = self.app_ctrl.close_application(target)

            # 2. Window Management
            elif act_clean == "focus_window":
                keyword = params.get("title") or params.get("keyword") or params.get("target", "")
                success, msg = self.win_ctrl.focus_window(keyword)

            elif act_clean == "list_windows":
                windows = self.win_ctrl.list_windows()
                titles = [w["title"] for w in windows[:10]]
                success = True
                msg = f"Visible windows ({len(windows)}): {', '.join(titles)}"

            elif act_clean == "close_window":
                keyword = params.get("title") or params.get("keyword", "")
                success, msg = self.win_ctrl.close_window(keyword)

            # 3. Web & Browser
            elif act_clean in ["search_web", "search_google"]:
                query = params.get("query", "")
                success, msg = self.browser_ctrl.search_web(query)

            elif act_clean == "open_url":
                url = params.get("url", "")
                success, msg = self.browser_ctrl.open_url(url)

            elif act_clean == "open_documentation":
                topic = params.get("topic") or params.get("library", "")
                success, msg = self.browser_ctrl.open_documentation(topic)

            elif act_clean == "browser_tab":
                tab_action = params.get("action", "new")
                if tab_action == "close":
                    success, msg = self.browser_ctrl.close_tab()
                elif tab_action == "next":
                    success, msg = self.browser_ctrl.next_tab()
                else:
                    success, msg = self.browser_ctrl.new_tab()

            # 4. Mouse Automation
            elif act_clean == "mouse_action":
                m_action = params.get("action", "click").lower()
                x = params.get("x")
                y = params.get("y")
                if m_action == "click":
                    btn = params.get("button", "left")
                    success, msg = self.mouse_ctrl.click(x=x, y=y, button=btn)
                elif m_action == "double_click":
                    success, msg = self.mouse_ctrl.double_click(x=x, y=y)
                elif m_action == "right_click":
                    success, msg = self.mouse_ctrl.right_click(x=x, y=y)
                elif m_action == "move":
                    if x is not None and y is not None:
                        success, msg = self.mouse_ctrl.move_to(x, y)
                    else:
                        success, msg = False, "Coordinates (x, y) required for mouse move."
                elif m_action == "scroll":
                    clicks = int(params.get("clicks", -5))
                    success, msg = self.mouse_ctrl.scroll(clicks)
                elif m_action == "drag":
                    sx, sy = params.get("start_x", 0), params.get("start_y", 0)
                    ex, ey = params.get("end_x", 0), params.get("end_y", 0)
                    success, msg = self.mouse_ctrl.drag(sx, sy, ex, ey)
                else:
                    success, msg = False, f"Unknown mouse action: {m_action}"

            # 5. Keyboard Automation
            elif act_clean == "keyboard_action":
                k_action = params.get("action", "type").lower()
                if k_action == "type":
                    text = params.get("text", "")
                    success, msg = self.keyboard_ctrl.type_text(text)
                elif k_action in ["hotkey", "shortcut"]:
                    keys = params.get("keys", [])
                    if isinstance(keys, str):
                        keys = [k.strip() for k in keys.split("+")]
                    success, msg = self.keyboard_ctrl.hotkey(*keys)
                elif k_action == "enter":
                    success, msg = self.keyboard_ctrl.press_enter()
                elif k_action == "esc":
                    success, msg = self.keyboard_ctrl.press_escape()
                else:
                    success, msg = False, f"Unknown keyboard action: {k_action}"

            # 6. Filesystem
            elif act_clean == "create_file":
                fpath = params.get("path") or params.get("file", "")
                content = params.get("content", "")
                success, msg = self.file_mgr.create_file(fpath, content)

            elif act_clean == "read_file":
                fpath = params.get("path") or params.get("file", "")
                max_c = int(params.get("max_chars", 4000))
                success, msg = self.file_mgr.read_file(fpath, max_chars=max_c)

            elif act_clean == "create_folder":
                folder = params.get("path") or params.get("folder", "")
                success, msg = self.file_mgr.create_folder(folder)

            elif act_clean == "list_directory":
                folder = params.get("path") or params.get("folder", ".")
                success, entries, msg = self.file_mgr.list_files(folder)
                if success:
                    msg = f"Directory contents ({len(entries)} items): {', '.join(entries[:15])}"

            elif act_clean == "rename_file":
                src = params.get("old_path", "")
                dst = params.get("new_path", "")
                success, msg = self.file_mgr.rename_file(src, dst)

            elif act_clean == "delete_path":
                target = params.get("path", "")
                success, msg = self.file_mgr.delete_path(target)

            # 7. Supervised Terminal Execution
            elif act_clean in ["run_command", "terminal_command"]:
                cmd = params.get("command") or params.get("cmd", "")
                cwd = params.get("cwd")
                timeout = int(params.get("timeout", 45))
                success, ret_code, out = self.terminal_ctrl.run_command(cmd, cwd=cwd, timeout=timeout)
                msg = f"[Exit code {ret_code}]\n{out}"

            # 8. System Controls
            elif act_clean == "lock_workstation":
                success, msg = self.win_ctrl.lock_workstation()

            elif act_clean == "media_control":
                sub_act = params.get("action", "play_pause")
                success, msg = self.win_ctrl.control_media(sub_act)

            elif act_clean == "get_battery_status":
                status = self.win_ctrl.get_battery_status()
                success = True
                msg = f"Battery is at {status.get('percent', 100)}% ({status.get('status', 'OK')})."

            # 9. Laptop Performance Diagnosis (Section 30)
            elif act_clean in ["diagnose_laptop_performance", "diagnose_performance"]:
                success, msg = self.sys_monitor.diagnose_performance()

            # 10. AI Web Research (Section 14 & 15)
            elif act_clean in ["research_web_topic", "research_topic"]:
                topic = params.get("topic", "")
                success, msg = self.research_agent.research_topic(topic)

            # 11. Project Run & Detection (Section 21 & 22)
            elif act_clean == "run_project":
                pdir = params.get("project_dir")
                profile = self.project_detector.detect_project(pdir)
                run_cmd = profile.get("run_cmd")
                if not run_cmd:
                    success, msg = False, "Could not determine launch command for project."
                else:
                    success, ret_code, out = self.terminal_ctrl.run_command(run_cmd, cwd=profile.get("root"), timeout=15)
                    msg = f"Executed {profile.get('framework', 'project')} command '{run_cmd}': {out[:200]}"

            # 12. Coding Project Investigation (Section 23 & 25)
            elif act_clean == "investigate_project":
                pdir = params.get("project_dir") or os.getcwd()
                profile = self.project_detector.detect_project(pdir)
                test_cmd = profile.get("test_cmd", "pytest")
                success, ret_code, out = self.terminal_ctrl.run_command(test_cmd, cwd=pdir, timeout=30)
                msg = f"Investigated {profile.get('framework')} at {pdir}. Verification output: {out[:300]}"

            else:
                return False, f"Unsupported action: '{action}'."

        except Exception as e:
            success = False
            msg = f"Execution error during {act_clean}: {str(e)}"

        # Step 5: Audit Logging (secrets sanitized)
        status = ActionStatus.SUCCESS if success else ActionStatus.FAILED
        safe_params = self.secrets_mgr.sanitize(str(params))
        safe_msg = self.secrets_mgr.sanitize(msg)

        self.audit_logger.log_action(
            task=f"{action} {safe_params}",
            speaker=speaker,
            action=act_clean,
            risk_level=risk_level,
            status=status,
            details={"result": safe_msg[:500]},
        )

        return success, msg
