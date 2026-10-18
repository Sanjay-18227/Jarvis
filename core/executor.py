"""
Trusted Action Executor Subsystem
Validates, permissions-checks, risk-scores, and executes structured computer actions.
Never executes raw or unvalidated commands directly.
"""

from typing import Dict, Any, Tuple, Optional
from config.constants import RiskLevel, SpeakerClassification, ActionStatus
from core.permissions import PermissionManager
from core.risk_engine import RiskEngine
from security.confirmation import ConfirmationManager
from security.audit_log import AuditLogger
from computer.app_controller import AppController
from computer.file_manager import FileManager
from computer.windows_controller import WindowsController


class ActionExecutor:
    """
    Central verified execution gatekeeper.
    """

    def __init__(
        self,
        permission_mgr: Optional[PermissionManager] = None,
        risk_engine: Optional[RiskEngine] = None,
        confirmation_mgr: Optional[ConfirmationManager] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.permission_mgr = permission_mgr or PermissionManager()
        self.risk_engine = risk_engine or RiskEngine()
        self.confirmation_mgr = confirmation_mgr or ConfirmationManager()
        self.audit_logger = audit_logger or AuditLogger()

        self.app_ctrl = AppController()
        self.file_mgr = FileManager()
        self.win_ctrl = WindowsController()

    def execute_action(
        self,
        action: str,
        parameters: Dict[str, Any],
        speaker: SpeakerClassification,
        confirmed: bool = False,
    ) -> Tuple[bool, str]:
        """
        Executes a computer action following full validation pipeline:
        Risk Check -> Permission Check -> Confirmation Check -> Safe Execution -> Verify -> Audit.

        Returns:
            (success: bool, response_message: str)
        """
        act_clean = action.lower().strip()

        # Step 1: Calculate Risk Level
        risk_level, rationale = self.risk_engine.classify_action(act_clean, parameters)

        # Step 2: Permission Check
        is_allowed, reason = self.permission_mgr.check_permission(speaker, act_clean, risk_level)
        if not is_allowed:
            self.audit_logger.log_action(
                task=f"{action} {parameters}",
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
                parameters=parameters,
                risk_level=risk_level,
                description=f"Action '{act_clean}' involves sensitive modifications.",
            )
            self.audit_logger.log_action(
                task=f"{action} {parameters}",
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
            if act_clean == "open_application":
                target = parameters.get("target") or parameters.get("app") or ""
                success, msg = self.app_ctrl.open_application(target)

            elif act_clean == "close_application":
                target = parameters.get("target") or parameters.get("app") or ""
                success, msg = self.app_ctrl.close_application(target)

            elif act_clean == "search_google":
                query = parameters.get("query", "")
                success, msg = self.win_ctrl.search_google(query)

            elif act_clean == "media_control":
                sub_act = parameters.get("action", "play_pause")
                success, msg = self.win_ctrl.control_media(sub_act)

            elif act_clean == "lock_workstation":
                success, msg = self.win_ctrl.lock_workstation()

            elif act_clean == "create_folder":
                folder = parameters.get("path") or parameters.get("folder", "")
                success, msg = self.file_mgr.create_folder(folder)

            elif act_clean == "create_file":
                fpath = parameters.get("path") or parameters.get("file", "")
                content = parameters.get("content", "")
                success, msg = self.file_mgr.create_file(fpath, content)

            elif act_clean == "rename_file":
                src = parameters.get("old_path", "")
                dst = parameters.get("new_path", "")
                success, msg = self.file_mgr.rename_file(src, dst)

            elif act_clean == "delete_path":
                target = parameters.get("path", "")
                success, msg = self.file_mgr.delete_path(target)

            else:
                return False, f"Unsupported action: {action}"

        except Exception as e:
            success = False
            msg = f"Execution error during {act_clean}: {str(e)}"

        # Step 5: Audit Logging
        status = ActionStatus.SUCCESS if success else ActionStatus.FAILED
        self.audit_logger.log_action(
            task=f"{action} {parameters}",
            speaker=speaker,
            action=act_clean,
            risk_level=risk_level,
            status=status,
            details={"result": msg},
        )

        return success, msg
