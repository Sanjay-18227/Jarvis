"""
Application Controller Subsystem
Safely launches, verifies, and closes desktop applications using local-first application discovery.
Adheres strictly to Section 13: Local-First Application Intelligence.
"""

import os
import subprocess
import time
from typing import Dict, Optional, Tuple
from computer.app_indexer import AppIndexer


class AppController:
    """
    Manages desktop applications with verified local discovery and zero Store redirects.
    """

    def __init__(self, indexer: Optional[AppIndexer] = None):
        self.indexer = indexer or AppIndexer()

    def is_process_running(self, process_name: str) -> bool:
        """Checks if process is currently running using Windows tasklist."""
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq {process_name}"],
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            return process_name.lower() in output.lower()
        except Exception:
            return False

    def open_application(self, app_name: str) -> Tuple[bool, str]:
        """
        Launches an application or brings it into focus if already open.
        Searches the laptop locally first.
        """
        return self.indexer.launch_app(app_name)

    def close_application(self, app_name: str) -> Tuple[bool, str]:
        """
        Closes an application by process name or window title.
        """
        clean_name = app_name.lower().strip()
        from computer.windows_controller import WindowsController
        win_ctrl = WindowsController()

        # Try clean window close first
        closed, close_msg = win_ctrl.close_window(clean_name)
        if closed:
            return True, f"Closed {app_name.capitalize()}, Sir."

        # Process termination fallback
        app_info = self.indexer.find_app(clean_name)
        if app_info and "process" in app_info:
            proc = app_info["process"]
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", proc],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                time.sleep(0.5)
                return True, f"{app_name.capitalize()} terminated."
            except Exception:
                pass

        return False, f"Could not find or close active application '{app_name}'."
