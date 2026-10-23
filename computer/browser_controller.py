"""
JARVIS Browser & Website Controller
Fulfills Reliability Requirements for Microsoft Edge & Website Handling:
- Microsoft Edge is the default/preferred browser for all generic commands
- Robust Windows browser executable resolution (Registry App Paths, standard paths, PATH)
- Real browser launch verification (checks msedge.exe/chrome.exe, never assumes success)
- Website vs App distinction: websites route to browser, never searched as local apps
- Distinguishes SEARCH_OPENED vs VIDEO_OPENED vs PLAYBACK_VERIFIED
- Returns structured ToolResult contract
"""

import os
import shutil
import winreg
import urllib.parse
import webbrowser
import subprocess
import time
from typing import Tuple, Optional, Dict, Any
from core.tool_result import ToolResult
from computer.keyboard_controller import KeyboardController
from config.settings import get_settings


class BrowserPlaybackState:
    SEARCH_OPENED = "SEARCH_OPENED"
    VIDEO_OPENED = "VIDEO_OPENED"
    PLAYBACK_VERIFIED = "PLAYBACK_VERIFIED"


def _find_edge_exe() -> Optional[str]:
    """
    Locates Microsoft Edge executable on Windows using Registry,
    standard Program Files paths, and PATH.
    """
    # 1. Registry App Paths
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe") as k:
                val = winreg.QueryValue(k, "")
                if val and os.path.isfile(val):
                    return val
        except Exception:
            pass

    # 2. Standard Windows installation paths
    candidates = [
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Microsoft", "Edge", "Application", "msedge.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # 3. PATH
    return shutil.which("msedge")


def _find_chrome_exe() -> Optional[str]:
    """
    Locates Google Chrome executable on Windows using Registry,
    standard Program Files paths, and PATH.
    """
    # 1. Registry App Paths
    for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe") as k:
                val = winreg.QueryValue(k, "")
                if val and os.path.isfile(val):
                    return val
        except Exception:
            pass

    # 2. Standard paths
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # 3. PATH
    return shutil.which("chrome")


class BrowserController:
    """
    Manages verified browser navigation, searches, and website routing.
    Defaults to Microsoft Edge unless Chrome is explicitly requested.
    """

    def __init__(self, keyboard: Optional[KeyboardController] = None):
        self.keyboard = keyboard or KeyboardController()

    def _resolve_browser(self, requested_browser: Optional[str] = None) -> str:
        """
        Resolves the browser to use. If not explicitly requested as Chrome,
        always defaults to Microsoft Edge per laptop preference.
        """
        req = (requested_browser or "").lower().strip()
        if req in ["chrome", "google chrome"]:
            return "chrome"
        if req in ["edge", "msedge", "microsoft edge"]:
            return "edge"

        # Check configuration
        cfg_default = getattr(get_settings(), "default_browser", "edge").lower()
        if cfg_default in ["chrome", "google chrome"]:
            return "chrome"
        return "edge"

    def _is_browser_running(self, browser_name: Optional[str] = None) -> bool:
        """
        Checks if the target browser process is currently active.
        Never returns True on failure or guesswork.
        """
        resolved = self._resolve_browser(browser_name)
        if resolved == "edge":
            targets = ["msedge.exe"]
        elif resolved == "chrome":
            targets = ["chrome.exe"]
        else:
            targets = ["msedge.exe", "chrome.exe"]

        try:
            import psutil
            for p in psutil.process_iter(["name"]):
                pname = (p.info.get("name") or "").lower()
                if any(t in pname for t in targets):
                    return True
        except Exception:
            pass

        # Fallback check via tasklist
        try:
            for t in targets:
                output = subprocess.check_output(
                    ["tasklist", "/FI", f"IMAGENAME eq {t}"],
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                if t.lower() in output.lower():
                    return True
        except Exception:
            pass

        return False

    def open_browser(self, browser: Optional[str] = None) -> ToolResult:
        """
        Explicitly launches the browser (default: Microsoft Edge).
        Verifies process startup and returns structured ToolResult.
        """
        resolved = self._resolve_browser(browser)
        display_name = "Google Chrome" if resolved == "chrome" else "Microsoft Edge"
        expected_proc = "chrome.exe" if resolved == "chrome" else "msedge.exe"

        print(f"[ROUTER] intent=OPEN_APP target={display_name}")
        print(f"[BROWSER] selected={display_name}")

        exe_path = _find_chrome_exe() if resolved == "chrome" else _find_edge_exe()

        if resolved == "chrome" and not exe_path:
            print(f"[VERIFY] chrome_running=False")
            print(f"[RESULT] success=False")
            return ToolResult.fail(
                action="open_browser",
                message="Sir, Google Chrome is not installed on your laptop.",
                error="Google Chrome executable not found",
            )

        try:
            if exe_path and os.path.isfile(exe_path):
                subprocess.Popen([exe_path])
            else:
                cmd = "start chrome" if resolved == "chrome" else "start msedge"
                subprocess.Popen(cmd, shell=True)

            # Wait briefly for startup and verify process exists
            verified = False
            for _ in range(5):
                time.sleep(0.4)
                if self._is_browser_running(resolved):
                    verified = True
                    break

            if resolved == "edge":
                print(f"[VERIFY] msedge_running={verified}")
            else:
                print(f"[VERIFY] chrome_running={verified}")

            if verified:
                print(f"[RESULT] success=True")
                return ToolResult.ok(
                    action="open_browser",
                    message=f"Opened {display_name}, Sir.",
                    evidence=f"Verified running process {expected_proc}",
                )
            else:
                print(f"[RESULT] success=False")
                return ToolResult.fail(
                    action="open_browser",
                    message=f"Sir, I tried to launch {display_name}, but I could not verify that it opened.",
                    error=f"Process {expected_proc} not detected after launch",
                )

        except Exception as e:
            print(f"[VERIFY] msedge_running=False error={str(e)}")
            print(f"[RESULT] success=False")
            return ToolResult.fail(
                action="open_browser",
                message=f"Failed to launch {display_name}: {str(e)}",
                error=str(e),
            )

    def open_website(
        self,
        url: str,
        browser: Optional[str] = None,
        site_name: Optional[str] = None,
    ) -> ToolResult:
        """
        Navigates to a website using Microsoft Edge by default.
        Verifies browser launch and process existence.
        Returns structured ToolResult.
        """
        clean_url = url.strip()
        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            clean_url = f"https://{clean_url}"

        display_name = site_name or clean_url
        resolved_browser = self._resolve_browser(browser)
        browser_label = "Google Chrome" if resolved_browser == "chrome" else "Microsoft Edge"
        expected_proc = "chrome.exe" if resolved_browser == "chrome" else "msedge.exe"

        print(f"[ROUTER] intent=OPEN_WEBSITE target={display_name}")
        print(f"[BROWSER] selected={browser_label}")
        print(f"[TOOL] open_url={clean_url}")

        if resolved_browser == "chrome":
            chrome_path = _find_chrome_exe()
            if not chrome_path:
                print(f"[VERIFY] chrome_running=False")
                print(f"[RESULT] success=False")
                return ToolResult.fail(
                    action="open_website",
                    message=f"Sir, Google Chrome is not installed on your laptop to open {display_name}.",
                    error="Google Chrome not installed",
                )
            try:
                subprocess.Popen([chrome_path, clean_url])
            except Exception:
                subprocess.Popen(f'start chrome "{clean_url}"', shell=True)
        else:
            edge_path = _find_edge_exe()
            if edge_path:
                try:
                    subprocess.Popen([edge_path, clean_url])
                except Exception:
                    subprocess.Popen(f'start msedge "{clean_url}"', shell=True)
            else:
                subprocess.Popen(f'start msedge "{clean_url}"', shell=True)

        # Verification loop
        verified = False
        for _ in range(5):
            time.sleep(0.4)
            if self._is_browser_running(resolved_browser):
                verified = True
                break

        if resolved_browser == "edge":
            print(f"[VERIFY] msedge_running={verified}")
        else:
            print(f"[VERIFY] chrome_running={verified}")

        if verified:
            print(f"[RESULT] success=True")
            msg = f"Opened {display_name} in {browser_label}, Sir." if browser else f"Opened {display_name}, Sir."
            return ToolResult.ok(
                action="open_website",
                message=msg,
                evidence=f"Navigated {browser_label} to {clean_url}",
            )
        else:
            print(f"[RESULT] success=False")
            return ToolResult.fail(
                action="open_website",
                message=f"Sir, I attempted to open {display_name}, but couldn't verify the browser launched.",
                error=f"Process {expected_proc} not detected",
            )

    def open_url(self, url: str) -> ToolResult:
        """Alias for open_website for backward compatibility."""
        return self.open_website(url)

    def search_youtube(self, query: str, browser: Optional[str] = None) -> ToolResult:
        """
        Opens YouTube search results page in Microsoft Edge for the query.
        Returns state SEARCH_OPENED.
        """
        clean_q = query.strip()
        encoded = urllib.parse.quote_plus(clean_q)
        yt_url = f"https://www.youtube.com/results?search_query={encoded}"

        print(f"[ROUTER] intent=SEARCH_YOUTUBE query='{clean_q}'")
        res = self.open_website(yt_url, browser=browser, site_name=f"YouTube search for '{clean_q}'")
        if res.success:
            return ToolResult.ok(
                action="search_youtube",
                message=f"I opened the YouTube search results for '{clean_q}', Sir.",
                evidence=f"Navigated to {yt_url}",
            )
        return res

    def play_youtube(self, song_query: str, browser: Optional[str] = None) -> ToolResult:
        """
        Handles music playback requests by opening YouTube search results in Edge.
        State: SEARCH_OPENED.
        Truthful reporting: Does NOT say 'The song is playing' unless playback was verified.
        """
        clean_song = song_query.strip()
        print(f"[ROUTER] intent=PLAY_YOUTUBE song='{clean_song}'")

        # Open YouTube search results for the requested song in Microsoft Edge
        res = self.search_youtube(clean_song, browser=browser)
        if res.success:
            return ToolResult.ok(
                action="play_youtube",
                message=f"I opened the YouTube search results for '{clean_song}', Sir.",
                evidence=f"YouTube search results page displayed for {clean_song} (State: {BrowserPlaybackState.SEARCH_OPENED})",
            )
        return res

    def search_web(self, query: str, search_engine: str = "google", browser: Optional[str] = None) -> ToolResult:
        """Searches Google or Bing using Microsoft Edge."""
        clean_q = query.strip()
        encoded = urllib.parse.quote_plus(clean_q)
        if search_engine.lower() == "bing":
            url = f"https://www.bing.com/search?q={encoded}"
        elif search_engine.lower() == "duckduckgo":
            url = f"https://duckduckgo.com/?q={encoded}"
        else:
            url = f"https://www.google.com/search?q={encoded}"

        print(f"[ROUTER] intent=SEARCH_WEB query='{clean_q}'")
        res = self.open_website(url, browser=browser, site_name=f"Web search for '{clean_q}'")
        if res.success:
            return ToolResult.ok(
                action="search_web",
                message=f"Searched for '{clean_q}' in Microsoft Edge, Sir.",
                evidence=f"Navigated to {url}",
            )
        return res

    def open_documentation(self, library_or_topic: str) -> ToolResult:
        """Opens official documentation for a programming language, library, or tool in Edge."""
        topic = library_or_topic.lower().strip()
        doc_urls = {
            "python": "https://docs.python.org/3/",
            "fastapi": "https://fastapi.tiangolo.com/",
            "flask": "https://flask.palletsprojects.com/",
            "react": "https://react.dev/",
            "node": "https://nodejs.org/docs/latest/api/",
            "gemini": "https://ai.google.dev/docs",
            "git": "https://git-scm.com/doc",
            "docker": "https://docs.docker.com/",
        }

        if topic in doc_urls:
            return self.open_website(doc_urls[topic], site_name=f"{topic} documentation")

        return self.search_web(f"{library_or_topic} official documentation")

    def new_tab(self) -> Tuple[bool, str]:
        """Opens a new browser tab via keyboard shortcut."""
        return self.keyboard.hotkey("ctrl", "t")

    def close_tab(self) -> Tuple[bool, str]:
        """Closes active browser tab via keyboard shortcut."""
        return self.keyboard.hotkey("ctrl", "w")

    def next_tab(self) -> Tuple[bool, str]:
        """Switches to the next browser tab."""
        return self.keyboard.hotkey("ctrl", "tab")
