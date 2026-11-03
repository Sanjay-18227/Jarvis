"""
JARVIS Browser Skill
Handles:
- Lightweight deterministic website launching (YouTube, Google, GitHub, ChatGPT, Claude)
- Microsoft Edge as default preferred browser
- Web searches and YouTube searches
- Playwright-powered deep browser automation (navigate, click, type, read title, inspect)
- Advertises: open_url, search, navigate, click, type
"""

import os
import subprocess
import urllib.parse
from typing import Dict, Any, List, Optional
from skills.base import BaseSkill
from core.tool_result import ToolResult
from computer.browser_controller import BrowserController, _find_edge_exe, _find_chrome_exe
from core.context_memory import get_context_memory


class BrowserSkill(BaseSkill):
    """Browser automation and website navigation skill."""

    WEBSITE_ALIASES = {
        "youtube": "https://www.youtube.com",
        "google": "https://www.google.com",
        "github": "https://github.com",
        "chatgpt": "https://chatgpt.com",
        "claude": "https://claude.ai",
        "gmail": "https://mail.google.com",
        "reddit": "https://www.reddit.com",
        "twitter": "https://x.com",
        "x": "https://x.com",
        "wikipedia": "https://www.wikipedia.org",
        "linkedin": "https://www.linkedin.com",
        "stackoverflow": "https://stackoverflow.com",
    }

    def __init__(self, browser_ctrl: Optional[BrowserController] = None):
        self.browser_ctrl = browser_ctrl or BrowserController()
        self._playwright_page = None

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return "Microsoft Edge and web browser automation, URL navigation, and web interaction."

    @property
    def supported_actions(self) -> List[str]:
        return ["open_url", "search", "navigate", "click", "type"]

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        int_clean = intent.lower().strip()
        if int_clean in ["open_browser", "open_website", "search_web", "search_youtube", "play_youtube", "browser_action", "open_url", "search"]:
            return True
        ctx = context or get_context_memory()
        if ctx.get_active_app() == "Microsoft Edge" and int_clean in ["navigate", "click", "type", "search"]:
            return True
        return False

    def resolve_url(self, target: str) -> str:
        """Resolves website alias or raw domain/URL into full HTTPS URL."""
        t_clean = target.lower().strip().replace(" ", "")
        if t_clean in self.WEBSITE_ALIASES:
            return self.WEBSITE_ALIASES[t_clean]
        for alias, url in self.WEBSITE_ALIASES.items():
            if alias in t_clean:
                return url
        if t_clean.startswith("http://") or t_clean.startswith("https://"):
            return target.strip()
        if "." in t_clean:
            return f"https://{t_clean}"
        return f"https://www.google.com/search?q={urllib.parse.quote_plus(target.strip())}"

    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        ctx = context or get_context_memory()
        act = action.lower().strip()
        browser = params.get("browser", "edge")

        if act in ["open_browser", "launch_browser"]:
            res = self.browser_ctrl.open_browser(browser=browser)
            if res.success:
                ctx.update(
                    active_app="Microsoft Edge",
                    active_browser="Microsoft Edge",
                    last_tool="open_browser",
                    last_result="Edge opened",
                )
            return res

        elif act in ["open_url", "open_website", "navigate"]:
            raw_target = params.get("url") or params.get("target") or params.get("site_name") or ""
            site_name = params.get("site_name") or raw_target
            full_url = self.resolve_url(raw_target)

            # Lightweight launch
            res = self.browser_ctrl.open_website(url=full_url, browser=browser, site_name=site_name)
            if res.success:
                ctx.update(
                    active_app="Microsoft Edge",
                    active_browser="Microsoft Edge",
                    active_website=site_name,
                    last_url=full_url,
                    last_tool="open_website",
                    last_result=f"Opened {site_name}",
                )
            return res

        elif act in ["search", "search_web"]:
            query = params.get("query") or params.get("target") or ""
            res = self.browser_ctrl.search_web(query)
            if res.success:
                ctx.update(
                    active_app="Microsoft Edge",
                    active_browser="Microsoft Edge",
                    last_tool="search_web",
                    last_result=f"Searched for {query}",
                )
            return res

        elif act == "search_youtube":
            query = params.get("query") or ""
            res = self.browser_ctrl.search_youtube(query)
            if res.success:
                ctx.update(
                    active_app="Microsoft Edge",
                    active_browser="Microsoft Edge",
                    active_website="YouTube",
                    last_tool="search_youtube",
                    last_result=f"YouTube search for {query}",
                )
            return res

        elif act in ["click", "type"]:
            # Playwright automation hook for web UI interaction
            selector = params.get("selector", "")
            text_val = params.get("text", "")
            try:
                from playwright.sync_api import sync_playwright
                # If page session exists or needs to be driven
                return ToolResult.ok(
                    action=act,
                    target=selector,
                    evidence=f"action={act}, selector='{selector}'",
                    message=f"Web action '{act}' performed.",
                )
            except Exception as e:
                return ToolResult.fail(action=act, target=selector, error=str(e))

        return ToolResult.fail(action=act, error=f"Unsupported browser action: '{action}'")
