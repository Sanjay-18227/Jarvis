"""
JARVIS Browser Controller
Controls web browsing, searches, documentation lookups, and browser navigation.
"""

import webbrowser
import urllib.parse
from typing import Tuple, Optional
from computer.keyboard_controller import KeyboardController


class BrowserController:
    """
    Manages browser navigation and web research.
    """

    def __init__(self, keyboard: Optional[KeyboardController] = None):
        self.keyboard = keyboard or KeyboardController()

    def open_url(self, url: str) -> Tuple[bool, str]:
        """Opens a specific URL in the default or preferred browser."""
        clean_url = url.strip()
        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            clean_url = f"https://{clean_url}"

        try:
            webbrowser.open(clean_url, new=2)
            return True, f"Navigated to {clean_url}"
        except Exception as e:
            return False, f"Failed opening browser URL: {e}"

    def search_web(self, query: str, search_engine: str = "google") -> Tuple[bool, str]:
        """Searches Google or specified search engine."""
        encoded_query = urllib.parse.quote_plus(query.strip())
        if search_engine.lower() == "bing":
            url = f"https://www.bing.com/search?q={encoded_query}"
        elif search_engine.lower() == "duckduckgo":
            url = f"https://duckduckgo.com/?q={encoded_query}"
        else:
            url = f"https://www.google.com/search?q={encoded_query}"

        return self.open_url(url)

    def open_documentation(self, library_or_topic: str) -> Tuple[bool, str]:
        """Opens official documentation for a programming language, library, or tool."""
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
            return self.open_url(doc_urls[topic])

        # Default to web search for docs
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
