"""
JARVIS Tool Planner & Declarative Tool System
Declares modular computer tools with strict permissions, risk levels, and schemas
for Gemini function calling and autonomous reasoning.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from config.constants import RiskLevel, SpeakerClassification


@dataclass
class ToolDefinition:
    """
    Standard tool declaration conforming to JARVIS Section 12 specifications:
    name, description, parameters, permission_level, risk_level, requires_confirmation.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    permission_level: str = "OWNER"
    risk_level: int = 1
    requires_confirmation: bool = False


class ToolCatalog:
    """
    Central registry of extensible computer tools available to JARVIS.
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register_tool(self, tool: ToolDefinition):
        """Registers a new or custom skill tool."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def _register_default_tools(self):
        # 1. Open Application
        self.register_tool(ToolDefinition(
            name="open_application",
            description="Launches or brings into focus a Windows desktop application such as Antigravity, VS Code, Chrome, Edge, Terminal, Calculator, Notepad, or Settings.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Name of the application to open (e.g. 'antigravity', 'vscode', 'chrome', 'calculator', 'terminal')."
                    }
                },
                "required": ["target"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 2. Close Application
        self.register_tool(ToolDefinition(
            name="close_application",
            description="Closes a desktop application by name.",
            parameters={
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Name of the application to close."
                    }
                },
                "required": ["target"]
            },
            permission_level="OWNER",
            risk_level=2,
            requires_confirmation=False,
        ))

        # 3. Focus Window
        self.register_tool(ToolDefinition(
            name="focus_window",
            description="Switches focus and brings a specific application window to the foreground by title keyword.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title keyword of the window (e.g. 'Antigravity', 'Visual Studio Code', 'Edge')."
                    }
                },
                "required": ["title"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 4. List Windows
        self.register_tool(ToolDefinition(
            name="list_windows",
            description="Lists all visible desktop application windows currently open on the computer.",
            parameters={
                "type": "object",
                "properties": {}
            },
            permission_level="OWNER",
            risk_level=0,
            requires_confirmation=False,
        ))

        # 5. Screen Vision
        self.register_tool(ToolDefinition(
            name="inspect_screen",
            description="Captures the computer screen and uses visual reasoning to inspect active applications, code, terminal tracebacks, errors, dialogs, or web pages.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for or analyze on screen (e.g. 'What error is shown in the terminal?', 'Inspect the code in Antigravity')."
                    }
                },
                "required": ["query"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 6. Web Search
        self.register_tool(ToolDefinition(
            name="search_web",
            description="Searches the web for latest documentation, error solutions, libraries, or general questions.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query."
                    }
                },
                "required": ["query"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 7. Open URL
        self.register_tool(ToolDefinition(
            name="open_url",
            description="Opens a specific website URL or documentation page in the browser.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL to open."
                    }
                },
                "required": ["url"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 8. Mouse Action
        self.register_tool(ToolDefinition(
            name="mouse_action",
            description="Controls the mouse: click, double_click, right_click, move, or scroll.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["click", "double_click", "right_click", "move", "scroll"],
                        "description": "Mouse action to perform."
                    },
                    "x": {"type": "integer", "description": "X screen coordinate (optional)."},
                    "y": {"type": "integer", "description": "Y screen coordinate (optional)."},
                    "clicks": {"type": "integer", "description": "Scroll clicks (negative for down, positive for up)."}
                },
                "required": ["action"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 9. Keyboard Action
        self.register_tool(ToolDefinition(
            name="keyboard_action",
            description="Simulates keyboard typing, pressing Enter, or keyboard shortcuts (e.g. ['ctrl', 's'], ['ctrl', 'c'], ['win', 'd']).",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["type", "hotkey", "enter", "esc"],
                        "description": "Keyboard operation."
                    },
                    "text": {"type": "string", "description": "Text to type when action is 'type'."},
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key names for shortcut when action is 'hotkey' (e.g. ['ctrl', 's'])."
                    }
                },
                "required": ["action"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 10. Read File
        self.register_tool(ToolDefinition(
            name="read_file",
            description="Reads the contents of a project file to inspect code, errors, or configuration.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."}
                },
                "required": ["path"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 11. Create or Write File
        self.register_tool(ToolDefinition(
            name="create_file",
            description="Creates a new file or updates an existing file with specified code or text.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "content": {"type": "string", "description": "Code or text content to write."}
                },
                "required": ["path", "content"]
            },
            permission_level="OWNER",
            risk_level=2,
            requires_confirmation=False,
        ))

        # 12. Create Folder
        self.register_tool(ToolDefinition(
            name="create_folder",
            description="Creates a directory for a new project or file organization.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path."}
                },
                "required": ["path"]
            },
            permission_level="OWNER",
            risk_level=2,
            requires_confirmation=False,
        ))

        # 13. List Directory
        self.register_tool(ToolDefinition(
            name="list_directory",
            description="Lists files and subdirectories in a workspace or folder.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to inspect (defaults to current folder)."}
                }
            },
            permission_level="OWNER",
            risk_level=0,
            requires_confirmation=False,
        ))

        # 14. Run Terminal Command
        self.register_tool(ToolDefinition(
            name="run_command",
            description="Runs a development, build, test, or diagnostic command in the terminal (e.g. 'pytest', 'python app.py', 'npm test', 'git status', 'pip install <pkg>').",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command string to execute."},
                    "cwd": {"type": "string", "description": "Optional working directory."}
                },
                "required": ["command"]
            },
            permission_level="OWNER",
            risk_level=2,
            requires_confirmation=False,
        ))

        # 15. Delete Path
        self.register_tool(ToolDefinition(
            name="delete_path",
            description="Deletes a file or directory permanently. REQUIRES EXPLICIT OWNER CONFIRMATION.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to delete."}
                },
                "required": ["path"]
            },
            permission_level="OWNER",
            risk_level=4,
            requires_confirmation=True,
        ))

        # 16. System Control
        self.register_tool(ToolDefinition(
            name="system_control",
            description="Controls system features: lock screen, volume adjustments, or battery checks.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["lock_workstation", "volume_up", "volume_down", "mute", "battery_status"],
                        "description": "System action."
                    }
                },
                "required": ["action"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 17. Laptop Performance Diagnosis (Section 30)
        self.register_tool(ToolDefinition(
            name="diagnose_laptop_performance",
            description="Inspects real CPU, RAM, disk, battery, and top memory-consuming processes to diagnose why the laptop is slow or lagging.",
            parameters={
                "type": "object",
                "properties": {},
            },
            permission_level="OWNER",
            risk_level=0,
            requires_confirmation=False,
        ))

        # 18. AI Web Research (Section 14 & 15)
        self.register_tool(ToolDefinition(
            name="research_web_topic",
            description="Searches live web sources, cross-checks findings, deduplicates info, and synthesizes an intelligent factual summary on any topic or current news.",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Research query or topic to investigate."
                    }
                },
                "required": ["topic"]
            },
            permission_level="OWNER",
            risk_level=1,
            requires_confirmation=False,
        ))

        # 19. Autonomous Project Execution (Section 21 & 22)
        self.register_tool(ToolDefinition(
            name="run_project",
            description="Automatically detects the project framework (FastAPI, Flask, Streamlit, Django, React, Next.js, Vite, Python), determines the correct run command and virtual environment, and executes/verifies the run.",
            parameters={
                "type": "object",
                "properties": {
                    "project_dir": {
                        "type": "string",
                        "description": "Optional path to the project directory. Defaults to current workspace."
                    }
                }
            },
            permission_level="OWNER",
            risk_level=2,
            requires_confirmation=False,
        ))

        # 20. Coding Investigation & Bug Fix (Section 23 & 25)
        self.register_tool(ToolDefinition(
            name="investigate_project",
            description="Investigates a broken project: inspects code files, terminal errors, missing packages, applies safe fixes, reruns, and verifies execution.",
            parameters={
                "type": "object",
                "properties": {
                    "project_dir": {
                        "type": "string",
                        "description": "Optional project path to investigate."
                    },
                    "issue_hint": {
                        "type": "string",
                        "description": "Optional description of what is failing."
                    }
                }
            },
            permission_level="OWNER",
            risk_level=2,
            requires_confirmation=False,
        ))

    def to_gemini_declarations(self) -> List[Dict[str, Any]]:
        """Converts tool catalog into declarations format for Gemini API."""
        declarations = []
        for tool in self.list_tools():
            declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        return declarations
