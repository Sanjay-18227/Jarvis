"""
JARVIS Calculator Skill
Handles:
- Safe local arithmetic evaluation (no unconstrained eval, sub-millisecond execution)
- Windows Calculator app automation when Calculator is active or requested
- Advertises: calculate, clear
"""

import re
import ast
import operator
from typing import Dict, Any, List, Optional, Union
from skills.base import BaseSkill
from core.tool_result import ToolResult
from computer.calculator_controller import CalculatorController
from core.context_memory import get_context_memory


# Safe arithmetic operators mapping
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_ast(node: ast.AST) -> Union[int, float]:
    """Recursively evaluates safe mathematical AST expressions without unrestricted eval."""
    if isinstance(node, ast.Expression):
        return safe_eval_ast(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            operand = safe_eval_ast(node.operand)
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            left = safe_eval_ast(node.left)
            right = safe_eval_ast(node.right)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division by zero")
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")
    raise ValueError(f"Unsupported AST node: {type(node)}")


class CalculatorSkill(BaseSkill):
    """Local arithmetic and Windows Calculator automation skill."""

    def __init__(self, calc_ctrl: Optional[CalculatorController] = None):
        self.calc_ctrl = calc_ctrl or CalculatorController()

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Safe local arithmetic calculation and Windows Calculator automation."

    @property
    def supported_actions(self) -> List[str]:
        return ["calculate", "clear"]

    def can_handle(self, intent: str, params: Optional[Dict[str, Any]] = None, context: Optional[Any] = None) -> bool:
        int_clean = intent.lower().strip()
        if int_clean in ["calculate", "arithmetic", "math", "clear"]:
            return True
        ctx = context or get_context_memory()
        if ctx.get_active_app() == "Calculator" and int_clean in ["calculate", "add", "subtract", "multiply", "divide"]:
            return True
        return False

    @staticmethod
    def parse_and_evaluate(expression: str) -> Optional[Union[int, float]]:
        """Cleans and evaluates mathematical expressions using safe AST parser."""
        if not expression:
            return None
        expr = expression.strip()
        # Word replacements
        expr = re.sub(r"\bminus\b", "-", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bplus\b", "+", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\btimes\b", "*", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bdivided\s+by\b", "/", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bby\b", "/", expr, flags=re.IGNORECASE)
        expr = re.sub(r"\bx\b", "*", expr, flags=re.IGNORECASE)

        # Word numbers replacement
        word_nums = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        }
        for w, d in word_nums.items():
            expr = re.sub(rf"\b{w}\b", d, expr, flags=re.IGNORECASE)

        # Strip everything except valid math characters
        clean = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s]", "", expr).strip()
        if not clean:
            return None

        try:
            tree = ast.parse(clean, mode="eval")
            res = safe_eval_ast(tree)
            if isinstance(res, float) and res.is_integer():
                return int(res)
            return res
        except Exception as e:
            print(f"[CALCULATOR] AST evaluation error for '{clean}': {e}")
            return None

    def execute(self, action: str, params: Dict[str, Any], context: Optional[Any] = None) -> ToolResult:
        ctx = context or get_context_memory()
        act = action.lower().strip()

        if act == "clear":
            hwnd = self.calc_ctrl.find_calculator_hwnd()
            if hwnd:
                import win32api, win32con
                win32api.keybd_event(0x1B, 0, 0, 0)  # ESC clears calculator
                win32api.keybd_event(0x1B, 0, win32con.KEYEVENTF_KEYUP, 0)
            ctx.update(last_calculation=None)
            return ToolResult.ok(action="clear", target="Calculator", message="Calculator cleared, Sir.")

        # action == "calculate"
        expression = params.get("expression") or params.get("query") or ""
        use_calc_app = params.get("use_calculator_app", False)
        if ctx.get_active_app() == "Calculator":
            use_calc_app = True

        result = self.parse_and_evaluate(expression)
        if result is None:
            return ToolResult.fail(
                action="calculate",
                target=expression,
                error=f"Could not safely parse expression: '{expression}'",
            )

        # If user is in Calculator or asked to use it: execute on Windows Calculator UI
        if use_calc_app:
            ok, msg, uia_res = self.calc_ctrl.execute_calculation_on_calculator(str(expression))
            evidence = f"expression='{expression}', display_result='{uia_res}', calculated='{result}'"
            ctx.update(active_app="Calculator", last_calculation=str(result), last_tool="calculate", last_result=str(result))
            return ToolResult.ok(
                action="calculate",
                target="CalculatorApp",
                evidence=evidence,
                message=f"{expression} is {result}, Sir.",
            )

        # Safe local execution
        evidence = f"expression='{expression}', safe_ast_result={result}"
        ctx.update(last_calculation=str(result), last_tool="calculate", last_result=str(result))
        return ToolResult.ok(
            action="calculate",
            target="local_math",
            evidence=evidence,
            message=f"{expression} is {result}, Sir.",
        )
