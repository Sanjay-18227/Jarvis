"""
JARVIS Core Package
"""

def __getattr__(name: str):
    if name == "JarvisAgent":
        from .agent import JarvisAgent
        return JarvisAgent
    if name in ["Phase3AgentLoop", "AgentState", "StructuredPlan", "PlanStep"]:
        from .agent_loop import Phase3AgentLoop, AgentState, StructuredPlan, PlanStep
        mapping = {
            "Phase3AgentLoop": Phase3AgentLoop,
            "AgentState": AgentState,
            "StructuredPlan": StructuredPlan,
            "PlanStep": PlanStep,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["JarvisAgent", "Phase3AgentLoop", "AgentState", "StructuredPlan", "PlanStep"]


