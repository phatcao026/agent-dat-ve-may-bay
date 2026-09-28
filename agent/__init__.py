"""
Package agent - Chứa 3 mẫu thiết kế Agentic AI cho BTVN#3.
- ReAct Agent (react_agent.py)
- Plan-then-Execute Agent (plan_execute_agent.py)
- Hybrid Agent (hybrid_agent.py)
"""

from agent.react_agent import run_react_agent
from agent.plan_execute_agent import run_plan_execute_agent
from agent.hybrid_agent import run_hybrid_agent
from agent.llm_setup import get_llm

__all__ = [
    "run_react_agent",
    "run_plan_execute_agent",
    "run_hybrid_agent",
    "get_llm"
]
