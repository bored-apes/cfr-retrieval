"""Supervisor-routed multi-agent variant of the CFR pipeline.

Three siblings now exist, deliberately:
  cfr.answer   imperative baseline
  cfr.graph    linear state graph
  cfr.agents   supervisor + specialists with routing authority
"""

from .build import build_agent_graph, render_mermaid  # noqa: F401
from .state import AgentState  # noqa: F401
