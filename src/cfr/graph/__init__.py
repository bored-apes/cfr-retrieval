"""LangGraph port of the CFR retrieval pipeline.

The hand-rolled version in `cfr.answer` remains the baseline; this exists so the
two can be measured against each other rather than assumed equivalent.
"""

from .build import build_graph, render_mermaid  # noqa: F401
from .state import CFRState  # noqa: F401
