# orchestrator.py — LangGraph-style typed state machine (minimal, dependency-free)
# Author: 晨星 (CJX0712)
from __future__ import annotations
from typing import Callable, Dict, List


class Node:
    """A single step in the RCG-NAS loop. Mirrors a LangGraph node: (state) -> state."""

    def __init__(self, name: str, fn: Callable[[dict], dict]) -> None:
        self.name = name
        self.fn = fn

    def __call__(self, state: dict) -> dict:
        return self.fn(state)


class StateGraph:
    """Minimal re-implementation of LangGraph's ``StateGraph`` contract.

    Top-tier OSS *LangGraph* is the reference pattern for agent orchestration;
    this class reproduces its essential surface (``add_node``, ``add_edge``,
    ``add_conditional_edges``, ``compile``, ``invoke``) with zero deps so the
    loop always runs. If ``langgraph`` is installed the engine can swap to the
    real graph without changing node code.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, str] = {}
        self.cond: Dict[str, Callable[[dict], str]] = {}
        self.entry: str | None = None

    def add_node(self, node: Node) -> "StateGraph":
        self.nodes[node.name] = node
        if self.entry is None:
            self.entry = node.name
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: Callable[[dict], str]) -> "StateGraph":
        self.cond[src] = router
        return self

    def compile(self) -> "CompiledGraph":
        return CompiledGraph(self)

    def invoke(self, state: dict) -> dict:
        return self.compile().invoke(state)


class CompiledGraph:
    def __init__(self, g: StateGraph) -> None:
        self.g = g

    def invoke(self, state: dict) -> dict:
        if self.g.entry is None:
            return state
        cur = self.g.entry
        guard = 0
        while guard < 256:
            guard += 1
            state = self.g.nodes[cur](state)
            if cur in self.g.cond:
                nxt = self.g.cond[cur](state)
                if nxt == "__end__":
                    break
                cur = nxt
            elif cur in self.g.edges:
                cur = self.g.edges[cur]
                if cur == "__end__":
                    break
            else:
                break
        return state
