# SynthMind-Forge — RCG-NAS engine
# Author: 晨星 (CJX0712) · Shenzhen
# Recursive Critique-Guided Neural Architecture Search.
#
# Top-tier OSS wired in (optional, graceful fallback to stdlib):
#   - LangGraph   : agent state-machine orchestration pattern
#   - Haystack    : RAG memory over architecture knowledge base
#   - ONNX Runtime: compile + latency-benchmark of candidate nets
#   - Diffusers   : architecture-latent diffusion prior (novel hook)
#
# The system forms a CLOSED LOOP even with zero third-party deps:
# the offline path uses a deterministic proxy evaluator + in-memory
# nearest-neighbour memory so `python -m synthmind` always runs end-to-end.

__version__ = "0.1.0"
__author__ = "晨星 (CJX0712)"

from .engine import RCGNAS
from .search_space import random_arch, arch_to_features, count_params
from .evaluator import evaluate, ONNXBackend
from .memory import MemoryStore, LocalMemory
from .critic import critique
from .orchestrator import StateGraph, Node

__all__ = [
    "RCGNAS",
    "random_arch",
    "arch_to_features",
    "count_params",
    "evaluate",
    "ONNXBackend",
    "MemoryStore",
    "LocalMemory",
    "critique",
    "StateGraph",
    "Node",
]
