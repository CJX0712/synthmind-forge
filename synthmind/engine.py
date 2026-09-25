# engine.py — RCG-NAS: the top-level recursive critique-guided search engine
# Author: 晨星 (CJX0712)
from __future__ import annotations
import random

from .search_space import random_arch
from .evaluator import evaluate, ONNXBackend
from .memory import MemoryStore
from .critic import critique, refine_arch
from .orchestrator import StateGraph, Node
from .diffusion_prior import DiffusionPrior


class RCGNAS:
    """Recursive Critique-Guided Neural Architecture Search.

    Innovation (RCG-NAS): a Reflexion-style self-critique loop over a NAS
    population, with a RAG memory of past critiques so the Planner avoids
    repeating failed architectures. Orchestrated as a typed LangGraph-style
    state machine. Runs end-to-end with zero third-party deps; upgrades to
    ONNX Runtime / Haystack / Diffusers automatically when installed.
    """

    def __init__(self, seed: int = 1790362095) -> None:
        self.rng = random.Random(seed)
        self.memory = MemoryStore()
        self.backend = ONNXBackend()
        self.prior = DiffusionPrior()

    # ---- node implementations (closures over self) -------------------------
    def _planner(self, s: dict) -> dict:
        if s["gen"] == 0:
            seeds = self.prior.seed_population(self.rng, s["goal"]["pop_size"])
            pop = []
            for i, sd in enumerate(seeds):
                r = random.Random(sd)
                pop.append(random_arch(r, name=f"g0_{i}"))
            s["population"] = pop
        return s

    def _evaluator(self, s: dict) -> dict:
        budget = s["goal"]["param_budget"]
        for arch in s["population"]:
            ev = evaluate(arch, budget, self.backend)
            s["last_eval"] = (arch, ev)
            if ev["accuracy"] > s["best"]["ev"]["accuracy"]:
                s["best"] = {"arch": arch, "ev": ev}
        return s

    def _critic(self, s: dict) -> dict:
        arch, ev = s["best"]["arch"], s["best"]["ev"]
        retrieved = self.memory.query(arch, k=3)
        actions, text = critique(arch, ev, retrieved)
        self.memory.add(arch, text or "ok", ev["accuracy"])
        s["pending"] = actions
        s["trace"].append({
            "gen": s["gen"], "name": arch.get("name"), "ev": ev, "critique": text,
        })
        return s

    def _refiner(self, s: dict) -> dict:
        s["gen"] += 1
        base = s["best"]["arch"]
        refined = refine_arch(base, s["pending"] or [{"op": "keep"}], self.rng)
        refined["name"] = f"g{s['gen']}_best"
        pop = [refined]
        for i in range(s["goal"]["pop_size"] - 1):
            pop.append(random_arch(self.rng, name=f"g{s['gen']}_{i}"))
        s["population"] = pop
        return s

    def _synthesize(self, s: dict) -> dict:
        s["report"] = {
            "best_arch": s["best"]["arch"],
            "best_eval": s["best"]["ev"],
            "generations": s["gen"] + 1,
            "memory_entries": len(self.memory),
            "used_onnx": self.backend.available,
            "used_haystack": getattr(self.memory, "using_haystack", False),
            "used_diffusers": self.prior.available,
            "trace": s["trace"],
        }
        return s

    def _router(self, s: dict) -> str:
        goal = s["goal"]
        cont = (s["gen"] < goal["iterations"] - 1) and (s["best"]["ev"]["accuracy"] < goal["target_acc"])
        return "refiner" if cont else "synthesizer"

    # ---- public API --------------------------------------------------------
    def run(self, task: str = "small image classifier", param_budget: int = 50000,
            iterations: int = 6, pop_size: int = 4, target_acc: float = 0.85) -> dict:
        goal = {"task": task, "param_budget": param_budget, "iterations": iterations,
                "pop_size": pop_size, "target_acc": target_acc}
        state = {
            "goal": goal, "gen": 0, "population": [], "best": {"arch": {}, "ev": {"accuracy": -1.0}},
            "pending": None, "trace": [], "report": None, "last_eval": None,
        }
        graph = (
            StateGraph()
            .add_node(Node("planner", self._planner))
            .add_node(Node("evaluator", self._evaluator))
            .add_node(Node("critic", self._critic))
            .add_node(Node("refiner", self._refiner))
            .add_node(Node("synthesizer", self._synthesize))
            .add_edge("planner", "evaluator")
            .add_edge("evaluator", "critic")
            .add_conditional_edges("critic", self._router)
            .add_edge("refiner", "evaluator")
        )
        graph.invoke(state)
        return state["report"]
