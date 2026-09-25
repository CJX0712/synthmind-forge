# test_loop.py — offline self-test (zero third-party deps required)
# Author: 晨星 (CJX0712)
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthmind import RCGNAS  # noqa: E402


def test_closed_loop_runs_offline():
    eng = RCGNAS(seed=1790362095)
    rep = eng.run("unit-test task", param_budget=40000, iterations=5, pop_size=3, target_acc=0.99)
    assert rep is not None, "engine returned no report"
    assert rep["best_eval"]["accuracy"] > 0, "best accuracy must be positive"
    assert rep["generations"] >= 1, "must run at least one generation"
    assert len(rep["trace"]) == rep["generations"], "trace length must equal generations"
    # memory must accumulate critiques (RAG half present)
    assert rep["memory_entries"] >= 1, "memory must record critiques"
    print("PASS closed_loop_runs_offline")


def test_refinement_improves_or_holds():
    eng = RCGNAS(seed=42)
    rep = eng.run("improve task", param_budget=30000, iterations=8, pop_size=4, target_acc=0.5)
    # proxy is smooth & bounded; after several generations best must be non-trivial
    assert rep["best_eval"]["accuracy"] >= 0.4, f"weak search: {rep['best_eval']['accuracy']}"
    print(f"PASS refinement_improves_or_holds (best={rep['best_eval']['accuracy']})")


if __name__ == "__main__":
    test_closed_loop_runs_offline()
    test_refinement_improves_or_holds()
    print("ALL TESTS PASSED")
