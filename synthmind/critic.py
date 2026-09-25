# critic.py — self-critique + targeted refinement (the "RCG" half)
# Author: 晨星 (CJX0712)
from __future__ import annotations
import random


def critique(arch: dict, ev: dict, retrieved: list[dict]) -> tuple[list[dict], str]:
    """Produce targeted refinement actions + a natural-language critique.

    This is the Reflexion-style self-critique: instead of only scoring, the
    Critic proposes *edits* to the candidate. Retrieved past critiques bias it
    to avoid repeating known mistakes (RAG-over-memory).
    """
    actions: list[dict] = []
    notes: list[str] = []

    if not ev["under_budget"]:
        actions.append({"op": "shrink_width", "by": 0.6})
        notes.append(f"超参数量预算({ev['params']} params)，需收窄宽度")
    elif ev["accuracy"] < 0.55 and ev["params"] < 15000:
        actions.append({"op": "widen", "to": 96})
        notes.append("容量不足，准确率偏低，建议加宽")

    has_drop = any(b["kind"] == "dropout" for b in arch.get("blocks", []))
    if ev["accuracy"] < 0.7 and not has_drop:
        actions.append({"op": "add_dropout", "p": 0.2})
        notes.append("缺少正则化(dropout)，存在过拟合风险")

    for b in arch.get("blocks", []):
        if b.get("act") == "tanh":
            actions.append({"op": "switch_act", "from": "tanh", "to": "relu"})
            notes.append("tanh 在小网络易饱和，建议改用 relu")

    for r in retrieved:
        if r["score"] < 0.5 and "dropout" in r["text"]:
            if not has_drop:
                actions.append({"op": "add_dropout", "p": 0.25})
                notes.append(f"记忆库提示类似架构曾失败：{r['text'][:40]}…")

    if not actions:
        actions.append({"op": "keep"})
        notes.append("当前候选已满足约束，保留")
    return actions, "；".join(notes) or "无显著问题"


def refine_arch(arch: dict, actions: list[dict], rng: random.Random) -> dict:
    """Apply critic actions to produce the next candidate (immutable copy)."""
    import copy
    new = copy.deepcopy(arch)
    for a in actions:
        op = a["op"]
        if op == "shrink_width":
            f = a.get("by", 0.6)
            for b in new["blocks"]:
                if b["kind"] == "linear":
                    b["width"] = max(8, int(b["width"] * f))
            new["stem_width"] = max(4, int(new["stem_width"] * f))
        elif op == "widen":
            for b in new["blocks"]:
                if b["kind"] == "linear":
                    b["width"] = a.get("to", 96)
                    break
        elif op == "add_dropout":
            p = a.get("p", 0.2)
            inserted = False
            for i, b in enumerate(new["blocks"]):
                if b["kind"] == "linear":
                    new["blocks"].insert(i + 1, {"kind": "dropout", "p": p})
                    inserted = True
                    break
            if not inserted:
                new["blocks"].append({"kind": "dropout", "p": p})
        elif op == "switch_act":
            for b in new["blocks"]:
                if b.get("act") == a.get("from"):
                    b["act"] = a.get("to", "relu")
        elif op == "keep":
            pass
    new["name"] = f"{arch.get('name','c')}_r"
    return new
