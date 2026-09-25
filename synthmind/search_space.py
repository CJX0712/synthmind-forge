# search_space.py — candidate architecture spec + feature embedding
# Author: 晨星 (CJX0712)
"""A candidate network is a plain dict so it is JSON-serialisable and ONNX-exportable.

Example arch::

    {
      "name": "c0",
      "stem_width": 16,
      "blocks": [
        {"kind": "linear", "width": 64, "act": "relu"},
        {"kind": "dropout", "p": 0.2},
        {"kind": "linear", "width": 32, "act": "relu"},
      ],
      "head_width": 10,
    }
"""
from __future__ import annotations
import math
import random

ACTIVATIONS = ["relu", "gelu", "tanh", "silu"]


def count_params(arch: dict) -> int:
    """Count trainable params for a linear-chain spec (in -> blocks -> head)."""
    total = 0
    in_w = arch.get("stem_width", 16)
    for b in arch.get("blocks", []):
        if b["kind"] == "linear":
            w = b["width"]
            total += in_w * w + w          # weight + bias
            in_w = w
        elif b["kind"] == "conv":
            c = b["out_channels"]
            k = b["kernel"]
            cin = b.get("in_channels", in_w)
            total += cin * c * k * k + c
            in_w = c
    total += in_w * arch.get("head_width", 10) + arch.get("head_width", 10)
    return total


def random_arch(rng: random.Random, name: str = "c0", max_blocks: int = 4) -> dict:
    depth = rng.randint(1, max_blocks)
    blocks = []
    for _ in range(depth):
        if rng.random() < 0.25:
            blocks.append({"kind": "dropout", "p": round(rng.uniform(0.05, 0.4), 2)})
        else:
            blocks.append({
                "kind": "linear",
                "width": rng.choice([16, 32, 48, 64, 96, 128]),
                "act": rng.choice(ACTIVATIONS),
            })
    return {
        "name": name,
        "stem_width": rng.choice([8, 16, 32]),
        "blocks": blocks,
        "head_width": rng.choice([4, 8, 10, 16]),
    }


def arch_to_features(arch: dict) -> list[float]:
    """Bag-of-features vector used by the offline memory (cosine NN)."""
    feats = [arch.get("stem_width", 16) / 32.0, len(arch.get("blocks", [])) / 4.0]
    has_drop = any(b["kind"] == "dropout" for b in arch.get("blocks", []))
    feats.append(1.0 if has_drop else 0.0)
    widths = [b["width"] for b in arch.get("blocks", []) if b["kind"] == "linear"]
    feats.append((sum(widths) / len(widths)) / 128.0 if widths else 0.0)
    feats.append(arch.get("head_width", 10) / 16.0)
    for a in ACTIVATIONS:
        feats.append(1.0 if any(b.get("act") == a for b in arch.get("blocks", [])) else 0.0)
    return feats


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(y * y for y in b)) or 1e-9
    return dot / (na * nb)
