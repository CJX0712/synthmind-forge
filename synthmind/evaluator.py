# evaluator.py — candidate scoring: offline proxy + ONNX Runtime real benchmark
# Author: 晨星 (CJX0712)
from __future__ import annotations
import math
import time

from .search_space import count_params


def _proxy_accuracy(arch: dict, param_budget: int) -> float:
    """Deterministic, smooth, monotonic-friendly quality proxy in [0, 1].

    Encodes domain priors a good NAS should discover:
      * capacity helps up to a budget, then overfits,
      * dropout regularises,
      * relu/gelu beat saturating acts at small scale.
    """
    p = count_params(arch)
    # capacity curve: tanh-sigmoid around budget (spans 0..1), slight overfit decay beyond 2x.
    cap = 0.5 + 0.5 * math.tanh((math.log(p + 1) - math.log(param_budget)) * 0.8)
    overfit = max(0.0, (p - param_budget * 2.0) / (param_budget * 4.0)) * 0.15
    cap = max(0.0, min(1.0, cap - overfit))

    reg = 0.0
    for b in arch.get("blocks", []):
        if b["kind"] == "dropout":
            reg += 0.06 * (b["p"] / 0.2)
    act_bonus = 0.0
    for b in arch.get("blocks", []):
        if b.get("act") in ("relu", "gelu"):
            act_bonus = max(act_bonus, 0.04)
    depth = len([b for b in arch.get("blocks", []) if b["kind"] == "linear"])
    depth_pen = 0.02 * max(0, depth - 3)
    return max(0.0, min(1.0, cap + reg + act_bonus - depth_pen))


def evaluate(arch: dict, param_budget: int = 50000, backend: "ONNXBackend | None" = None) -> dict:
    """Return a scored evaluation record for one candidate architecture."""
    p = count_params(arch)
    acc = _proxy_accuracy(arch, param_budget)
    latency = p / 1000.0  # ms, cheap stand-in
    if backend is not None and backend.available:
        real = backend.benchmark(arch)
        if real is not None:
            latency = real
    return {
        "name": arch.get("name"),
        "params": p,
        "accuracy": round(acc, 4),
        "latency_ms": round(latency, 4),
        "under_budget": p <= param_budget,
    }


class ONNXBackend:
    """Real inference benchmark via ONNX Runtime. Optional dependency.

    Wires in the top-tier OSS *ONNX Runtime* so candidate nets are compiled
    to ONNX and latency-profiled for real instead of proxy-estimated.
    Degrades to ``available=False`` when ``onnx`` / ``onnxruntime`` are absent.
    """

    def __init__(self) -> None:
        self.available = False
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort  # noqa: F401
            self._onnx = onnx
            self._ort = ort
            self.available = True
        except Exception:
            self.available = False

    def benchmark(self, arch: dict, bs: int = 1, in_dim: int = 32, reps: int = 20) -> float | None:
        if not self.available:
            return None
        try:
            onnx, ort = self._onnx, self._ort
            import numpy as np
            nodes, inits, inp = [], [], in_dim
            for i, b in enumerate(arch.get("blocks", [])):
                if b["kind"] != "linear":
                    continue
                w = np.random.randn(inp, b["width"]).astype("float32")
                c = np.zeros(b["width"], dtype="float32")
                inits.append(onnx.numpy_helper.from_array(w, f"W{i}"))
                inits.append(onnx.numpy_helper.from_array(c, f"B{i}"))
                nodes.append(onnx.helper.make_node("Gemm", [f"x{i}", f"W{i}", f"B{i}"], [f"g{i}"], name=f"gemm{i}"))
                if b.get("act") in ("relu", "gelu"):
                    nodes.append(onnx.helper.make_node("Relu", [f"g{i}"], [f"x{i+1}"], name=f"act{i}"))
                else:
                    nodes.append(onnx.helper.make_node("Tanh", [f"g{i}"], [f"x{i+1}"], name=f"act{i}"))
                inp = b["width"]
            graph = onnx.helper.make_graph(
                nodes, "net", [onnx.helper.make_tensor_value_info("x0", onnx.TensorProto.FLOAT, [bs, in_dim])],
                [onnx.helper.make_tensor_value_info(f"x{inp and len([b for b in arch['blocks'] if b['kind']=='linear'])}", onnx.TensorProto.FLOAT, [bs, inp])],
                inits,
            )
            model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
            sess = ort.InferenceSession(model.SerializeToString())
            x = np.random.randn(bs, in_dim).astype("float32")
            t0 = time.perf_counter()
            for _ in range(reps):
                sess.run(None, {sess.get_inputs()[0].name: x})
            return (time.perf_counter() - t0) / reps * 1000.0
        except Exception:
            return None
