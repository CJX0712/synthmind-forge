# cli.py — command line entrypoint
# Author: 晨星 (CJX0712)
from __future__ import annotations
import argparse
import json
import sys

from .engine import RCGNAS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="synthmind", description="RCG-NAS: recursive critique-guided neural architecture search")
    p.add_argument("--task", default="small image classifier")
    p.add_argument("--budget", type=int, default=50000, help="param budget")
    p.add_argument("--iters", type=int, default=6, help="search generations")
    p.add_argument("--pop", type=int, default=4, help="population per generation")
    p.add_argument("--target", type=float, default=0.85, help="target accuracy proxy")
    p.add_argument("--seed", type=int, default=1790362095)
    p.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = p.parse_args(argv)

    eng = RCGNAS(seed=args.seed)
    rep = eng.run(args.task, args.budget, args.iters, args.pop, args.target)

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    print(f"# SynthMind-Forge · RCG-NAS report")
    print(f"task         : {args.task}")
    print(f"generations  : {rep['generations']}")
    print(f"memory entries: {rep['memory_entries']}")
    print(f"top-tier OSS : ONNX={rep['used_onnx']} Haystack={rep['used_haystack']} Diffusers={rep['used_diffusers']}")
    print(f"best accuracy: {rep['best_eval']['accuracy']}  params={rep['best_eval']['params']}  latency={rep['best_eval']['latency_ms']}ms")
    print(f"\n## best architecture\n```json\n{json.dumps(rep['best_arch'], ensure_ascii=False, indent=2)}\n```")
    print("\n## evolution trace")
    for t in rep["trace"]:
        print(f"  gen{t['gen']:>2} {t['name']:<10} acc={t['ev']['accuracy']:.3f} | {t['critique']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
