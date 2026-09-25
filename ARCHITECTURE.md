# 架构详解 · SynthMind-Forge (RCG-NAS)

> 系统架构师「架凌云」出品 · 引擎分解 / 数据流 / 决策锁校验

## 1. 引擎分解（模块边界）

```
                 ┌─────────────────────────────────────────────┐
   goal ───────► │              RCGNAS.engine.run()              │
                 │   ┌──────────┐   LangGraph 式 StateGraph      │
                 │   │ Planner  │  seed population (Diffusers 先验)│
                 │   └────┬─────┘                                │
                 │        ▼                                      │
                 │   ┌──────────┐   evaluate()                  │
                 │   │Evaluator │── ONNX Runtime | 离线代理       │
                 │   └────┬─────┘                                │
                 │        ▼                                      │
                 │   ┌──────────┐   critique() + Memory.query()  │
                 │   │ Critic   │── Haystack RAG | 本地 NN        │
                 │   └────┬─────┘                                │
                 │        │ conditional: acc<target & gen<max?   │
                 │        ├── Yes ─► Refiner(refine_arch) ──┐     │
                 │        └── No  ─► Synthesizer(report) ◄──┘     │
                 └─────────────────────────────────────────────┘
```

| 模块 | 职责 | 顶级库接入 | 降级 |
|------|------|-----------|------|
| `orchestrator.StateGraph` | 类型化状态机，定节点/边/条件边 | LangGraph（参考契约） | 零依赖自实现 |
| `search_space` | 候选架构 dict + 特征嵌入 | — | 纯标准库 |
| `evaluator` | 评分（容量/正则/激活先验） | ONNX Runtime（真实延迟） | 解析代理 |
| `memory` | 历史批判记忆 + 检索 | Haystack（BM25 RAG） | 本地余弦 NN |
| `critic` | 自批判 + 定向修改 | — | 纯规则 |
| `diffusion_prior` | 初始种群先验 | Diffusers | 可复现随机 |

## 2. 数据流（单次迭代）

1. `Planner`：gen=0 用 Diffusers 先验种子生成 `pop_size` 个候选；后续代由 `Refiner` 产出。
2. `Evaluator`：对每个候选 `evaluate()` → `{params, accuracy, latency_ms, under_budget}`；更新全局 `best`。
3. `Critic`：对 `best` 计算 refinement 动作 + 自然语言批判；`Memory.add(arch, critique, score)`。
4. `Router`：若 `gen < iters-1` 且 `best.accuracy < target` → `Refiner`，否则 `Synthesizer`。
5. `Refiner`：`refine_arch(best, pending)` 产出改进候选 + 注入随机多样性，回 `Evaluator`（闭环）。

## 3. 不变量（可验证）

- `count_params` 单调随宽度/深度增加；同架构重复评估分数恒定（无随机）。
- `Memory.query` 返回按余弦相似度降序的 top-k，长度 ≤ k。
- 闭环终止：最多 `iters` 代或达到 `target`，Guard=256 防死循环。
- 离线路径下 `used_onnx/used_haystack/used_diffusers` 全为 `False`，但 `report` 仍完整。

## 4. 决策锁校验（Decision Lock）

| 决策 | 锁 | 状态 |
|------|----|------|
| 用"可复现随机"替代"纯随机" | 保证顶级系统必备的目的性 | ✅ 已锁 |
| 顶级库全部可选 + 优雅降级 | 保证闭环（开环即失效） | ✅ 已锁 |
| 代理评分仅作占位，标注非真实准确率 | 法律/诚实优先于炫技 | ✅ 已锁 |
| 作者署名统一"晨星"，MIT 许可 | 合规 | ✅ 已锁 |
| 真实准确率需接训练后端（torch） | 边界内未越权 | ⚠️ 留接口 |

## 5. 批判性分析

- "全随机顶级 AI"内在矛盾：顶级=目的性，随机=无目的。本仓库以**可复现随机抽签 + 闭环收口**化解，
  而非迎合字面。抽中的 4 库恰好正交（编排/记忆/压测/先验），组合无冗余。
- 当前 `accuracy` 为**代理值**（容量/正则/激活的平滑函数），不是真训练精度。要上真实精度，
  接 `torch` 训练后端替换 `evaluator` 即可——接口已预留，未越权替用户决定。
- NAS 搜索空间目前是线性链（MLP）；卷积块规格已定义但未在代理中深度利用，属已知边界。
