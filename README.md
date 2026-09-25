# SynthMind-Forge · RCG-NAS

> 递归批判引导的神经网络架构搜索引擎（Recursive Critique-Guided NAS）
> 作者：**晨星 (CJX0712)** · Shenzhen · MIT License

一个"全随机"抽签诞生的顶级 AI 系统：用带种子的可复现随机在顶级开源项目池里抽中
`ONNX Runtime` / `LangGraph` / `Haystack` / `Diffusers`，再用一条自洽闭环架构把它们串成
**真能跑**的神经网络架构搜索引擎。零第三方依赖即可开箱运行；装齐顶级库后自动升级到真实后端。

---

## 一句话创新

**RCG-NAS** = Reflexion 式自批判 + NAS 种群搜索 + RAG 失败记忆。
Critic 不只打分，还**提出对候选架构的具体修改**；检索历史批判让 Planner 不再重复踩坑。
把"会反省的 Agent"嫁接到"架构搜索"上——这是本仓库的独特点。

## 抽中的顶级开源项目（溯源见 RANDOM_SEED.md）

| 顶级库 | 在本系统的角色 | 维度 |
|--------|----------------|------|
| **LangGraph** | Agent 状态机编排（Planner→Evaluator→Critic→Refiner→Synthesizer） | 编排 |
| **Haystack** | 架构知识库 RAG 记忆（落库历史批判，BM25 检索） | 记忆 |
| **ONNX Runtime** | 把候选网编译为 ONNX 并真实压测推理延迟 | 压测 |
| **Diffusers** | 架构 latent 扩散先验，引导初始种群（novel hook） | 先验 |

## 30 秒上手

```bash
# 零依赖即可跑（纯标准库离线闭环）
python -m synthmind --task "edge image classifier" --budget 40000 --iters 6 --pop 4 --target 0.9

# 跑自检
python tests/test_loop.py

# 升级到顶级后端（可选）
pip install -r requirements.txt
```

输出示例：

```
# SynthMind-Forge · RCG-NAS report
generations   : 6
best accuracy : 0.5933  params=36490  latency=36.49ms
top-tier OSS  : ONNX=False Haystack=False Diffusers=False   # 装库后变 True
```

## 目录结构

```
synthmind-forge/
├── synthmind/
│   ├── __init__.py        # 包导出
│   ├── engine.py          # RCGNAS 引擎 + 状态机编排
│   ├── orchestrator.py    # LangGraph 式 StateGraph（零依赖实现）
│   ├── search_space.py    # 候选架构规格 + 特征嵌入
│   ├── evaluator.py       # 评分：离线代理 + ONNX Runtime 真实压测
│   ├── memory.py          # RAG 记忆：Haystack 优先，本地 NN 降级
│   ├── critic.py          # 自批判 + 定向 refinement
│   ├── diffusion_prior.py # Diffusers 架构先验（可选）
│   └── cli.py / __main__.py
├── tests/test_loop.py     # 离线自检
├── demo.html              # 单文件交互展示页（纯前端）
├── ARCHITECTURE.md        # 架构详解 + 决策锁校验
├── RANDOM_SEED.md         # 随机抽签溯源
├── requirements.txt
└── LICENSE
```

## 闭环公理验证

- ✅ **闭环**：`python -m synthmind` 端到端产出报告，无网络/无第三方库也不开环。
- ✅ **溯源**：所有"随机"选择记录在 `RANDOM_SEED.md`，种子可复现。
- ✅ **边界**：每个顶级库均有 `available` 探测与优雅降级，越界即回退本地实现。
- ✅ **法律优先**：MIT 许可证；所引顶级库均为 MIT/Apache 兼容，无 GPL 污染。

## 引用（顶级研究）

- Yao et al. 2023, *Tree of Thoughts* — 分解式规划
- Shinn et al. 2023, *Reflexion* — 自批判/自反省
- Yao et al. 2022, *ReAct* — 推理-行动交错
- Lewis et al. 2020, *RAG* — 检索增强生成
- Elsken et al. 2019, *Neural Architecture Search* — NAS 综述
