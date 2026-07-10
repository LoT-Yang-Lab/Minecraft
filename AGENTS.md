# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作提供指导。

## 环境与常用命令

```bash
# 安装依赖 (Poetry)
poetry install

# 若使用 Conda
conda run -n CogMap <命令>

# 若 PATH 被其他软件污染，优先使用 CogMap 解释器绝对路径
/home/zzh/anaconda3/envs/CogMap/bin/python -m pytest -q
/home/zzh/anaconda3/envs/CogMap/bin/python -m compileall -q src script tests

# 运行全部测试
pytest -q

# 格式化 / Lint / 类型检查
black src tests script
isort src tests script
flake8 src tests script
mypy src
```

行宽限制 120 字符。Flake8 忽略 E203、W503、E231（与 Black 兼容）。

## 当前架构概览

这是一个研究导航任务中人类表征方式的科研项目。探索阶段的
`cognitivemap.loop_grid_agents` 旧仿真系统已经删除；后续正向生成模型实现应以
`docs/开发文档/generative_model_software_design_cn.html` 为准，不应继续沿用旧
`T1/T2/T3/S1/S2/S3/SN` concrete agent 体系。

当前保留的可复用子系统：

**1. `cognitivemap.map_estimation`** — 从多 trial 序列数据构建认知地图。包含 SR、LoPS、
Action JS、Transition Similarity 四种距离方法，MDS 二维嵌入，以及 HTML 可视化支持。

**2. `cognitivemap.inference`** — 从行为序列推断 action chunk 与 landmark candidate。
这里的 inference 指“从观测行为反推表征”，底层仍可使用 miner/parser 这类具体算法组件。

**3. `cognitivemap.generative_model`** — 使用显式 primitive / chunk / landmark 表征正向生成导航行为。

**4. `script/`** — 按真实数据 pipeline、模拟恢复 pipeline 和 demo 三类组织脚本。

## 新生成模型开发依据

下一阶段系统设计文档位于：

- `docs/开发文档/generative_model_requirements_cn.html`
- `docs/开发文档/generative_model_context_constraints_cn.html`
- `docs/开发文档/generative_model_software_design_cn.html`
- `docs/开发文档/landmark_model_issue_notes_cn.html`

实现新系统时应建立新的生成模型模块，而不是恢复或改造旧 `loop_grid_agents`。

## PEP8 安全 Lint 规则

修复 lint 问题时：**禁止删除未使用的导入**——在其上方添加 `# unused import` 注释。
对于未使用的变量，优先注释掉或改为 `_`。不得修改 `__init__.py` 中的 re-export。

## 核心开发理念

### 简洁至上

恪守 KISS 原则。优先选择最直接、最清晰、最易维护的实现；避免过度工程化、过早抽象和不必要的防御性设计。

### 深度分析

在动手实现前，基于第一性原理理解问题。分析必须服务于当前任务本身，不脱离文档进行无边界扩展。

### 事实为本

以文档、现有代码、可验证结果和数值行为为准。若发现用户表述、代码实现或自身理解存在错误，必须明确指出并给出依据，不得迎合或模糊处理。

### 高效协作

当任务可被清晰拆分为相对独立的子任务时，优先采用多 agent 并行协作以提高效率。分工必须明确，避免重复劳动和职责混乱。所有子任务结果由主 agent 统一整合，确保术语、接口、实现风格和代码质量一致。若任务本身较小或依赖紧密，则由单一 agent 完成。

## 工作规则

1. 始终使用中文与用户交流。
2. 文档写作始终使用中文。
3. 修改代码前先说明计划和会影响的文件。
4. 优先采用最小改动方案，除非用户明确指出需要重构。
5. 执行命令前先说明命令目的。
6. 修改完成后总结改动内容、测试结果和潜在风险。
