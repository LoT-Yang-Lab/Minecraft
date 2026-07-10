# Minecraft

Minecraft 是一个用于研究导航任务中人类表征方式的科研代码库。当前仓库按三层能力组织：

- `map_estimation`：从多 trial 行为序列估计认知地图。
- `inference`：从观测行为中推断 action chunk 与 landmark candidate。
- `generative_model`：在显式给定表征的前提下，正向生成导航行为，用于模拟数据与恢复验证。

探索阶段的旧 `loop_grid_agents`、旧 `cognitive_navigation`、旧 `pattern_mining` 入口已经不再作为当前接口使用。

## 核心模块

### `cognitivemap.map_estimation`

从状态/动作序列构建认知地图，输入通常是一组 `Trial(state_sequence, action_sequence)`。

主要能力：

- `compute_sr_distance`：Successor Representation 距离。
- `compute_lops_distance`：基于转移概率对数的最短路径距离。
- `compute_action_js_distance`：动作分布 Jensen-Shannon 距离。
- `compute_transition_similarity_distance`：转移分布 Jensen-Shannon 距离。
- `build_cognitive_map` / `mds_embed`：二维 MDS 嵌入。
- `enrich_from_trials`：为可视化补充边上 action/chunk 使用统计。
- `render_cognitive_map_html`：生成交互式 HTML 图。

### `cognitivemap.inference`

从行为序列反推可解释表征单元。这里的 inference 指“表征推断”，不是通用贝叶斯推断框架。

文件职责：

- `parser_chunk.py`：Trie 最长匹配 chunk parser。
- `miner_chunk.py`：基于相邻 token 依赖和 BDeu 证据的 action chunk miner。
- `miner_chunk_state_conditioned.py`：使用离散状态父变量的 chunk miner 版本。
- `miner_landmark.py`：基于状态序列特征与 bootstrap stability selection 的 landmark candidate miner。

### `cognitivemap.generative_model`

正向生成模型模块。当前版本关注主要逻辑功能，不做 EM、表征恢复、UI 或随机转移环境。

主要对象：

- `TaskGraph`：确定性 primitive 任务图。
- `PrimitiveRepresentation`：primitive action 权重表征。
- `ChunkRepresentation`：primitive + chunk token 权重表征。
- `LandmarkRepresentation`：具体 state landmark 与 LL/LU/UL CPD 表征。
- `build_transition_kernel`：由表征诱导状态转移 kernel 和 support。
- `build_geometry_artifacts`：构建 cognitive graph、最短路距离和 MDS embedding。
- `plan_cognitive_path`：在 cognitive graph 上运行 A*。
- `expand_cognitive_path`：将 cognitive path 展开为 primitive action/state 轨迹。
- `CognitiveNavigationAgent`：封装 artifact 构建、trial 运行和 JSON-compatible 输出。

## 脚本流程

### Demo

认知地图估计 demo：

```bash
python script/demo/demo_map_estimation.py \
  --mode simulated \
  --output-dir results/figs/demo/cognitive_maps
```

九宫格生成模型 demo：

```bash
python script/demo/demo_generative_model_grid9.py \
  --n-trials 8 \
  --output-dir data/demo/generative_model_grid9
```

### 真实数据流程

真实数据流程位于 `script/real_data_pipeline/`，数据和结果按用途分开保存：

- 原始数据：`data/real_data_pipeline/raw/`
- 预处理数据：`data/real_data_pipeline/processed/`
- 推断结果：`data/real_data_pipeline/inference_results/`
- 认知地图图形：`results/figs/real_data_pipeline/cognitive_maps/`
- 3D landmark 图形：`results/figs/real_data_pipeline/landmarks_3d/`

推荐运行顺序：

```bash
python script/real_data_pipeline/preprocess_human_data.py
python script/real_data_pipeline/infer_chunk_human.py
python script/real_data_pipeline/infer_landmark_human.py
python script/real_data_pipeline/visualize_map_human.py
python script/real_data_pipeline/visualize_landmark_human_3d.py
```

### 模拟恢复流程

模拟恢复流程位于 `script/simulated_recovery_pipeline/`。它先用生成模型产生模拟数据，再用
`inference` 模块挖掘 landmark，最后比较恢复结果与真实 landmark。

输出位置：

- 模拟轨迹池：`data/simulated_recovery_pipeline/simulated_datasets/`
- 恢复验证结果：`data/simulated_recovery_pipeline/recovery_results/`
- 恢复报告和 elbow 图：`results/figs/simulated_recovery_pipeline/landmark_recovery/`

快速检查流程：

```bash
python script/simulated_recovery_pipeline/generate_landmark_simulated_data.py --quick
python script/simulated_recovery_pipeline/validate_landmark_recovery.py --quick \
  --simulated-data-input data/simulated_recovery_pipeline/simulated_datasets/landmark_grid9_quick.joblib
python script/simulated_recovery_pipeline/visualize_landmark_recovery.py \
  --input data/simulated_recovery_pipeline/recovery_results/landmark_recovery_quick.joblib \
  --output results/figs/simulated_recovery_pipeline/landmark_recovery/recovery_report_quick.html
python script/simulated_recovery_pipeline/visualize_landmark_elbow.py \
  --input data/simulated_recovery_pipeline/recovery_results/landmark_recovery_quick.joblib
```

## 数据与输出约定

`data/` 保存可复现实验数据和中间结果，`results/figs/` 保存 HTML/PNG 等图形结果。两者都视为可重新生成的本地实验产物，不作为核心源码接口的一部分。

当前主要目录约定：

```text
data/
  real_data_pipeline/
    raw/
    processed/
    inference_results/
  simulated_recovery_pipeline/
    simulated_datasets/
    recovery_results/

results/figs/
  real_data_pipeline/
    cognitive_maps/
    landmarks_3d/
  simulated_recovery_pipeline/
    landmark_recovery/
  demo/
```

## 代码结构

```text
src/cognitivemap/
  map_estimation/
    __init__.py
    distances.py
    embedding.py
    enrichment.py
    io.py
    simulation.py
    visualization.py
    templates/
      cognitive_map_template.html

  inference/
    __init__.py
    parser_chunk.py
    miner_chunk.py
    miner_chunk_state_conditioned.py
    miner_landmark.py

  generative_model/
    __init__.py
    types.py
    task.py
    representations.py
    transition_kernel.py
    cognitive_graph.py
    planning.py
    agent.py
```

## 设计文档

方法推导与模型讨论：

- `docs/method/analysis-7.tex`
- `docs/method/analysis-7-1.tex`
- `docs/method/analysis-8.tex`
- `docs/method/analysis-9.tex`
- `docs/method/analysis-10.tex`

当前生成模型宏观设计文档：

- `docs/开发文档/generative_model_macro_framework_cn.html`
- `docs/开发文档/generative_model_macro_requirements_cn.html`
- `docs/开发文档/generative_model_macro_software_design_cn.html`

`docs/开发文档/bak/` 中的旧生成模型文档作为历史参考保留，不再作为当前实现的首要接口说明。

## 环境

目标 Python 版本为 3.10。推荐使用 Poetry 安装依赖：

```bash
poetry install
```

若使用 Conda 环境 `CogMap`：

```bash
conda activate CogMap
poetry install
```

若不通过 Poetry 运行命令，请确保项目源码在 `PYTHONPATH` 中：

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

如果 PATH 被其他软件污染，可直接使用当前环境解释器：

```bash
/home/zzh/anaconda3/envs/CogMap/bin/python -m pytest -q
/home/zzh/anaconda3/envs/CogMap/bin/python -m compileall -q src script tests
```

## 开发与验证

常用检查命令：

```bash
/home/zzh/anaconda3/envs/CogMap/bin/python -m black src script tests
/home/zzh/anaconda3/envs/CogMap/bin/python -m flake8 src script tests
/home/zzh/anaconda3/envs/CogMap/bin/python -m compileall -q src script tests
/home/zzh/anaconda3/envs/CogMap/bin/python -m pytest -q
```

也可以使用本地 PATH 中的工具：

```bash
black src script tests
flake8 src script tests
pytest -q
```

项目代码和文档以中文注释/说明为主；保留英文技术名词用于 API、论文术语和第三方库接口。
