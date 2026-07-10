# 数据字典

本文档描述当前 NC mix jsPsych 范式、真实数据分析 pipeline、adaptive B/D/E 以及正式可视化输出的数据结构。后续维护以 `src/cognitivemap`、`script/real_data_pipeline`、`script/simulation_recovery_pipeline` 和 `script/visualization` 为主；`experiments/新数据` 只作为历史原型来源，不作为正式数据入口。

## 1. 目录约定

| 路径 | 定位 | 当前状态 |
|---|---|---|
| `data/raw_data/` | 当前已归档的旧版 jsPsych 原始 JSON | 包含 P003、P005 的 `mix_nc_jspsych_*.json` |
| `data/real_data_pipeline/raw/` | `script/real_data_pipeline/preprocess_human_data.py` 的默认原始数据目录 | 若使用默认命令，需要把原始 JSON 放到这里，或显式传 `--input-dir data/raw_data` |
| `data/real_data_pipeline/processed/` | 预处理后每被试每 domain 的 step-level joblib | 由 `preprocess_human_data.py` 生成 |
| `data/real_data_pipeline/inference_results/` | chunk / landmark 推断结果 | 由 `infer_chunk_human.py`、`infer_landmark_human.py` 生成 |
| `script/visualization/` | 正式维护的可视化脚本入口 | 当前包含 `visualize_mix_jspsych_data.py` |
| `result/figure/` | 正式图表输出根目录 | 当前 `result/figure/mix_jspsych/` 已生成旧版数据图 |
| `experiments/新数据/` | 历史探索性分析和可视化原型 | 可退出正式依赖；删除前确认是否需要归档旧图和旧说明 |

推荐命令：

```bash
python script/visualization/visualize_mix_jspsych_data.py --input-dir data/raw_data --output-dir result/figure/mix_jspsych
python script/real_data_pipeline/preprocess_human_data.py --input-dir data/raw_data --output-dir data/real_data_pipeline/processed
python script/real_data_pipeline/infer_chunk_human.py --input-dir data/real_data_pipeline/processed --output data/real_data_pipeline/inference_results/chunks.joblib
python script/real_data_pipeline/infer_landmark_human.py --input-dir data/real_data_pipeline/processed --output data/real_data_pipeline/inference_results/landmarks.joblib
```

## 2. 原始 jsPsych JSON

每个原始文件是一个 jsPsych trial 数组。文件名形如：

```text
mix_nc_jspsych_1780586225576.json
mix_nc_jspsych_1780586325394.json
```

顶层常用字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `screen_id` | string | 页面或 trial 类型标识，例如 `mix_navigation_trial`、`tail_trial_B_B-27`、`map_reconstruction_navigation` |
| `phase` | string | 阶段标识，例如 `main`、`tail_task`、`map_reconstruction`、`adaptive_tail_task` |
| `trial_type` | string | jsPsych 插件类型 |
| `trial_index` | int | jsPsych 全局 trial 序号 |
| `time_elapsed` | int | 当前记录完成时累计时间，单位 ms |
| `internal_node_id` | string | jsPsych 内部节点 id |
| `participant_id` | string | 被试编号，建议统一补零为三位，如 `003` |
| `experiment_order` | string | 顺序条件，例如 `navigation-first` 或 `crafting-first` |
| `value` | object | 该 trial 的主要负载；主任务、B/D/E、地图还原和 adaptive 负载都在这里 |

读取时需要注意：部分字段同时出现在顶层和 `value` 内层。正式脚本通常先展开 `value`，但不覆盖已有顶层字段。

## 3. 主任务 trial

主任务记录的 `screen_id` 为：

```text
mix_navigation_trial
mix_crafting_trial
```

顶层或 `value` 中的 trial 级字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `phase` | string | 固定为 `main` |
| `domain` / `mix_session_domain` | string | `navigation` 或 `crafting` |
| `mix_session` | int | Session 编号，旧版数据为 1、2、3 |
| `mix_step_index` | int | 当前 session 内主任务 trial 序号 |
| `mix_step_total` | int | 当前 session 内主任务 trial 总数 |
| `pair_id` | string | 起终点 pair 编号 |
| `category` | string | 试次类别，例如 loop / grid 类别 |
| `trial_id` | string | 实验程序内 trial id，例如 `nav_1` |
| `start_code` | int | 起点节点，1..9 |
| `goal_code` | int | 目标节点，1..9 |
| `goal_codes` | list[int] | 目标节点列表，当前通常只有一个目标 |
| `steps` | list[object] | 被试在该 trial 中的逐步行为 |
| `total_steps` | int | 有效移动步数 |
| `outcome` | string | 结果，例如 `success` |
| `total_time_ms` | int | trial 总耗时 |
| `recipe` / `task_id` | string/object | crafting 相关任务信息；旧版 schema 中可能存在 |

`steps` 中常用字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `step` | int | 有效步序号；无效按键通常不推进 step |
| `from_code` / `prev` | int/string | 动作执行前节点 |
| `to_code` / `next` | int/string/null | 动作执行后节点；无效动作可能为 null |
| `key` | string | 原始按键：`q`、`e`、`a`、`d`、`w` |
| `valid` | bool | 是否为合法动作 |
| `rt_ms` | int | 本步反应时，单位 ms |
| `mode` | string | 呈现/动作模式 |
| `direction` | string | 面向被试的动作说明 |
| `dir_code` | string | 程序内部方向编码，如 `left`、`right`、`up`、`down`、`cw` |
| `order_completed` | bool/null | crafting 任务中订单是否完成 |
| `order_target` | string/int/null | crafting 任务中当前目标 |

动作映射：

| 原始按键 | pipeline 动作 | adaptive JS token | 含义 |
|---|---|---|---|
| `q` | `L` | `q` | 左移 |
| `e` | `R` | `e` | 右移 |
| `a` | `U` | `a` | 上移 |
| `d` | `D` | `d` | 下移 |
| `w` | `LOOP` | `w` | 角点顺时针 shortcut：`1→3→9→7→1` |

## 4. 预处理 step-level 数据

`script/real_data_pipeline/preprocess_human_data.py` 将原始 JSON 转成每被试每 domain 一个 joblib：

```text
participant_{participant_id}_{domain}.joblib
```

每个 joblib 内是一个 pandas DataFrame，列顺序如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `trial_id` | string | pipeline 生成的唯一 trial id，例如 `navigation_001` |
| `raw_trial_id` | string | 原始 jsPsych trial id |
| `domain` | string | `navigation` 或 `crafting` |
| `trial_index` | int | 原始 trial 内/程序内序号 |
| `start` | int | 起点节点 |
| `goal` | int | 目标节点 |
| `outcome` | string | trial 结果 |
| `total_steps` | int | trial 有效步数 |
| `total_time_ms` | int | trial 总耗时 |
| `step` | int | step 序号 |
| `state` | int | 动作执行前节点 |
| `key` | string | 原始按键 |
| `action` | string | 抽象动作：`U`、`D`、`L`、`R`、`LOOP` |
| `next_state` | int | 动作执行后节点；无效动作保持原状态 |
| `valid` | bool | 是否为合法动作 |
| `order_completed` | bool/null | crafting 订单完成标记 |
| `rt_ms` | int | step 反应时 |

原则上，chunk / landmark 推断只使用 `valid == True` 的动作或由有效动作恢复出的状态序列。

## 5. 固定 B/D/E 数据

旧版数据在 Session 1、Session 2 后保存固定 B/D/E。新版主设计中，Session 3 后改为 adaptive B/D/E。

共同顶层字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `phase` | string | 固定为 `tail_task` |
| `tail_domain` | string | 题目对应 domain |
| `tail_subphase` | string | `B`、`D` 或 `E` |
| `tail_trial_id` | string | 题目 id |
| `tail_category` | string | 题目类别 |
| `tail_present_version` | string | 呈现版本，例如 `v3` |
| `value.tail_rt_ms` | int | 该题反应时 |

固定 B：节点对可达性评分。

| 字段 | 类型 | 含义 |
|---|---|---|
| `value.tail_response_reachability` | int | 被试评分，通常 1..5 |
| `value.tail_stimulus_pair` | string | 呈现节点对 |
| `value.tail_grid_distance` | int | 网格距离 |
| `value.value.a` / `value.value.b` | int | 节点对 |
| `value.value.display_left` / `value.value.display_right` | int | 左右呈现顺序 |

固定 D：序列整体感评分。

| 字段 | 类型 | 含义 |
|---|---|---|
| `value.tail_response_fluency` | int | 被试评分，通常 1..5 |
| `value.tail_stimulus_sequence` | string | 呈现序列 |
| `value.value.category` | string | 序列类别 |
| `value.value.sequence` | list[int] | 节点序列 |

固定 E：理想中转节点选择。

| 字段 | 类型 | 含义 |
|---|---|---|
| `value.tail_response_hub` | int | 被试选择的中转节点 |
| `value.tail_stimulus_route` | string | 起点到终点 |
| `value.tail_e_hub_options` | list[int] | 可选中转节点 |
| `value.value.start` / `value.value.end` | int | 起点和终点 |
| `value.value.chosen` | int | 被试选择 |
| `value.value.hub_options_order` | list[int] | 选项顺序 |

## 6. 地图还原数据

旧版数据包含最终地图还原，新版 adaptive B/D/E 方案默认去掉最终地图还原。

`screen_id`：

```text
map_reconstruction_navigation
map_reconstruction_crafting
```

主要字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `value.phase` | string | `map_reconstruction` |
| `value.domain` | string | `navigation` 或 `crafting` |
| `value.map_reconstruction_ui_version` | string | UI 版本 |
| `value.map_reconstruction.nodes` | list[object] | 9 个节点的摆放坐标 |
| `value.map_reconstruction.nodes[].code` | int | 节点编号 |
| `value.map_reconstruction.nodes[].xp` | float | 被试摆放 x 坐标，归一化 |
| `value.map_reconstruction.nodes[].yp` | float | 被试摆放 y 坐标，归一化 |
| `value.duration_ms` / `value.total_time_ms` | int | 完成耗时 |

当前 `script/visualization/visualize_mix_jspsych_data.py` 会对旧版地图还原做 Procrustes stress 可视化；新版数据若没有该任务则返回空表，不报错。

## 7. Adaptive B/D/E 数据

新版实验在 Session 3 后生成 adaptive B/D/E。当前实验端生成器为：

```text
experiments/NC_MIX_TASK_jspsych版本/mix_jspsych_3session_v2/ae-paradigm/js/adaptive_tail_task_embed_v3.js
```

关键记录：

| `screen_id` | `phase` | 含义 |
|---|---|---|
| `adaptive_bde_rest` | `transition_rest` | Session 3 后休息页，并在后台生成 adaptive pack |
| `adaptive_bde_preface` | `adaptive_tail_task` | adaptive B/D/E 开始说明 |
| `adaptive_bde_runner` | `adaptive_tail_task` | adaptive B/D/E 正式作答结果 |

`adaptive_bde_rest` 的主要负载：

| 字段 | 类型 | 含义 |
|---|---|---|
| `rest_min_duration_ms` | int | 最短休息时间，当前 5000 ms |
| `rest_actual_duration_ms` | int | 实际休息时间 |
| `background_task_used` | bool | 是否后台生成 |
| `background_task_done` | bool | adaptive pack 是否生成完成 |
| `background_error` | string/null | 生成错误 |
| `adaptive_pack_summary` | object/null | pack 摘要 |

`adaptive_bde_runner` 的主要负载：

| 字段 | 类型 | 含义 |
|---|---|---|
| `adaptive_tail_task_version` | string | adaptive 任务版本 |
| `adaptive_pack` | object | 生成依据、候选和全部 adaptive trials |
| `adaptive_results` | list[object] | 被试实际回答 |
| `interrupted` | bool | 是否中断 |
| `trial_count` | int | 完成题数 |

`adaptive_pack` 的核心字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema` | string | 当前为 `adaptive_bde_pack` |
| `version` | string | 当前为 `1.1-js-bdeu-bootstrap` |
| `participant_id` | string | 被试编号 |
| `experiment_order` | string | 顺序条件 |
| `source_data_window` | string | 当前为 `last_80_percent_by_domain` |
| `window_rule` | string | 当前为 `ceil_drop_first_20_percent_keep_last_80_percent` |
| `items_per_phase_domain` | int | 每个 domain 每个 B/D/E phase 的目标题数，当前默认 8 |
| `domains.navigation` / `domains.crafting` | object | 各 domain 的候选 chunk / landmark |
| `trials` | list[object] | 混合后的 adaptive 题目 |

每个 domain 的候选元数据：

| 字段 | 类型 | 含义 |
|---|---|---|
| `trial_count_total` | int | 该 domain 主任务 trial 总数 |
| `trial_count_used` | int | 去掉前 20% 后实际用于 mining 的 trial 数 |
| `chunk_method` | string | 当前为 state-conditioned BDeu recursive chunk mining |
| `chunk_bootstrap_iterations` | int | 当前 80 |
| `landmark_method` | string | coverage / path commonality / betweenness rank aggregation + bootstrap |
| `landmark_bootstrap_iterations` | int | 当前 500 |
| `landmark_selection_threshold` | float | 当前 0.7 |
| `chunks` | list[object] | chunk 候选 |
| `landmarks` | list[object] | landmark 候选 |

chunk 候选常用字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | string | 候选 id |
| `chunk` | string | 原始 action token 组合，如 `w-w` |
| `actions` | list[string] | action token 列表 |
| `states` | list[int] | 最代表性的状态 realization |
| `length` | int | primitive action 长度 |
| `count` | int | 全数据最长匹配后的出现次数 |
| `support_trials` | int | 支持该 chunk 的 trial 数 |
| `score` | float | 单位样本 BDeu evidence |
| `log_bayes_factor` | float | BDeu log BF |
| `joint_count` | int | 相邻二元合成的联合计数 |
| `bootstrap_stability` | float | chunk bootstrap 选择稳定性 |
| `state_realizations` | list[object] | 该 action pattern 实际对应过的状态序列 |
| `source` | string | `bdeu_state_conditioned` 或 fallback |

landmark 候选常用字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | string | 候选 id |
| `node` | int | 节点编号 |
| `score` | float | rank aggregation 综合分 |
| `selection_rate` / `bootstrap_stability` | float | subsampling bootstrap 选择率 |
| `coverage` | float | 跨 trial 覆盖率 |
| `path_commonality` | float | 服务不同 start-goal pair 的中间路径共同性 |
| `betweenness` | float | 经验转移图上的 weighted betweenness |
| `source` | string | `bootstrap_landmark` 或低稳定性标记 |

adaptive trial 通用字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | string | 题目 id |
| `phase` | string | `B`、`D` 或 `E` |
| `domain` | string | `navigation` 或 `crafting` |
| `target_construct` | string | `chunk`、`landmark`、`baseline` 等 |
| `condition_label` | string | 条件标签，如 chunk target/control/conflict |
| `candidate_chunk_id` | string/null | 关联 chunk |
| `candidate_landmark_id` | string/null | 关联 landmark |
| `chunk_confidence` | float/null | chunk 候选置信度 |
| `landmark_confidence` | float/null | landmark 候选置信度 |
| `nodes` / `sequence` | list[int] | B/D 呈现节点或序列 |
| `actions` | list[string] | D 序列动作 |
| `start` / `end` | int | E 起终点 |
| `hub_options` | list[int] | E 候选中转节点 |
| `rating` | int | B/D 评分结果，出现在 `adaptive_results` |
| `chosen_hub` | int | E 选择结果，出现在 `adaptive_results` |
| `rt_ms` | int | 该题反应时，出现在 `adaptive_results` |

## 8. Inference 结果

chunk inference 输出文件：

```text
data/real_data_pipeline/inference_results/chunks.joblib
```

结构为：

```text
{domain: {participant_id: result}}
```

常用字段：

| 字段 | 含义 |
|---|---|
| `participant_id` | 被试编号 |
| `domain` | 任务域 |
| `trial_count` | 纳入 trial 数 |
| `input_token_count` | 拼接后 token 数，含 trial 分隔符时包含 `S` |
| `effective_action_count` | 有效动作数 |
| `vocab` | 最终词表 |
| `learned_chunks` | 推断出的复合 chunk |
| `original_action_sequence` / `original_state_sequence` | 原始动作/状态序列 |
| `parsed_action_sequence` / `parsed_state_sequence` | chunk 最长匹配后的解析序列 |
| `components` | chunk 的组成关系 |
| `js_history` | 迭代重解析的 JS 距离历史 |

landmark inference 输出文件：

```text
data/real_data_pipeline/inference_results/landmarks.joblib
```

结构同样为：

```text
{domain: {participant_id: result}}
```

常用字段：

| 字段 | 含义 |
|---|---|
| `landmarks` | 达到 selection threshold 的稳定 landmark |
| `top_landmarks` | 全数据 score 排名前列的 landmark |
| `candidate_ranking` | 每个节点的 score、selection_rate 和特征 |
| `transition_counts` | 经验有向转移计数 |
| `visit_counts` | 节点访问计数 |
| `state_scores` | 节点级特征表 |
| `config` | mining 配置 |

## 9. 可视化输出

当前正式可视化脚本：

```text
script/visualization/visualize_mix_jspsych_data.py
```

默认输出：

```text
result/figure/mix_jspsych/
```

当前图表：

| 输出 | 含义 |
|---|---|
| `group/path_efficiency_by_trial.png` | domain 内 trial 顺序的路径效率趋势 |
| `group/path_efficiency_by_session.png` | session × domain 平均路径效率 |
| `group/reaction_time_trends.png` | trial 总耗时和平均 step RT |
| `group/tail_D_fluency_boxplot.png` | 固定 D 整体感评分 |
| `group/tail_E_hub_on_shortest_path.png` | 固定 E 所选 hub 是否在最短路上 |
| `P{pid}/tail_B_heatmap_{domain}.png` | 固定 B 主观可达性热图 |
| `P{pid}/tail_B_scatter_{domain}.png` | 固定 B 评分与真实图距离关系 |
| `P{pid}/map_reconstruction_{domain}.png` | 旧版地图还原 Procrustes 可视化 |
| `P{pid}/summary.json` | 单被试轻量汇总 |
| `all_summaries.json` | 全部被试轻量汇总 |

旧 `experiments/新数据/outputs/chunks/` 中还保留过一些探索性图，如 transition heatmap、TPM convergence、macro catalog、chunk usage、chunk vs efficiency、behavioral cognitive map HTML/PNG/JSON。这些不是正式依赖；若研究汇报仍需要，应迁移到 `script/visualization` 或通过 `script/real_data_pipeline` / `src/cognitivemap.map_estimation` 重新生成。

## 10. 删除 `experiments/新数据` 的判断

从正式代码依赖看，可以删除 `experiments/新数据`，理由是：

- 原始 JSON 已在 `data/raw_data/` 另存；
- 当前正式可视化脚本可从 `data/raw_data/` 生成 `result/figure/mix_jspsych/`；
- chunk / landmark / map estimation 的正式算法在 `src/cognitivemap` 与 `script/real_data_pipeline`；
- 当前仓库内没有发现正式模块 import `experiments/新数据` 下的 Python 文件。

但从历史材料保留看，删除前建议先确认是否还需要：

- `experiments/新数据/结果解读指南.md`；
- `experiments/新数据/outputs/chunks/` 下的历史探索图和 HTML；
- `analyze_mix_data.py`、`behavioral_cognitive_map.py`、`chunk_analysis.py` 中尚未迁移的图表思路。

若只关心后续 adaptive B/D/E 正式分析，删除是可行的；若需要复现旧报告图，建议先归档该目录，或把未迁移图表补到 `script/visualization` 后再删。
