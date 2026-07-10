# 任务设计总览

本文档记录当前 NC mix 范式的正式设计取向。后续开发默认面向新版 adaptive B/D/E；已有 P003、P005 数据来自上一版固定 B/D/E + 最终地图还原流程，只作为历史数据和脚本迁移验证材料。

更细的 adaptive B/D/E 数学方案见：

```text
docs/task_design/adaptive_BDE_生成方案.md
```

## 1. 科学目标

核心问题是：被试在同构的 navigation 与 crafting 任务中，究竟更依赖 action chunk，还是更依赖 landmark / hub？

实验希望得到四类证据：

| 任务域 | chunk 证据 | landmark 证据 |
|---|---|---|
| navigation | 导航行为中形成的动作组块 | 导航行为中稳定组织路线的节点 |
| crafting | 合成行为中形成的动作组块 | 合成行为中稳定组织加工路线的节点 |

最终分析至少形成四个被试级指标：

```text
navigation_chunk_score
navigation_landmark_score
crafting_chunk_score
crafting_landmark_score
```

进一步比较：

```text
representation_bias_navigation = navigation_chunk_score - navigation_landmark_score
representation_bias_crafting   = crafting_chunk_score - crafting_landmark_score
domain_chunk_difference        = navigation_chunk_score - crafting_chunk_score
domain_landmark_difference     = navigation_landmark_score - crafting_landmark_score
```

## 2. 任务图和动作

两个任务域共享同一个 3 × 3 有向任务图，节点编号为 1..9：

```text
1 2 3
4 5 6
7 8 9
```

普通移动键：

| 按键 | 抽象动作 | 含义 |
|---|---|---|
| `q` | `L` | 左移 |
| `e` | `R` | 右移 |
| `a` | `U` | 上移 |
| `d` | `D` | 下移 |

特殊 shortcut：

```text
w / LOOP：1 → 3 → 9 → 7 → 1
```

`w` 是有方向的角点顺时针 shortcut，不是无向边。因此后续 B 题和模型分析中，方向性信息很重要。

## 3. 任务域

| domain | 表面语义 | 底层结构 |
|---|---|---|
| `navigation` | 站点/线路导航 | 3 × 3 图 + `w` shortcut |
| `crafting` | 合成/加工流程 | 与 navigation 同构的 3 × 3 图 + `w` shortcut |

两个 domain 底层拓扑同构，但分析时不默认混合。chunk 和 landmark 都应按 domain 分开估计，因为我们关心同构结构在不同语义域里是否形成一致表征。

## 4. 新版实验流程

推荐流程：

```text
Session 1：纯 navigation 或纯 crafting 主任务
→ 固定 B/D/E

Session 2：另一个纯任务域主任务
→ 固定 B/D/E

Session 3：navigation/crafting 混合主任务
→ adaptive B/D/E
```

顺序条件：

```text
navigation-first
crafting-first
```

新版方案去掉最终地图还原任务。理由是 adaptive B/D/E 更直接服务 chunk vs landmark 的核心判别，而地图还原更像额外的显性空间报告。

## 5. 旧版数据和新版数据的区别

当前已有 P003、P005 原始 JSON 是上一版数据：

```text
Session 1/2：固定 B/D/E
Session 3：混合主任务
最终：地图还原
无 adaptive B/D/E
```

后续正式开发和新采集默认使用新版：

```text
Session 3 后根据该被试行为在线生成 adaptive B/D/E
不再默认使用最终地图还原
```

因此旧数据可用于：

- 检查数据读取和预处理；
- 验证行为效率、RT、固定 B/D/E、地图还原等可视化；
- 用已有主任务行为估算 adaptive B/D/E 在线生成耗时；
- 作为迁移旧可视化原型的测试材料。

旧数据不能用于：

- 直接分析 adaptive B/D/E 作答；
- 验证 adaptive item 的真实被试反应；
- 替代新版采集中的 adaptive 证据。

## 6. 固定 B/D/E 的定位

固定 B/D/E 出现在 Session 1 和 Session 2 后，用于在纯 domain 阶段收集基础主观表征证据。

| 子任务 | 表面问题 | 主要含义 |
|---|---|---|
| B | 两个节点之间多容易到达 | 心理距离 / 主观可达性 |
| D | 一个节点序列多像熟悉路线或加工工序 | 序列整体感 / chunk fluency |
| E | 起点到终点之间最理想的中转节点 | landmark / hub 偏好 |

固定 B/D/E 的题池是预设的，不针对被试个人行为在线调整。它适合做横向比较和基线测量，但对个体 chunk-vs-landmark 冲突的判别力有限。

## 7. Adaptive B/D/E 的定位

adaptive B/D/E 出现在 Session 3 后。核心原则：

```text
主任务行为数据用于提出个体化假设；
adaptive B/D/E 用于独立验证这些假设。
```

adaptive 题目不应只是重放被试走过的高频路线，而应包含：

- 候选 chunk / landmark 的验证题；
- 匹配控制题；
- chunk 与 landmark 预测不一致的冲突题；
- 尽量匹配客观距离、序列长度和呈现复杂度的 probes。

## 8. Adaptive 数据窗口

adaptive 生成只使用正式主任务数据，不使用：

- 练习阶段；
- 无效按键；
- 固定 B/D/E 作答；
- adaptive B/D/E 作答；
- 地图还原。

数据按 domain 分开排序和截取：

```text
对每个 domain：
1. 取该 domain 的正式主任务 trial；
2. 按实际发生顺序排序；
3. 去掉最早 ceil(N × 0.2) 个 trial；
4. 使用剩余后 80% trial 做 chunk mining 和 landmark mining。
```

当前程序中保存的规则名为：

```text
ceil_drop_first_20_percent_keep_last_80_percent
```

这个规则的目的不是追求样本量最大，而是保证早期探索期被充分排除。若每个 domain 有 30 个 trial，则去掉前 6 个，使用后 24 个。

## 9. Adaptive chunk mining

当前优化方向是让实验端 adaptive 生成尽量等同于 Python 文档/模块中的：

```text
BDeu chunk mining + bootstrap stability
```

实验端当前实现：

```text
state_conditioned_bdeu_recursive_chunk_mining
```

基本流程：

1. 提取每个 domain 后 80% trial 的有效 action sequence 和 state sequence；
2. trial 之间加入结构分隔符 `S`，避免跨 trial 合成 chunk；
3. 对相邻 token 的二元合成候选 `u-v` 做 BDeu 依赖评分；
4. 比较 `C^v <- P^u, state` 与 `C^v <- state`；
5. 过滤低频、低 lift、低 Bayes factor 的候选；
6. 递归合成更长 chunk，当前建议最大 primitive 长度为 3；
7. 每轮回到原始动作序列做 Trie 最长匹配重解析；
8. 用 JS 距离收敛或最大迭代数停止；
9. 对 trial 做 subsampling bootstrap，得到 `bootstrap_stability`。

当前实验端默认参数：

| 参数 | 当前值 |
|---|---:|
| 最大 primitive chunk 长度 | 3 |
| chunk bootstrap iterations | 80 |
| BDeu equivalent sample size | 1.0 |
| 最小 log BF | `log(3)` |
| 最小 joint probability | 0.03 |
| score ratio threshold | 0.9 |

每个 chunk 候选保留：

```text
chunk
actions
states
count
support_trials
score
log_bayes_factor
joint_count
bootstrap_stability
state_realizations
source
```

## 10. Adaptive landmark mining

当前优化方向是让实验端 adaptive 生成尽量等同于 Python 文档/模块中的：

```text
bootstrap landmark mining
```

实验端当前实现：

```text
rank_aggregation_coverage_commonality_betweenness_bootstrap
```

基本流程：

1. 提取每个 domain 后 80% trial 的 state sequence；
2. 建立经验有向转移计数；
3. 计算节点级特征：
   - `coverage`
   - `path_commonality`
   - weighted directed `betweenness`
4. 将各特征转成 percentile rank；
5. 对 rank 求平均得到综合 score；
6. 对 trial 做 0.8 subsampling bootstrap；
7. 每次 bootstrap 选择 top landmark；
8. 用 selection rate 作为稳定性。

当前实验端默认参数：

| 参数 | 当前值 |
|---|---:|
| landmark bootstrap iterations | 500 |
| bootstrap sample ratio | 0.8 |
| selection threshold | 0.7 |
| max candidates | 4 |

每个 landmark 候选保留：

```text
node
score
selection_rate
bootstrap_stability
coverage
path_commonality
betweenness
source
```

## 11. Adaptive B/D/E 题目结构

Adaptive D：chunk 主观整体感。

| 条件 | 目的 |
|---|---|
| observed chunk | 行为中真实出现的候选 chunk 是否有整体感 |
| generalized chunk | 同一 action pattern 是否可泛化到其他起点 |
| matched control | 排除起终点熟悉或长度因素 |
| landmark-conflict sequence | 区分 chunk 整体感和 landmark 路线感 |

Adaptive E：landmark / hub 选择。

| 条件 | 目的 |
|---|---|
| landmark-on-path | landmark 在合理路径上时是否被选 |
| landmark-off-path lure | landmark 不在最短路上时是否仍有偏置 |
| chunk-vs-landmark conflict | chunk 预测与 landmark 预测冲突时的偏好 |
| neutral control | 中性基线 |

Adaptive B：方向性心理距离。

| 条件 | 解释 |
|---|---|
| C+L+ | chunk 和 landmark 都预测近 |
| C+L- | chunk-only 压缩 |
| C-L+ | landmark-only 压缩 |
| C-L- | matched control |

由于 `w` 是有向 shortcut，adaptive B 建议使用：

```text
从 A 到 B 多容易到达？
```

而不是完全对称的：

```text
A 和 B 之间多容易到达？
```

## 12. 题量

当前实验端默认：

```text
items_per_phase_domain = 8
```

即：

| domain | B | D | E | 小计 |
|---|---:|---:|---:|---:|
| navigation | 8 | 8 | 8 | 24 |
| crafting | 8 | 8 | 8 | 24 |
| 总计 | 16 | 16 | 16 | 48 |

若需要缩短实验，可降到每 domain 每 phase 6 题，总计 36 题。若疲劳风险较高，优先保留：

- E conflict items；
- D generalized chunk items；
- B 的 C+L- / C-L+ matched items。

## 13. 在线生成耗时

用已有 P003、P005 旧版数据模拟生成真正 adaptive B/D/E 时，实验端 JavaScript 本地基准约为：

| 数据 | 平均耗时 | 最大耗时 |
|---|---:|---:|
| P003 | 约 0.84 s | 约 1.1 s |
| P005 | 约 1.00 s | 约 1.1 s |

当前 Session 3 后休息页设置最短 5 秒，并在后台生成 adaptive pack。因此在目前数据规模下，adaptive 生成预计不会成为被试等待瓶颈。

这个估计基于当前两名旧版被试数据和当前 JS 实现；正式大样本采集前仍建议保留 `background_task_done`、`background_error` 和 `adaptive_pack_summary`，以便发现异常被试或浏览器性能问题。

## 14. 数据保存要求

每个 adaptive pack 必须保存：

```text
schema
version
participant_id
experiment_order
source_data_window
window_rule
items_per_phase_domain
domains
trials
```

每个 adaptive 结果必须保存：

```text
phase
domain
condition_label
target_construct
candidate_chunk_id
candidate_landmark_id
chunk_confidence
landmark_confidence
presented_nodes / presented_sequence / presented_start / presented_end
rating 或 chosen_hub
rt_ms
```

这样后续可以追溯：

- 题目为什么生成；
- 候选 chunk / landmark 的证据强度；
- 被试回答是否符合 chunk 预测、landmark 预测或控制条件；
- 某个被试是否落入 fallback / low-confidence adaptive generation。

## 15. 分析和可视化主线

正式分析主线：

```text
data/raw_data 或 data/real_data_pipeline/raw
→ script/real_data_pipeline/preprocess_human_data.py
→ script/real_data_pipeline/infer_chunk_human.py
→ script/real_data_pipeline/infer_landmark_human.py
→ src/cognitivemap.map_estimation / generative_model
→ script/visualization
→ result/figure
```

旧 `experiments/新数据` 的定位：

```text
历史原型 / 迁移参考 / 可删除候选
```

当前已经迁移到 `script/visualization` 的旧版图：

- 路径效率；
- 反应时；
- 固定 B 热图和距离关系；
- 固定 D 整体感；
- 固定 E hub-on-shortest-path；
- 旧版地图还原。

尚可按需要继续迁移的旧图思路：

- transition heatmap；
- TPM convergence；
- macro catalog；
- chunk usage；
- chunk vs efficiency；
- behavioral cognitive map HTML/PNG/JSON；
- exploration coverage。

如果这些旧图不再服务后续 adaptive B/D/E 论文主线，则 `experiments/新数据` 可以删除；如果还要用于汇报或补充材料，应先迁移或归档。

## 16. 当前实现结论

当前最合理的开发基线是：

```text
adaptive B/D/E 生成方向 = BDeu chunk mining + bootstrap landmark mining
数据窗口 = 每个 domain 向上取整去掉前 20%，使用后 80%
分析入口 = src + script/real_data_pipeline + script/simulation_recovery_pipeline
可视化入口 = script/visualization
图表输出 = result/figure
旧 experiments/新数据 = 可删除候选，不再作为正式依赖
```
