# Session 3 后 adaptive B/D/E 题目生成方案

## 1. 最终实验流程

当前推荐流程为：

```text
Session 1：纯 navigation 或纯 crafting 主任务
→ 固定 B/D/E 题目

Session 2：另一个纯任务域主任务
→ 固定 B/D/E 题目

Session 3：navigation/crafting 混合主任务
→ adaptive B/D/E 题目
```

本方案去掉最终地图还原任务，将 Session 3 后的测量重点转向更直接服务科学问题的 adaptive B/D/E 诊断题。

核心科学问题：

```text
被试在形成认知地图和完成路径规划时，
更依赖 action chunk，还是更依赖 landmark / hub？
这种表征是否在 navigation 与 crafting 两个任务域中一致？
```

因此 adaptive 题目必须同时覆盖：

```text
navigation chunk
navigation landmark
crafting chunk
crafting landmark
```

也就是一个完整的 2 × 2 证据框架：

| 任务域 | chunk 证据 | landmark 证据 |
|---|---|---|
| navigation | navigation 行为中提取的动作组块 | navigation 行为中提取的地标/枢纽 |
| crafting | crafting 行为中提取的动作组块 | crafting 行为中提取的地标/枢纽 |

## 2. adaptive 题目的基本原则

adaptive 题目的生成遵循一个核心原则：

```text
主任务行为数据用于提出个体化假设；
adaptive B/D/E 用于独立验证这些假设。
```

因此，不能只把被试刚刚走过的高频路线重新呈现给他评分。题目必须包含：

- 候选 chunk / landmark 的验证题；
- 与候选结构匹配的控制题；
- chunk 与 landmark 预测不一致的冲突题；
- 同客观距离、同长度、同呈现复杂度的 matched probes。

这样才能避免把熟悉度、走过次数、路径长度误判为 chunk 或 landmark 表征。

## 3. 用于生成 adaptive 题目的数据

Session 3 结束后，对该被试已经完成的正式主任务数据进行在线分析。

纳入数据：

```text
Session 1 主任务 trial
Session 2 主任务 trial
Session 3 主任务 trial
```

排除数据：

```text
练习阶段
探索阶段
无效按键
Session 1/2 固定 B/D/E 答题结果
任何 adaptive 答题结果
```

数据按 domain 分开处理：

```text
navigation 数据
→ 提取 navigation chunks
→ 提取 navigation landmarks

crafting 数据
→ 提取 crafting chunks
→ 提取 crafting landmarks
```

即使两个任务域底层拓扑同构，也不能将两个 domain 混合后统一提取。我们关心的是同构结构在不同语义域中是否形成了相同或不同的内部表征。

## 4. 后 80% trial 的使用规则

为了降低早期探索噪音，候选 chunk 和 landmark 只从每个 domain 的后 80% 正式 trial 中提取。

具体规则：

```text
对每个 domain：
1. 将该 domain 的正式主任务 trial 按实际发生时间排序；
2. 去掉最早 20% trial；
3. 使用剩余后 80% trial 提取 chunk 和 landmark。
```

理由：

- 前 20% trial 可能包含规则摸索、试错和偶然按键；
- 早期重复动作可能形成伪 chunk；
- 早期频繁经过的节点可能只是迷路造成的伪 landmark；
- 后 80% 更接近被试稳定学习后的表征表达。

若某个 domain trial 数为 30，则去掉最早 6 个 trial，使用后 24 个 trial。若 trial 数无法整除，建议使用向上取整去掉前 20%，保证探索期被充分排除。

## 5. 候选 chunk 的提取

chunk 提取基于后 80% trial 中的动作序列和状态序列。

每个 trial 需要保留：

```text
trial_id
domain
start
goal
state_sequence
action_sequence
valid
rt_ms
path_efficiency
```

### 5.1 当前 chunk 搜索的数学对象

对某个被试、某个 domain，记后 80% 正式 trial 为：

```text
D_domain = {τ_1, τ_2, ..., τ_N}
```

第 n 个 trial 的行为流写成：

```text
τ_n = (s_{n,0}, a_{n,1}, s_{n,1}, ..., a_{n,T_n}, s_{n,T_n})
```

当前代码中，chunk mining 主要读取动作序列，也可以把动作发生前的状态作为条件变量。动作会先被映射到统一 token 空间：

```text
R, D, L, U, P
```

其中 `P` 表示实验中的 `LOOP` 动作。默认脚本会把同一 domain 内的多个 trial 拼接成一条长序列，并在 trial 之间插入结构分隔符 `S`：

```text
X = a_{1,1}, ..., a_{1,T_1}, S, a_{2,1}, ..., a_{2,T_2}, S, ...
```

`S` 只用于保留 trial 边界，不参与 chunk 候选评分，也不允许和真实动作合并成 chunk。若使用状态条件变量，则与 `S` 对齐的位置记为 `-1`，同样不作为真实状态参与评分。

算法第 r 轮维护一条当前解析序列：

```text
X^(r) = x_1^(r), x_2^(r), ..., x_L^(r)
```

其中每个 `x_t^(r)` 既可能是原子动作，也可能是上一轮已经学到的复合 chunk。当前实现每一轮只搜索有序二元合成：

```text
g = u-v
```

也就是问：当前词表中的 token `u` 后面是否稳定跟着 token `v`。更长的 chunk 不是一次性穷举所有 n-gram，而是通过递归合成得到，例如先学到 `R-P`，下一轮再把 `R-P` 和 `D` 合成 `R-P-D`。在线生成 adaptive 题目时建议保留当前脚本默认约束：

```text
primitive token length(g) <= 3
g 不包含结构分隔符 S
```

### 5.2 BDeu 依赖评分

对每个候选有序对 `(u, v)`，代码把相邻时间步转成二值变量：

```text
P_t^u = 1[x_t = u]
C_t^v = 1[x_{t+1} = v]
```

如果启用状态条件，还会加入当前动作发生前的状态变量：

```text
Q_t = s_t
```

当前有两种状态编码：默认版本把状态展开成 9 个 one-hot 父变量；离散状态版本把状态作为 1 个 9 值离散父变量。两者的目的相同：控制“某个动作对只是因为当前状态固定而出现”的解释。

候选 chunk 的核心评分是比较两个 Bayesian network：

```text
依赖模型 M1：C_t^v <- P_t^u, Q_t
独立模型 M0：C_t^v <- Q_t
```

若不使用状态变量，则 `Q_t` 从两个模型中同时去掉，变成：

```text
M1：C_t^v <- P_t^u
M0：C_t^v 无父节点
```

用 BDeu 边际似然计算：

```text
log BF(u, v) =
log p_BDeu(C^v | P^u, Q; α)
- log p_BDeu(C^v | Q; α)
```

这里 `α` 是 equivalent sample size。对一个离散 child 变量 `Y`，若父节点组合数为 `q`，child 状态数为 `r`，BDeu log marginal likelihood 为：

```text
log p_BDeu(Y | Π; α)
= Σ_j [
    log Γ(α / q) - log Γ(α / q + N_j)
    + Σ_k { log Γ(α / (q r) + N_jk) - log Γ(α / (q r)) }
  ]
```

其中 `N_jk` 是父配置 `j` 下 child 取值 `k` 的计数，`N_j = Σ_k N_jk`。在当前 chunk 搜索里，child 是二值变量，所以 `r = 2`。

直观上，`log BF(u, v)` 问的是：在控制状态 `Q_t` 后，知道当前位置是不是 `u`，是否还能显著提高对下一位置是不是 `v` 的解释力。如果能，`u-v` 才有资格作为 action chunk 候选。

### 5.3 频率过滤、提升度过滤和候选选择

BDeu 只回答统计依赖强不强。为了避免极低频偶然对被误选，代码还会计算经验频率：

```text
N_uv = Σ_t 1[x_t = u, x_{t+1} = v]
N_u  = Σ_t 1[x_t = u]
N_v  = Σ_t 1[x_t = v]
L_eff = 非结构 token 总数

p(u, v) = N_uv / L_eff
p(u)    = N_u  / L_eff
p(v)    = N_v  / L_eff
```

候选 `(u, v)` 必须同时满足：

```text
p(u, v) >= p(u) p(v)
p(u, v) >= p_min
log BF(u, v) > τ_BF
```

第一条相当于 lift 不低于 1，要求二者共同出现不能少于独立模型预期；第二条是最小联合频率阈值；第三条要求 Bayes factor 达到绝对证据阈值。当前脚本中常用的 `p_min` 默认约为 `0.05`，`τ_BF` 的类默认值为 `log(3)`。

通过过滤后，每个候选的选择分数不是原始 `log BF`，而是单位样本证据：

```text
score(u, v) = log BF(u, v) / N_pair
```

其中 `N_pair` 是进入 BDeu 评分的有效相邻 pair 数。这样可以避免数据量稍大时，某个总 `log BF` 很大的候选压过所有其他同样稳定的候选。

设本轮最高分为：

```text
score_max = max_{(u,v)} score(u,v)
```

保留候选集合：

```text
G_new = { u-v : score(u,v) - score_max > log ρ }
```

等价地说，候选的单位样本 BF 至少要达到最优候选的 `ρ` 倍。当前人类数据脚本默认 `ρ = 0.9`。

### 5.4 重新解析与停止条件

本轮选出的 `G_new` 会加入词表，然后代码不是在当前解析序列上局部替换，而是回到原始动作序列 `X` 上重新解析。解析器使用 Trie 做从左到右最长匹配：

```text
在当前位置 i，选择词表中能匹配 X_i, X_{i+1}, ... 的最长 chunk；
如果没有复合 chunk 命中，则退回为单个原子 token。
```

这一步保证所有 chunk 都在同一条原始行为流上竞争解释权。若一个 chunk 覆盖多个动作，则 chunk 级状态取该 chunk 第一个动作执行前的状态，用于下一轮继续做状态条件 BDeu。

重新解析后，计算旧解析分布与新解析分布之间的 Jensen-Shannon 距离：

```text
d_JS(P_old, P_new)
= sqrt( 1/2 KL(P_old || M) + 1/2 KL(P_new || M) )
M = 1/2 (P_old + P_new)
```

算法在以下任一情况下停止：

```text
本轮没有任何候选通过过滤；
最近若干轮 d_JS 的平均值低于阈值；
达到最大迭代次数。
```

### 5.5 chunk 稳定性和 adaptive 候选置信度

当前核心 chunk miner 负责从给定序列中学习 chunk，本身不直接做 bootstrap。若 adaptive 题目生成需要 `bootstrap_stability`，建议在外层对后 80% trial 做 subsampling：

```text
for b = 1 ... B:
    从 D_domain 中无放回抽取 ceil(0.8 N) 个 trial
    运行同一套 chunk mining
    记录学到的 learned_chunks

stability(g) = (1 / B) Σ_b 1[g ∈ learned_chunks_b]
```

最终候选 chunk 的排序建议优先考虑：

```text
1. stability(g)
2. 全数据 score 或 log BF
3. 全数据 occurrence_count
4. mean_path_efficiency
```

其中 `occurrence_count` 应从最终最长匹配后的 parsed sequence 中统计；`state_realizations` 可由 parsed chunk 的原始起始位置回溯到对应的状态片段得到。

候选 chunk 的选择标准：

| 标准 | 含义 |
|---|---|
| 高频 | 在后 80% trial 中出现次数足够多 |
| 稳定 | bootstrap 子样本中反复被提取 |
| 有效 | 多出现在成功或高效路径中 |
| 可解释 | 可以映射回明确的节点序列 |
| 非早期噪音 | 不依赖前 20% trial 的偶然探索 |

每个 domain 最多保留 top 1-3 个候选 chunk。每个候选 chunk 记录：

```text
chunk_id
domain
action_pattern
state_realizations
start_states_observed
end_states_observed
occurrence_count
bootstrap_stability
mean_rt_ms
mean_path_efficiency
```

示例：

```text
domain: navigation
action_pattern: W-W
state_realizations:
  - 1→3→9
  - 3→9→7
occurrence_count: 8
bootstrap_stability: 0.82
```

## 6. 候选 landmark 的提取

landmark 提取基于后 80% trial 中的状态序列。

候选 landmark 不应只由 visit frequency 决定，重点应放在节点是否承担路线组织功能。

建议指标：

| 指标 | 含义 |
|---|---|
| coverage | 出现在多少 trial 中 |
| interior frequency | 作为中间节点出现的比例 |
| path commonality | 是否服务多个不同 start-goal pair |
| betweenness | 是否连接不同路径区域 |
| bootstrap stability | 子样本中是否稳定被选为 top landmark |

注意：

```text
起点/终点频繁出现不等于 landmark。
```

因此 landmark 提取时应弱化或排除 endpoint 贡献，优先计算 interior role。

### 6.1 当前 landmark 搜索的数学对象

对某个被试、某个 domain，landmark mining 只读取后 80% trial 的状态序列：

```text
S_n = (s_{n,0}, s_{n,1}, ..., s_{n,T_n})
```

候选集合是该 domain 中实际出现过的状态节点：

```text
V = {i : i 在任一 S_n 中出现}
```

当前 landmark miner 是纯数据驱动的候选筛选器：

```text
不读取 chunk mining 结果；
不拟合 landmark CPD；
不假设某个节点一定是 landmark。
```

它的目标不是找访问次数最高的点，而是找在经验路径网络中承担组织功能的点。具体做法是先把状态序列压成若干计数矩阵和节点特征，再用 rank aggregation 和 subsampling stability 选出稳定候选。

### 6.2 经验转移图和基础计数

首先计算有向经验转移计数：

```text
N_ij = Σ_n Σ_t 1[s_{n,t} = i, s_{n,t+1} = j]
```

以及节点访问计数：

```text
V_i = Σ_n Σ_t 1[s_{n,t} = i]
```

trial 覆盖数定义为：

```text
C_i = Σ_n 1[i ∈ S_n]
coverage(i) = C_i / N
```

这里 `coverage(i)` 衡量某个节点是否跨 trial 反复出现，而不是在单条很长路径里被来回踩很多次。

对 start-goal pair 的共同路径角色，当前代码按不同起终点 pair 聚合。记：

```text
P = {(s_{n,0}, s_{n,T_n}) : n = 1...N}
```

对每个 pair `p = (start, goal)`，先收集所有属于该 pair 的 trial 中出现过的中间节点：

```text
I_p = union over n with pair p of {s_{n,1}, ..., s_{n,T_n-1}}
```

默认排除 trial 的起点和终点。然后定义：

```text
M_i = Σ_{p ∈ P} 1[i ∈ I_p]
path_commonality(i) = M_i / |P|
```

因此，一个节点只有在多个不同 start-goal pair 的中间路径中反复承担中转角色，`path_commonality` 才会高。若只是频繁作为任务起点或终点出现，默认不会靠这一项得分。

如果需要单独输出 `interior_frequency`，可按同一思想定义：

```text
interior_frequency(i)
= Σ_n Σ_{t=1}^{T_n-1} 1[s_{n,t}=i]
  / max(Σ_n max(T_n - 1, 0), 1)
```

当前代码默认不把它作为独立排序特征，而是通过 endpoint-excluded `path_commonality` 实现对 interior role 的偏重。

### 6.3 加权 betweenness：把高概率转移看成低成本边

landmark 的核心图论指标是 weighted directed betweenness。先把经验转移计数转成每个 source 的转移概率：

```text
p_ij = N_ij / Σ_k N_ik
```

若 `p_ij > 0`，定义经验图上的边成本：

```text
w_ij = max(-log(max(p_ij, ε)), c_min)
```

若 `p_ij = 0`，则视为没有这条边，`w_ij = ∞`。这个定义的含义是：被试越常使用的转移，成本越低；很少或从未使用的转移，不会轻易成为经验最短路的一部分。

在这个加权有向图上，对每个节点 `i` 计算 Brandes betweenness：

```text
betweenness(i)
= [1 / ((|V|-1)(|V|-2))]
  Σ_{s≠i≠t} σ_st(i) / σ_st
```

其中 `σ_st` 是从 `s` 到 `t` 的加权最短路径条数，`σ_st(i)` 是这些最短路径中经过 `i` 的条数。若一个节点连接了多个高概率路径区域，它会得到较高 betweenness。这个指标对应“节点是否像枢纽一样组织路线”，比单纯 visit frequency 更接近 landmark/hub 的构念。

### 6.4 可选 boundary 指标

当前 landmark miner 还提供一个可选的 `boundary` 特征，用于捕捉“多方向进入、多方向离开”的边界节点。它由入边熵、出边熵和入出度构成：

```text
H_in(i)  = -Σ_j p(j | i as target) log p(j | i as target)
H_out(i) = -Σ_j p(j | i as source) log p(j | i as source)

boundary(i)
= H_in(i) + H_out(i)
  + (in_degree(i) + out_degree(i)) / (|V| - 1)
```

默认特征集合暂时不包含 `boundary`，因为它更像“分叉/汇合复杂度”，不一定等同于 landmark。但在某些被试中，如果路线明显围绕交汇点展开，可以把它作为敏感性分析特征。

### 6.5 percentile-rank aggregation

不同特征的量纲不同，当前代码不直接手动加权原始值，而是先把每个特征转成百分位秩：

```text
rank_m(i) ∈ [0, 1]
```

其中 `m` 表示某个特征，例如 `coverage`、`path_commonality`、`betweenness`。特征值越大，百分位秩越接近 1；并列值使用平均秩。

综合分数定义为：

```text
score(i) = mean_{m ∈ features} rank_m(i)
```

当前默认：

```text
features = (coverage, path_commonality, betweenness)
```

所以一个节点会在三种意义上同时被偏好：

```text
跨 trial 出现稳定；
服务多个 start-goal pair 的中间路线；
在经验转移图上连接不同路径区域。
```

全数据 top landmark 由 `score(i)` 排序得到：

```text
top_landmarks = top_k_i score(i)
```

### 6.6 subsampling 稳定性选择

为了避免把某一次路径偶然造成的高分节点当成 landmark，当前 landmark miner 会做 trial subsampling：

```text
for b = 1 ... B:
    从 N 个 trial 中无放回抽取 ceil(r N) 个 trial
    重新计算所有节点 score_b(i)
    选出本轮 top landmark 集合 L_b
```

默认 `B = 500`，`r = 0.8`。本轮 `L_b` 的选择可以用固定 top-k，也可以用 elbow 规则。固定 top-k 时：

```text
L_b = top_{max_landmarks} score_b(i)
```

稳定性定义为：

```text
selection_rate(i)
= (1 / B) Σ_b 1[i ∈ L_b]
```

最终稳定 landmark 候选为：

```text
landmarks
= {i : selection_rate(i) >= θ_stability}
```

再按以下顺序截取 `max_landmarks` 个：

```text
1. selection_rate(i) 从高到低
2. full-data score(i) 从高到低
3. 节点 label 作为稳定排序兜底
```

当前默认 `θ_stability = 0.7`。这意味着，一个节点即便全数据分数很高，如果在大多数 trial 子样本中不稳定，也不会进入最终 `landmarks`，但仍会保留在 `top_landmarks` 和 `candidate_ranking` 中供人工检查。

每个 domain 最多保留 top 1-3 个候选 landmark。每个候选 landmark 记录：

```text
landmark_id
domain
node
coverage
interior_frequency
path_commonality
betweenness
bootstrap_stability
related_start_goal_pairs
```

## 7. adaptive D：主观 chunk 验证题

D 题的表面任务：

```text
呈现一串节点序列，让被试判断它多大程度上像一条线路/一道加工工序。
```

D 题主要服务 chunk 证据。每个 domain 的 D 题围绕该 domain 的候选 chunk 生成。

### 7.1 D 题类型

| 类型 | 设计逻辑 | 目的 |
|---|---|---|
| observed chunk | 被试后 80% 中真实高频出现的 chunk 序列 | 检查行为 chunk 是否有主观整体感 |
| generalized chunk | 同一动作串，换到较少见或未见起点 | 检查 chunk 是否是可泛化动作单元 |
| endpoint-matched control | 起终点相近或相同，但动作串不同 | 排除只是起终点熟悉 |
| landmark-conflict sequence | 经过候选 landmark，但不符合候选 chunk | 区分 chunk 整体感与 landmark 路线感 |

### 7.2 D 题生成规则

对每个候选 chunk：

1. 从真实出现过的 state realization 中选 1-2 条 observed chunk 题；
2. 在任务图上寻找同一 action pattern 的其他可执行 realization，生成 generalized chunk 题；
3. 为每条 chunk 题生成长度匹配、客观距离匹配的 control sequence；
4. 若存在候选 landmark，则生成经过 landmark 但不匹配 chunk action pattern 的 conflict sequence。

示例：

```text
若提取到 chunk = W-W：

observed chunk:
  1→3→9

generalized chunk:
  3→9→7
  9→7→1

matched control:
  1→2→3
  3→6→9

landmark-conflict:
  起终点/长度相近，但中间经过候选 landmark，且动作模式不是 W-W
```

### 7.3 D 题指标

```text
chunk_validation_score =
mean(rating_observed_chunk) - mean(rating_matched_control)

chunk_generalization_score =
mean(rating_generalized_chunk) - mean(rating_matched_control)

chunk_specificity_score =
mean(rating_chunk_consistent) - mean(rating_landmark_conflict)
```

解释：

- observed chunk 高于 control：行为 chunk 有主观整体感；
- generalized chunk 高于 control：支持可迁移的 action chunk；
- 只有 observed 高、generalized 不高：可能只是熟悉路线；
- landmark-conflict 高：可能不是 chunk，而是 landmark 路线感。

## 8. adaptive E：landmark / hub 验证题

E 题的表面任务：

```text
呈现起点和终点，让被试选择心里最理想的中转节点。
```

E 题主要服务 landmark / hub 证据，也承担 chunk-vs-landmark 的关键判别功能。

### 8.1 E 题类型

| 类型 | 设计逻辑 | 目的 |
|---|---|---|
| landmark-on-path | 候选 landmark 位于合理最短/高效路径上 | 检查 landmark 是否自然被选为中转 |
| landmark-off-path lure | 候选 landmark 不在最短路上，但作为选项出现 | 检查是否存在 landmark 偏置 |
| chunk-vs-landmark conflict | chunk 预测中转与 landmark 预测中转不同 | 区分 chunk-route 与 landmark-route |
| neutral control | 没有明显候选 landmark 或 chunk bias | 提供选择基线 |

### 8.2 E 题生成规则

对每个候选 landmark：

1. 找到若干 start-goal pair，使该 landmark 位于至少一条最短路径或高效路径上；
2. 找到若干 start-goal pair，使该 landmark 不在最短路径上，但仍可作为可选中转节点；
3. 若存在候选 chunk，寻找 chunk-predicted hub 与 landmark-predicted hub 不同的 pair；
4. 生成 neutral control pair，控制客观距离和可选节点数量。

E 题中候选项建议排除 start 和 goal，只呈现其余 7 个节点，与现有固定 E 题保持一致。

### 8.3 chunk-vs-landmark conflict 的定义

一条 E conflict 题需要满足：

```text
给定 start 和 goal：
chunk 模型预测的中转节点 ≠ landmark 模型预测的中转节点
```

chunk 预测可以来自：

- 候选 chunk 的起点、终点或边界节点；
- 被试后 80% 中对该 start-goal pair 或相似 pair 的常用 chunk 路径；
- 由 chunk 压缩后的最短 cognitive route。

landmark 预测可以来自：

- top landmark；
- 位于或接近高效路径上的候选 landmark；
- 该被试最稳定的 hub 节点。

若被试在 conflict 题中选择 chunk-predicted hub，更支持 chunk-route planning。若选择 landmark-predicted hub，更支持 landmark/hub planning。

### 8.4 E 题指标

```text
landmark_choice_rate =
P(choice == candidate_landmark)

on_path_landmark_rate =
P(choice == candidate_landmark | landmark on shortest/high-efficiency path)

off_path_landmark_bias =
P(choice == candidate_landmark | landmark off shortest path)

conflict_landmark_preference =
P(choice == landmark_predicted_hub | conflict item)

conflict_chunk_preference =
P(choice == chunk_predicted_hub | conflict item)
```

解释：

- landmark 只在最短路上被选：可能只是路线知识；
- landmark 不在最短路上仍被选：更像 landmark 偏置；
- conflict 中稳定选择 landmark：支持 landmark/hub 表征；
- conflict 中稳定选择 chunk 边界：支持 action-chunk 表征。

## 9. adaptive B：心理距离压缩题

B 题的表面任务：

```text
判断两个节点之间多容易到达。
```

B 题用于检验 chunk 和 landmark 是否改变被试的主观距离结构。

### 9.1 B 题 2 × 2 设计

每个 domain 内构造如下四类节点对：

| 条件 | chunk | landmark | 解释 |
|---|---|---|---|
| C+L+ | 有 chunk 关系 | 有 landmark 关系 | 两种结构都预测心理距离近 |
| C+L- | 有 chunk 关系 | 无 landmark 关系 | chunk-only 压缩 |
| C-L+ | 无 chunk 关系 | 有 landmark 关系 | landmark-only 压缩 |
| C-L- | 无 chunk 关系 | 无 landmark 关系 | matched control |

其中：

```text
C+ 表示节点对是候选 chunk 的端点、内部相邻点，或被候选 chunk 强连接；
L+ 表示节点对经过同一候选 landmark、靠近候选 landmark，或由候选 landmark 组织；
C-/L- 表示不满足相应结构，但客观距离和呈现复杂度匹配。
```

### 9.2 B 题方向性

由于任务中的 W / loop 是有方向的，adaptive B 建议至少一部分题改为方向性问题：

```text
从 A 到 B 多容易到达？
```

而不是完全对称的：

```text
A 和 B 之间多容易到达？
```

方向性 B 题更敏感于 action chunk 和 directed shortcut。若保留原有固定 B 的对称题，adaptive B 可以作为方向性扩展。

### 9.3 B 题生成规则

对每个 domain：

1. 根据 top chunk 生成 C+ 节点对；
2. 根据 top landmark 生成 L+ 节点对；
3. 组合出 C+L+、C+L-、C-L+、C-L- 四类条件；
4. 尽量匹配每类节点对的客观最短距离；
5. 平衡节点出现次数，避免某个候选 landmark 被过度提示；
6. 随机化呈现顺序。

### 9.4 B 题指标

```text
chunk_distance_compression =
mean(rating_C+L-) - mean(rating_C-L-)

landmark_distance_compression =
mean(rating_C-L+) - mean(rating_C-L-)

combined_distance_compression =
mean(rating_C+L+) - mean(rating_C-L-)
```

解释：

- C+L- 高：支持 chunk 造成心理距离压缩；
- C-L+ 高：支持 landmark 造成心理距离压缩；
- C+L+ 最高：可能两种表征叠加；
- 四类无差异：候选结构未显著进入主观距离判断。

## 10. adaptive 题量建议

为同时覆盖 domain × representation，推荐两种题量版本。

### 10.1 最小可行版本

| domain | B | D | E | 小计 |
|---|---:|---:|---:|---:|
| navigation | 6 | 6 | 6 | 18 |
| crafting | 6 | 6 | 6 | 18 |
| 总计 | 12 | 12 | 12 | 36 |

### 10.2 推荐稳健版本

| domain | B | D | E | 小计 |
|---|---:|---:|---:|---:|
| navigation | 8 | 8 | 8 | 24 |
| crafting | 8 | 8 | 8 | 24 |
| 总计 | 16 | 16 | 16 | 48 |

若担心 Session 3 后疲劳，优先保留：

```text
E conflict items
D generalized chunk items
B C+L- / C-L+ matched items
```

这些题对 chunk vs landmark 的区分力最高。

## 11. 每个 domain 内的最低覆盖要求

每个 domain 至少需要覆盖：

```text
1-2 个 top chunk
1-2 个 top landmark
chunk matched control
landmark matched control
chunk-vs-landmark conflict item
```

若某个被试在某个 domain 中无法稳定提取 chunk 或 landmark，则使用 fallback 规则：

```text
无法稳定提取 chunk：
  使用 group-level 或 task-structure-defined chunk candidates 生成 exploratory chunk probes

无法稳定提取 landmark：
  使用 data-driven top hub / graph-theoretic candidate 生成 exploratory landmark probes

二者都不稳定：
  呈现固定 adaptive-backup B/D/E 题池，并标记为 low-confidence adaptive generation
```

所有 adaptive 题目必须保存生成依据和置信度，避免事后解释时混淆。

## 12. 呈现与随机化约束

adaptive block 中应控制：

- navigation 与 crafting 题量相同；
- B/D/E 题量相同或接近；
- 不连续出现过多同一 domain；
- 不连续出现过多同一题型；
- 不连续重复同一节点或同一候选 landmark；
- control 与 target item 在客观距离、序列长度、节点数量上尽量匹配；
- 所有题目保存 condition label，供事后建模。

推荐呈现结构：

```text
adaptive block intro
→ mixed adaptive B/D/E trials
→ adaptive block outro
```

也可以按 B、D、E 分块呈现，但需要在每块内部混合 navigation/crafting，避免被试明显意识到某个 domain 或某个节点正在被重点考察。

## 13. 数据保存字段

每道 adaptive 题建议保存以下字段：

```text
participant_id
session_id
adaptive_version
domain
subphase               # B / D / E
trial_id
condition_label
target_construct       # chunk / landmark / conflict / control
source_data_window     # e.g. last_80_percent
candidate_chunk_id
candidate_landmark_id
chunk_confidence
landmark_confidence
stimulus_nodes
stimulus_actions
start
goal
options
correct_or_normative   # optional, not shown to participant
model_prediction_chunk
model_prediction_landmark
response
rt_ms
generation_notes
```

这些字段能支持后续将 adaptive 题目纳入模型比较，而不是只做描述统计。

## 14. 最终分析指标

最终至少形成四个核心指标：

```text
navigation_chunk_score
navigation_landmark_score
crafting_chunk_score
crafting_landmark_score
```

进一步可计算：

```text
domain_chunk_difference =
navigation_chunk_score - crafting_chunk_score

domain_landmark_difference =
navigation_landmark_score - crafting_landmark_score

representation_bias_navigation =
navigation_chunk_score - navigation_landmark_score

representation_bias_crafting =
crafting_chunk_score - crafting_landmark_score
```

核心解释目标：

- 被试是否在 navigation 中更依赖 chunk？
- 被试是否在 crafting 中更依赖 landmark？
- 两个 domain 是否共享同构表征？
- Session 3 混合后是否出现跨域迁移？
- 当 chunk 与 landmark 预测冲突时，被试更跟随哪一种表征？

## 15. 简要结论

最终 adaptive B/D/E 的定位是：

```text
B：检验心理距离是否被 chunk 或 landmark 压缩；
D：检验行为 chunk 是否具有主观整体感和泛化性；
E：检验 candidate landmark/hub 是否真正参与路线规划，并通过 conflict items 区分 chunk-route 与 landmark-route。
```

该方案比最终地图还原更直接服务 chunk vs landmark 的核心科学问题，同时保留了 navigation 与 crafting 两个任务域的可比性。
