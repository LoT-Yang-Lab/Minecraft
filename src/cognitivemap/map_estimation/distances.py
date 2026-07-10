# -*- coding: utf-8 -*-
"""
距离计算方法模块
================
本模块实现了 4 种基于多 trial 序列数据计算状态间距离的方法。

关键约束：
    - 所有统计均在单个 trial 内部进行，**不跨越 trial 边界**。
      即 trial A 的最后一个 state 与 trial B 的第一个 state 之间不存在转移关系。
    - 所有函数签名统一为：
      (trials: List[Trial], ..., state_filter=None) -> Tuple[np.ndarray, List[int]]

距离方法：
    1. Successor Representation Distance — TD 学习继承表示矩阵，行向量欧氏距离
    2. LoPS-based Distance — 转移概率 → sqrt(-log(P)) 边权重 → 最短路径 → 对称化
    3. Action Distribution JS Distance — 每个状态发出的 action 概率分布 → JS 散度
    4. Transition Similarity Distance — 每个状态的出边转移概率分布 → JS 散度
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon

# ============================================================================
# 数据类型
# ============================================================================


@dataclass(frozen=True)
class Trial:
    """单个 trial 的序列数据。

    Attributes:
        state_sequence: 状态序列，长度 N，每个元素为 int 类型的状态编号。
        action_sequence: 动作序列，长度 N-1，每个元素为 str 类型的动作名（如 "U", "D", "L", "R", "P"）。
    """

    state_sequence: List[int]
    action_sequence: List[str]

    def __post_init__(self):
        """验证序列长度一致性。"""
        if len(self.action_sequence) != len(self.state_sequence) - 1:
            raise ValueError(
                f"action_sequence 长度 ({len(self.action_sequence)}) 应等于 "
                f"state_sequence 长度 ({len(self.state_sequence)}) - 1"
            )


# ============================================================================
# 内部辅助函数
# ============================================================================


def _collect_unique_states(
    trials: List[Trial],
    state_filter: Optional[Callable[[List[int]], List[int]]] = None,
) -> List[int]:
    """收集所有 trial 中出现过的唯一状态，排序后返回。

    Args:
        trials: trial 列表。
        state_filter: 可选的有效状态筛选函数。为 None 时保留所有状态。

    Returns:
        排序后的唯一状态列表。
    """
    states_set: set = set()
    for trial in trials:
        states_set.update(trial.state_sequence)
    states = sorted(states_set)
    if state_filter is not None:
        states = state_filter(states)
    return states


def _build_state_index(states: List[int]) -> Dict[int, int]:
    """构建 state -> index 的映射字典。

    Args:
        states: 状态列表。

    Returns:
        {state: index} 映射。
    """
    return {s: i for i, s in enumerate(states)}


def build_transition_counts(
    trials: List[Trial],
    states: List[int],
) -> np.ndarray:
    """统计 trial 内部的转移计数矩阵。

    关键约束：只统计每个 trial 内部相邻状态对 (s_t, s_{t+1})，不跨越 trial 边界。

    Args:
        trials: trial 列表。
        states: 唯一状态列表。

    Returns:
        (n_states, n_states) 的转移计数矩阵，counts[i][j] 表示从状态 i 转移到状态 j 的次数。
    """
    state_to_idx = _build_state_index(states)
    n = len(states)
    counts = np.zeros((n, n), dtype=np.float64)

    for trial in trials:
        seq = trial.state_sequence
        # 只在 trial 内部遍历相邻状态对
        for t in range(len(seq) - 1):
            s_from = seq[t]
            s_to = seq[t + 1]
            if s_from not in state_to_idx or s_to not in state_to_idx:
                continue
            if s_from != s_to:  # 忽略自环（P 动作产生的自转移）
                i = state_to_idx[s_from]
                j = state_to_idx[s_to]
                counts[i, j] += 1

    return counts


def build_action_counts(
    trials: List[Trial],
    states: List[int],
) -> Tuple[np.ndarray, List[str]]:
    """统计每个状态下各动作的出现次数。

    关键约束：只统计每个 trial 内部的 (state[t], action[t]) 对，不跨越 trial 边界。

    Args:
        trials: trial 列表。
        states: 唯一状态列表。

    Returns:
        (action_counts, action_labels):
            action_counts: (n_states, n_actions) 矩阵
            action_labels: 动作标签列表
    """
    # 先收集所有出现过的动作
    actions_set: set = set()
    state_to_idx = _build_state_index(states)
    for trial in trials:
        for state, action in zip(trial.state_sequence[:-1], trial.action_sequence):
            if state in state_to_idx:
                actions_set.add(action)
    action_labels = sorted(actions_set)
    action_to_idx = {a: i for i, a in enumerate(action_labels)}

    n_states = len(states)
    n_actions = len(action_labels)
    counts = np.zeros((n_states, n_actions), dtype=np.float64)

    for trial in trials:
        seq = trial.state_sequence
        acts = trial.action_sequence
        # 只在 trial 内部遍历 (state[t], action[t]) 对
        for t in range(len(acts)):
            s = seq[t]
            a = acts[t]
            if s not in state_to_idx:
                continue
            i = state_to_idx[s]
            j = action_to_idx[a]
            counts[i, j] += 1

    return counts, action_labels


def build_edge_action_counts(
    trials: List[Trial],
    states: List[int],
) -> Dict[Tuple[int, int], Dict[str, int]]:
    """统计每条有向边上各动作的使用次数。

    对每个 trial 内部的 (state[t], action[t]) → state[t+1] 三元组，
    在 edge_actions[(state[t], state[t+1])][action[t]] 上累加计数。

    关键约束：只统计每个 trial 内部的转移，不跨越 trial 边界。

    Args:
        trials: trial 列表。
        states: 唯一状态列表。

    Returns:
        {(from_state_idx, to_state_idx): {action_name: count}} 嵌套字典。
        键为状态标签（int），值为动作名到计数的映射。
    """
    edge_actions: Dict[Tuple[int, int], Dict[str, int]] = {}
    state_to_idx = _build_state_index(states)

    for trial in trials:
        seq = trial.state_sequence
        acts = trial.action_sequence
        # 只在 trial 内部遍历 (state[t], action[t]) → state[t+1]
        for t in range(len(acts)):
            s_from = seq[t]
            s_to = seq[t + 1]
            a = acts[t]
            if s_from not in state_to_idx or s_to not in state_to_idx:
                continue
            edge = (s_from, s_to)
            if edge not in edge_actions:
                edge_actions[edge] = {}
            edge_actions[edge][a] = edge_actions[edge].get(a, 0) + 1

    return edge_actions


def _counts_to_probs(counts: np.ndarray) -> np.ndarray:
    """将计数矩阵按行归一化为概率分布。

    如果某行全为 0（该状态无转移记录），则该行保持全 0。

    Args:
        counts: (n, m) 计数矩阵。

    Returns:
        (n, m) 概率矩阵，每行之和为 1（或全 0）。
    """
    row_sums = counts.sum(axis=1, keepdims=True)
    # 避免除以 0：全零行保持 0
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(row_sums > 0, counts / row_sums, 0.0)
    return probs


def _js_distance_matrix(prob_matrix: np.ndarray) -> np.ndarray:
    """计算概率矩阵行之间的 Jensen-Shannon 距离矩阵。

    对每对行 (i, j)，计算 JS 散度的平方根作为距离。

    Args:
        prob_matrix: (n, d) 概率矩阵，每行为一个概率分布。

    Returns:
        (n, n) 距离矩阵，对角线为 0。
    """
    n = prob_matrix.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            # scipy 的 jensenshannon 返回的是 JS 距离（即 JS 散度的平方根）
            js_val = jensenshannon(prob_matrix[i], prob_matrix[j])
            # 如果两个分布都是全零，jensenshannon 返回 nan，设为 0
            if np.isnan(js_val):
                js_val = 0.0
            dist[i, j] = js_val
            dist[j, i] = js_val

    return dist


# ============================================================================
# 距离方法 1: Successor Representation Distance
# ============================================================================


def compute_sr_distance(
    trials: List[Trial],
    gamma: float = 0.98,
    alpha: float = 0.1,
    state_filter: Optional[Callable[[List[int]], List[int]]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """基于 Successor Representation 计算状态间距离矩阵。

    算法步骤：
        1. 收集所有 trial 中的唯一状态，建立 state -> index 映射。
        2. 初始化继承表示矩阵 M (n_states × n_states) 为全零。
        3. 对每个 trial 独立进行 TD 学习：
           - 遍历 trial 内部的每对相邻状态 (s_t, s_{t+1})（跳过自环 s_t == s_{t+1}）：
             - indicator = one_hot(s_t)
             - td_error = indicator + gamma * M[s_{t+1}] - M[s_t]
             - M[s_t] += alpha * td_error
           - trial 之间不产生更新（不在 trial 边界处计算转移）。
        4. 对 M 的行向量计算欧氏距离 → 距离矩阵。

    Args:
        trials: trial 列表。
        gamma: TD 学习的折扣因子，默认 0.98。
        alpha: TD 学习的学习率，默认 0.1。
        state_filter: 可选的有效状态筛选函数。

    Returns:
        (distance_matrix, state_labels):
            distance_matrix: (n_states, n_states) 距离矩阵。
            state_labels: 状态标签列表。
    """
    states = _collect_unique_states(trials, state_filter)
    state_to_idx = _build_state_index(states)
    n = len(states)

    # 初始化继承表示矩阵 M
    M = np.zeros((n, n), dtype=np.float64)

    # 对每个 trial 独立进行 TD 学习
    for trial in trials:
        seq = trial.state_sequence
        # 只在 trial 内部遍历相邻状态对
        for t in range(len(seq) - 1):
            s_t = seq[t]
            s_next = seq[t + 1]

            if s_t == s_next:  # 忽略自环，与 build_transition_counts 保持一致
                continue
            if s_t not in state_to_idx or s_next not in state_to_idx:
                continue

            i = state_to_idx[s_t]
            j = state_to_idx[s_next]

            # 构造 one-hot 向量表示当前状态
            indicator = np.zeros(n, dtype=np.float64)
            indicator[i] = 1.0

            # TD 更新
            td_error = indicator + gamma * M[j] - M[i]
            M[i] += alpha * td_error

    # 计算 M 行向量之间的欧氏距离
    distance_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(M[i] - M[j])
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

    return distance_matrix, states


# ============================================================================
# 距离方法 2: LoPS-based Distance（基于转移概率对数的最短路径距离）
# ============================================================================


def compute_lops_distance(
    trials: List[Trial],
    unreachable_distance: float = 100.0,
    state_filter: Optional[Callable[[List[int]], List[int]]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """基于转移概率对数的最短路径（LoPS）距离。

    算法步骤：
        1. 收集所有 trial 中的唯一状态。
        2. 对每个 trial 独立统计内部的转移计数（不跨 trial）。
        3. 归一化为转移概率矩阵 transition_probs。
        4. 边权重 D = sqrt(-log(transition_probs))，对角线设为 inf。
        5. 构建有向图，使用 Dijkstra 算法计算全源最短路径。
        6. 对称化：双向可达取均值，单向可达取可达方向的值。
        7. 不可达节点对的距离设为 unreachable_distance。

    Args:
        trials: trial 列表。
        unreachable_distance: 不可达节点对之间的默认距离，默认 100.0。
        state_filter: 可选的有效状态筛选函数。

    Returns:
        (distance_matrix, state_labels):
            distance_matrix: (n_states, n_states) 对称距离矩阵。
            state_labels: 状态标签列表。
    """
    states = _collect_unique_states(trials, state_filter)
    n = len(states)

    # 统计 trial 内部的转移计数
    transition_counts = build_transition_counts(trials, states)

    # 归一化为转移概率
    transition_probs = _counts_to_probs(transition_counts)

    # 计算边权重：D = sqrt(-log(P))
    # 对于概率为 0 的边，权重为 inf（不可达）
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_weights = np.where(transition_probs > 0, np.sqrt(-np.log(transition_probs)), np.inf)
    # 对角线设为 inf（不比较自身到自身的转移）
    np.fill_diagonal(edge_weights, np.inf)

    # 构建有向图
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(n):
            if edge_weights[i, j] != np.inf:
                G.add_edge(i, j, weight=edge_weights[i, j])

    # 使用全源 Dijkstra 计算最短路径长度。边权重 sqrt(-log(P)) 始终非负，
    # 因此不需要 Johnson 算法或负环 fallback。
    directed_dist = np.full((n, n), np.inf)
    for source, lengths in nx.all_pairs_dijkstra_path_length(G, weight="weight"):
        for target, dist in lengths.items():
            directed_dist[source, target] = dist

    # 对称化：双向可达取均值，单向可达取可达方向的值
    undirected_dist = np.where(
        np.isfinite(directed_dist) & np.isfinite(directed_dist.T),
        (directed_dist + directed_dist.T) / 2.0,  # 双向可达：取均值
        np.fmin(directed_dist, directed_dist.T),  # 单向可达：取有限值
    )

    # 将不可达节点对（inf）替换为默认距离
    undirected_dist = np.where(np.isinf(undirected_dist), unreachable_distance, undirected_dist)
    # 对角线设为 0
    np.fill_diagonal(undirected_dist, 0.0)

    return undirected_dist, states


# ============================================================================
# 距离方法 3: Action Distribution JS Distance
# ============================================================================


def compute_action_js_distance(
    trials: List[Trial],
    state_filter: Optional[Callable[[List[int]], List[int]]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """基于动作分布的 Jensen-Shannon 距离。

    算法步骤：
        1. 收集所有 trial 中的唯一状态。
        2. 对每个 trial 独立统计 (state[t], action[t]) 对的出现次数（不跨 trial）。
        3. 构建概率矩阵 P(action | state)，形状 (n_states, n_actions)。
        4. 对每对状态 (i, j)，计算其动作概率分布之间的 Jensen-Shannon 距离。

    Args:
        trials: trial 列表。
        state_filter: 可选的有效状态筛选函数。

    Returns:
        (distance_matrix, state_labels):
            distance_matrix: (n_states, n_states) 距离矩阵。
            state_labels: 状态标签列表。
    """
    states = _collect_unique_states(trials, state_filter)

    # 统计每个状态下的动作分布
    action_counts, _ = build_action_counts(trials, states)

    # 归一化为概率分布
    action_probs = _counts_to_probs(action_counts)

    # 计算 JS 距离矩阵
    distance_matrix = _js_distance_matrix(action_probs)

    return distance_matrix, states


# ============================================================================
# 距离方法 4: Transition Similarity Distance
# ============================================================================


def compute_transition_similarity_distance(
    trials: List[Trial],
    state_filter: Optional[Callable[[List[int]], List[int]]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """基于转移相似性的 Jensen-Shannon 距离。

    算法步骤：
        1. 收集所有 trial 中的唯一状态。
        2. 对每个 trial 独立统计内部的出边转移计数（不跨 trial）。
        3. 构建概率矩阵 P(s' | s)，形状 (n_states, n_states)。
        4. 对每对状态 (i, j)，计算其出边转移概率分布之间的 Jensen-Shannon 距离。

    Args:
        trials: trial 列表。
        state_filter: 可选的有效状态筛选函数。

    Returns:
        (distance_matrix, state_labels):
            distance_matrix: (n_states, n_states) 距离矩阵。
            state_labels: 状态标签列表。
    """
    states = _collect_unique_states(trials, state_filter)

    # 统计 trial 内部的转移计数
    transition_counts = build_transition_counts(trials, states)

    # 归一化为转移概率分布 P(s' | s)
    transition_probs = _counts_to_probs(transition_counts)

    # 计算 JS 距离矩阵
    distance_matrix = _js_distance_matrix(transition_probs)

    return distance_matrix, states
