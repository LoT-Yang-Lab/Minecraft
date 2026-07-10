# -*- coding: utf-8 -*-
"""
基于状态序列的 landmark 推断。

这个模块只根据玩家状态序列推断候选 landmark，不依赖生成模型。

核心流程：
1. 将 trial state sequences 编码成紧凑整数数组；
2. 用矩阵累计经验转移、访问覆盖、start-goal 中间点覆盖；
3. 计算 coverage / path_commonality / weighted betweenness 等特征；
4. 用 percentile-rank aggregation 得到弱参数综合分数；
5. 通过 trial subsampling stability selection 估计候选 landmark 稳定性。
"""

from __future__ import annotations

import heapq
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

DEFAULT_FEATURES = ("coverage", "path_commonality", "betweenness")


@dataclass(frozen=True)
class LandmarkMiningConfig:
    """landmark 推断参数。

    参数默认值尽量保持弱参数：
    - 特征用 rank aggregation，不手调权重；
    - landmark 数量通过 bootstrap selection rate 辅助判断；
    - max_landmarks 只是稀疏上限，不强制每个被试一定有这么多个 landmark。
    """

    features: Tuple[str, ...] = DEFAULT_FEATURES
    max_landmarks: int = 4
    min_landmarks: int = 1
    bootstrap_selection_mode: str = "fixed"
    bootstrap_iterations: int = 500
    bootstrap_sample_ratio: float = 0.8
    elbow_min_relative_drop: float = 0.15
    elbow_min_score: float = 0.0
    selection_threshold: float = 0.7
    random_state: int = 42
    n_jobs: int = 1
    exclude_trial_endpoints_for_commonality: bool = True
    eps: float = 1e-12
    min_edge_cost: float = 1e-9


def mine_landmarks(
    state_sequences: Sequence[Sequence[Any]],
    config: LandmarkMiningConfig | None = None,
) -> Dict[str, Any]:
    """根据 trial state sequences 推断候选 landmark。

    Args:
        state_sequences: 每个 trial 一条状态序列，长度至少为 1。
        config: landmark inference 参数。

    Returns:
        可 joblib/json 序列化的结果 dict。
    """

    config = config or LandmarkMiningConfig()
    _validate_config(config)

    encoded_sequences, state_labels = _encode_state_sequences(state_sequences)
    n_trials = len(encoded_sequences)
    n_states = len(state_labels)
    if n_trials == 0:
        raise ValueError("state_sequences must contain at least one non-empty trial")
    if n_states == 0:
        raise ValueError("state_sequences must contain at least one state")

    full_score = _score_encoded_sequences(encoded_sequences, n_states, config)
    top_indices = _top_k_indices(full_score["score"], config.max_landmarks)

    selection_counts = _bootstrap_selection_counts(encoded_sequences, n_states, config)
    selection_rates = selection_counts / max(config.bootstrap_iterations, 1)

    ranking_indices = sorted(
        range(n_states),
        key=lambda i: (-selection_rates[i], -full_score["score"][i], str(state_labels[i])),
    )
    stable_indices = [i for i in ranking_indices if selection_rates[i] >= config.selection_threshold]
    selected_indices = stable_indices[: config.max_landmarks]

    return {
        "landmarks": [state_labels[i] for i in selected_indices],
        "top_landmarks": [state_labels[i] for i in top_indices],
        "candidate_ranking": [
            {
                "state": state_labels[i],
                "score": float(full_score["score"][i]),
                "selection_rate": float(selection_rates[i]),
            }
            for i in ranking_indices
        ],
        "state_scores": _state_score_table(state_labels, full_score, selection_rates),
        "transition_counts": _matrix_to_nested_dict(state_labels, full_score["transition_counts"]),
        "config": asdict(config),
        "n_trials": n_trials,
        "n_states": n_states,
    }


def score_landmark_candidates(
    state_sequences: Sequence[Sequence[Any]],
    config: LandmarkMiningConfig | None = None,
) -> Dict[str, Any]:
    """只计算全数据 candidate score，不运行 bootstrap。"""

    config = config or LandmarkMiningConfig(bootstrap_iterations=0)
    _validate_config(config)
    encoded_sequences, state_labels = _encode_state_sequences(state_sequences)
    full_score = _score_encoded_sequences(encoded_sequences, len(state_labels), config)
    selection_rates = np.zeros(len(state_labels), dtype=np.float64)
    return {
        "state_scores": _state_score_table(state_labels, full_score, selection_rates),
        "transition_counts": _matrix_to_nested_dict(state_labels, full_score["transition_counts"]),
        "config": asdict(config),
        "n_trials": len(encoded_sequences),
        "n_states": len(state_labels),
    }


def _validate_config(config: LandmarkMiningConfig) -> None:
    """校验 landmark miner 配置是否处于支持范围。"""

    if not config.features:
        raise ValueError("features must not be empty")
    allowed = {"visit", "coverage", "path_commonality", "betweenness", "boundary"}
    unknown = [feature for feature in config.features if feature not in allowed]
    if unknown:
        raise ValueError(f"unknown landmark features: {unknown}; allowed={sorted(allowed)}")
    if config.max_landmarks <= 0:
        raise ValueError("max_landmarks must be positive")
    if not 1 <= config.min_landmarks <= config.max_landmarks:
        raise ValueError("min_landmarks must be in [1, max_landmarks]")
    if config.bootstrap_selection_mode not in {"fixed", "elbow"}:
        raise ValueError("bootstrap_selection_mode must be 'fixed' or 'elbow'")
    if config.bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be non-negative")
    if not 0 < config.bootstrap_sample_ratio <= 1:
        raise ValueError("bootstrap_sample_ratio must be in (0, 1]")
    if config.elbow_min_relative_drop < 0:
        raise ValueError("elbow_min_relative_drop must be non-negative")
    if not 0 <= config.elbow_min_score <= 1:
        raise ValueError("elbow_min_score must be in [0, 1]")
    if not 0 <= config.selection_threshold <= 1:
        raise ValueError("selection_threshold must be in [0, 1]")
    if config.n_jobs < 0:
        raise ValueError("n_jobs must be non-negative; use 0 for all CPUs")
    if config.eps <= 0:
        raise ValueError("eps must be positive")
    if config.min_edge_cost <= 0:
        raise ValueError("min_edge_cost must be positive")


def _encode_state_sequences(state_sequences: Sequence[Sequence[Any]]) -> tuple[List[np.ndarray], List[Any]]:
    """把任意可比较状态标签编码为紧凑整数索引序列。"""

    clean_sequences = [list(seq) for seq in state_sequences if len(seq) > 0]
    states = sorted({state for seq in clean_sequences for state in seq})
    state_to_idx = {state: idx for idx, state in enumerate(states)}
    encoded = [np.asarray([state_to_idx[state] for state in seq], dtype=np.int64) for seq in clean_sequences]
    return encoded, states


def _score_encoded_sequences(
    encoded_sequences: Sequence[np.ndarray],
    n_states: int,
    config: LandmarkMiningConfig,
) -> Dict[str, np.ndarray | Dict[str, np.ndarray]]:
    """在已编码状态序列上计算候选状态特征、rank 和综合分数。"""

    transition_counts = _transition_counts(encoded_sequences, n_states)
    visit_counts = _visit_counts(encoded_sequences, n_states)
    coverage_counts = _coverage_counts(encoded_sequences, n_states)
    path_commonality_counts = _path_commonality_counts(
        encoded_sequences,
        n_states,
        exclude_trial_endpoints=config.exclude_trial_endpoints_for_commonality,
    )

    n_trials = max(len(encoded_sequences), 1)
    n_pairs = max(_distinct_start_goal_pair_count(encoded_sequences), 1)
    features = {
        "visit": visit_counts / max(float(visit_counts.sum()), 1.0),
        "coverage": coverage_counts / float(n_trials),
        "path_commonality": path_commonality_counts / float(n_pairs),
        "betweenness": _weighted_betweenness_from_counts(
            transition_counts,
            eps=config.eps,
            min_edge_cost=config.min_edge_cost,
        ),
        "boundary": _boundary_score(transition_counts),
    }
    feature_ranks = {name: _percentile_rank(values) for name, values in features.items()}
    score = np.mean(np.vstack([feature_ranks[name] for name in config.features]), axis=0)

    return {
        "score": score,
        "features": features,
        "feature_ranks": feature_ranks,
        "transition_counts": transition_counts,
    }


def _transition_counts(encoded_sequences: Sequence[np.ndarray], n_states: int) -> np.ndarray:
    """累计经验转移计数矩阵 ``count(s -> s')``。"""

    counts = np.zeros((n_states, n_states), dtype=np.int64)
    for seq in encoded_sequences:
        if len(seq) < 2:
            continue
        np.add.at(counts, (seq[:-1], seq[1:]), 1)
    return counts


def _visit_counts(encoded_sequences: Sequence[np.ndarray], n_states: int) -> np.ndarray:
    """统计每个状态在所有 trial 中出现的总次数。"""

    if not encoded_sequences:
        return np.zeros(n_states, dtype=np.int64)
    concatenated = np.concatenate(encoded_sequences)
    return np.bincount(concatenated, minlength=n_states).astype(np.int64)


def _coverage_counts(encoded_sequences: Sequence[np.ndarray], n_states: int) -> np.ndarray:
    """统计每个状态至少出现过的 trial 数。"""

    counts = np.zeros(n_states, dtype=np.int64)
    for seq in encoded_sequences:
        if len(seq) == 0:
            continue
        counts[np.unique(seq)] += 1
    return counts


def _path_commonality_counts(
    encoded_sequences: Sequence[np.ndarray],
    n_states: int,
    exclude_trial_endpoints: bool,
) -> np.ndarray:
    """统计每个状态覆盖了多少个不同 start-goal pair。"""

    pair_to_states: Dict[Tuple[int, int], set[int]] = {}
    for seq in encoded_sequences:
        if len(seq) == 0:
            continue
        key = (int(seq[0]), int(seq[-1]))
        if exclude_trial_endpoints and len(seq) > 2:
            states = set(int(s) for s in seq[1:-1])
        elif exclude_trial_endpoints:
            states = set()
        else:
            states = set(int(s) for s in seq)
        pair_to_states.setdefault(key, set()).update(states)

    counts = np.zeros(n_states, dtype=np.int64)
    for states in pair_to_states.values():
        if states:
            counts[list(states)] += 1
    return counts


def _distinct_start_goal_pair_count(encoded_sequences: Sequence[np.ndarray]) -> int:
    """计算非空 trial 中不同 start-goal pair 的数量。"""

    return len({(int(seq[0]), int(seq[-1])) for seq in encoded_sequences if len(seq) > 0})


def _weighted_betweenness_from_counts(
    transition_counts: np.ndarray,
    eps: float,
    min_edge_cost: float,
) -> np.ndarray:
    """在小规模有向加权图上计算 Brandes betweenness。"""

    n_states = transition_counts.shape[0]
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        transition_counts,
        row_sums,
        out=np.zeros_like(transition_counts, dtype=np.float64),
        where=row_sums > 0,
    )
    weights = np.full_like(probabilities, np.inf, dtype=np.float64)
    mask = probabilities > 0
    weights[mask] = np.maximum(-np.log(np.maximum(probabilities[mask], eps)), min_edge_cost)
    adjacency = [np.flatnonzero(mask[source]).tolist() for source in range(n_states)]

    betweenness = np.zeros(n_states, dtype=np.float64)
    for source in range(n_states):
        stack: List[int] = []
        predecessors: List[List[int]] = [[] for _ in range(n_states)]
        sigma = np.zeros(n_states, dtype=np.float64)
        sigma[source] = 1.0
        distance = np.full(n_states, np.inf, dtype=np.float64)
        distance[source] = 0.0
        queue: List[Tuple[float, int]] = [(0.0, source)]

        while queue:
            dist_v, vertex = heapq.heappop(queue)
            if dist_v > distance[vertex] + eps:
                continue
            stack.append(vertex)
            for neighbor in adjacency[vertex]:
                candidate = distance[vertex] + weights[vertex, neighbor]
                if candidate < distance[neighbor] - eps:
                    distance[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
                    sigma[neighbor] = sigma[vertex]
                    predecessors[neighbor] = [vertex]
                elif abs(candidate - distance[neighbor]) <= eps:
                    sigma[neighbor] += sigma[vertex]
                    predecessors[neighbor].append(vertex)

        dependency = np.zeros(n_states, dtype=np.float64)
        while stack:
            vertex = stack.pop()
            if sigma[vertex] > 0:
                coeff = (1.0 + dependency[vertex]) / sigma[vertex]
                for predecessor in predecessors[vertex]:
                    dependency[predecessor] += sigma[predecessor] * coeff
            if vertex != source:
                betweenness[vertex] += dependency[vertex]

    if n_states > 2:
        betweenness /= float((n_states - 1) * (n_states - 2))
    return betweenness


def _boundary_score(transition_counts: np.ndarray) -> np.ndarray:
    """计算入/出转移熵与入/出度共同构成的边界性分数。"""

    incoming = transition_counts.sum(axis=0)
    outgoing = transition_counts.sum(axis=1)
    in_entropy = _column_entropy(transition_counts, incoming)
    out_entropy = _row_entropy(transition_counts, outgoing)
    in_degree = (transition_counts > 0).sum(axis=0)
    out_degree = (transition_counts > 0).sum(axis=1)
    degree_scale = max(float(transition_counts.shape[0] - 1), 1.0)
    return in_entropy + out_entropy + (in_degree + out_degree) / degree_scale


def _row_entropy(counts: np.ndarray, row_sums: np.ndarray) -> np.ndarray:
    """计算每个 source row 的出边熵。"""

    probs = np.divide(
        counts,
        row_sums[:, None],
        out=np.zeros_like(counts, dtype=np.float64),
        where=row_sums[:, None] > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(probs > 0, -probs * np.log(probs), 0.0)
    return entropy_terms.sum(axis=1)


def _column_entropy(counts: np.ndarray, col_sums: np.ndarray) -> np.ndarray:
    """计算每个 target column 的入边熵。"""

    probs = np.divide(
        counts,
        col_sums[None, :],
        out=np.zeros_like(counts, dtype=np.float64),
        where=col_sums[None, :] > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_terms = np.where(probs > 0, -probs * np.log(probs), 0.0)
    return entropy_terms.sum(axis=0)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    """将原始特征值转换为百分位 rank，平分并列名次。"""

    values = np.asarray(values, dtype=np.float64)
    n_values = len(values)
    if n_values == 0:
        return values.copy()
    if n_values == 1:
        return np.ones(1, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(n_values, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < n_values:
        end = start + 1
        while end < n_values and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank / (n_values - 1)
        start = end
    return ranks


def _top_k_indices(values: np.ndarray, k: int) -> List[int]:
    """按分数降序、索引升序返回前 k 个状态索引。"""

    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def _select_landmark_indices(values: np.ndarray, config: LandmarkMiningConfig) -> List[int]:
    """根据 fixed 或 elbow 模式选择本次 bootstrap 的 landmark 索引。"""

    if config.bootstrap_selection_mode == "fixed":
        return _top_k_indices(values, config.max_landmarks)
    return _elbow_indices(values, config)


def _elbow_indices(values: np.ndarray, config: LandmarkMiningConfig) -> List[int]:
    """用相邻分数下降的 elbow 规则决定候选 landmark 数量。"""

    ordered = _top_k_indices(values, len(values))
    if not ordered:
        return []

    candidate_count = min(config.max_landmarks, len(ordered))
    top_ordered = ordered[:candidate_count]
    sorted_scores = np.asarray([values[idx] for idx in ordered], dtype=np.float64)
    if candidate_count <= config.min_landmarks or len(sorted_scores) < 2:
        return top_ordered[:candidate_count]

    score_span = max(float(sorted_scores[0] - sorted_scores[-1]), config.eps)
    gap_limit = min(candidate_count, len(sorted_scores) - 1)
    gaps = sorted_scores[:gap_limit] - sorted_scores[1 : gap_limit + 1]
    relative_gaps = gaps / score_span

    best_gap_index = int(np.argmax(relative_gaps))
    best_k = best_gap_index + 1
    if best_k < config.min_landmarks or relative_gaps[best_gap_index] < config.elbow_min_relative_drop:
        score_k = candidate_count
        if config.elbow_min_score > 0:
            score_k = sum(float(score) >= config.elbow_min_score for score in sorted_scores[:candidate_count])
        best_k = max(config.min_landmarks, min(score_k, candidate_count))

    return top_ordered[:best_k]


def _bootstrap_selection_counts(
    encoded_sequences: Sequence[np.ndarray],
    n_states: int,
    config: LandmarkMiningConfig,
) -> np.ndarray:
    """通过 trial subsampling 统计每个状态被选为 landmark 的次数。"""

    if config.bootstrap_iterations == 0:
        return np.zeros(n_states, dtype=np.int64)

    n_trials = len(encoded_sequences)
    sample_size = max(1, int(math.ceil(n_trials * config.bootstrap_sample_ratio)))
    rng = np.random.default_rng(config.random_state)
    sample_indices = np.vstack(
        [rng.choice(n_trials, size=sample_size, replace=False) for _ in range(config.bootstrap_iterations)]
    )

    n_jobs = _resolve_n_jobs(config.n_jobs)
    if n_jobs == 1 or config.bootstrap_iterations < 2:
        return _bootstrap_worker(
            encoded_sequences,
            n_states,
            config,
            sample_indices,
        )

    chunks = [chunk for chunk in np.array_split(sample_indices, min(n_jobs, config.bootstrap_iterations)) if len(chunk)]
    counts = np.zeros(n_states, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(chunks))) as executor:
        futures = [executor.submit(_bootstrap_worker, encoded_sequences, n_states, config, chunk) for chunk in chunks]
        for future in futures:
            counts += future.result()
    return counts


def _bootstrap_worker(
    encoded_sequences: Sequence[np.ndarray],
    n_states: int,
    config: LandmarkMiningConfig,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """执行一批 bootstrap 样本的打分和 landmark 选择。"""

    counts = np.zeros(n_states, dtype=np.int64)
    for sample in sample_indices:
        sampled_sequences = [encoded_sequences[int(idx)] for idx in sample]
        score = _score_encoded_sequences(sampled_sequences, n_states, config)["score"]
        selected = _select_landmark_indices(score, config)
        counts[selected] += 1
    return counts


def _resolve_n_jobs(n_jobs: int) -> int:
    """把配置中的 worker 数转换为实际进程数。"""

    if n_jobs == 0:
        return max(os.cpu_count() or 1, 1)
    return max(n_jobs, 1)


def _state_score_table(
    state_labels: Sequence[Any],
    score_result: Mapping[str, Any],
    selection_rates: np.ndarray,
) -> Dict[Any, Dict[str, float]]:
    """把数组形式的分数和特征转换成按原始状态标签索引的表格。"""

    features: Mapping[str, np.ndarray] = score_result["features"]
    feature_ranks: Mapping[str, np.ndarray] = score_result["feature_ranks"]
    score: np.ndarray = score_result["score"]
    table: Dict[Any, Dict[str, float]] = {}
    for idx, state in enumerate(state_labels):
        row: Dict[str, float] = {
            "score": float(score[idx]),
            "selection_rate": float(selection_rates[idx]),
        }
        for name, values in features.items():
            row[name] = float(values[idx])
            row[f"{name}_rank"] = float(feature_ranks[name][idx])
        table[state] = row
    return table


def _matrix_to_nested_dict(state_labels: Sequence[Any], matrix: np.ndarray) -> Dict[Any, Dict[Any, int]]:
    """把转移计数矩阵转换为稀疏嵌套 dict，便于保存和阅读。"""

    nested: Dict[Any, Dict[Any, int]] = {}
    for i, source in enumerate(state_labels):
        row = {}
        for j, target in enumerate(state_labels):
            value = int(matrix[i, j])
            if value:
                row[target] = value
        nested[source] = row
    return nested


def state_sequences_from_transition_rows(
    rows: Iterable[Mapping[str, Any]],
    trial_key: str = "trial_id",
    step_key: str = "step",
    state_key: str = "state",
    next_state_key: str = "next_state",
    valid_key: str = "valid",
) -> List[List[Any]]:
    """从行式 transition 记录恢复每个 trial 的状态序列。

    这个 helper 只依赖 mapping 协议，脚本中可直接传 DataFrame.to_dict("records")。
    """

    grouped: Dict[Any, List[Mapping[str, Any]]] = {}
    for row in rows:
        if not bool(row[valid_key]):
            continue
        grouped.setdefault(row[trial_key], []).append(row)

    sequences: List[List[Any]] = []
    for _, trial_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        ordered = sorted(trial_rows, key=lambda row: row[step_key])
        if not ordered:
            continue
        states = [row[state_key] for row in ordered]
        states.append(ordered[-1][next_state_key])
        sequences.append(states)
    return sequences
