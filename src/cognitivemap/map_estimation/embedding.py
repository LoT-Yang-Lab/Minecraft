# -*- coding: utf-8 -*-
"""
MDS 降维与 Cognitive Map 构建模块
==================================
本模块负责将距离矩阵通过 MDS（多维缩放）降维到 2D 空间，并构建完整的 CognitiveMapResult。
"""

# unused import
from dataclasses import dataclass, field  # noqa: F401
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.manifold import MDS

# ============================================================================
# 数据类型
# ============================================================================


@dataclass
class CognitiveMapResult:
    """cognitive map 构建的完整结果。

    Attributes:
        distance_matrix: (n_states, n_states) 距离矩阵（RDM）。
        state_labels: 状态标签列表，长度为 n_states。
        mds_coordinates: (n_states, 2) MDS 降维后的 2D 坐标。
        stress: MDS 拟合的 stress 值（越小表示降维质量越好）。
        method: 使用的距离方法名称。
        transition_counts: (n_states, n_states) 状态间转移计数矩阵（可选，由 enrich_from_trials 填充）。
        action_counts: (n_states, n_actions) 每状态的动作计数矩阵（可选）。
        action_labels: 动作标签列表（可选）。
        edge_actions: 每条有向边上的动作使用次数，格式 {(from_idx, to_idx): {action: count}}（可选，仅 LoPS 方法）。
    """

    distance_matrix: np.ndarray
    state_labels: List[int]
    mds_coordinates: np.ndarray
    stress: float
    method: str
    # 可视化用的中间数据（可选）
    transition_counts: Optional[np.ndarray] = None
    action_counts: Optional[np.ndarray] = None
    action_labels: Optional[List[str]] = None
    edge_actions: Optional[Dict[Tuple[int, int], Dict[str, int]]] = None


# ============================================================================
# MDS 降维
# ============================================================================


def mds_embed(
    distance_matrix: np.ndarray, n_components: int = 2, random_state: int = 42, **kwargs
) -> Tuple[np.ndarray, float]:
    """对距离矩阵进行 MDS（多维缩放）降维。

    使用 sklearn 的 MDS 实现，基于预计算的距离矩阵进行度量 MDS。

    Args:
        distance_matrix: (n, n) 距离矩阵，必须是对称的且对角线为 0。
        n_components: 降维后的维度数，默认 2（2D 空间）。
        random_state: 随机种子，保证结果可复现。
        **kwargs: 传递给 sklearn.manifold.MDS 的其他参数。

    Returns:
        (coordinates, stress):
            coordinates: (n, n_components) 降维后的坐标。
            stress: MDS 的 stress 值。
    """
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
        **kwargs,
    )
    coordinates = mds.fit_transform(distance_matrix)
    stress = mds.stress_
    return coordinates, stress


# ============================================================================
# 构建 Cognitive Map
# ============================================================================


def build_cognitive_map(
    distance_matrix: np.ndarray,
    state_labels: List[int],
    method: str,
    n_components: int = 2,
    random_state: int = 42,
) -> CognitiveMapResult:
    """构建完整的 cognitive map 结果。

    将距离矩阵通过 MDS 降维到 2D 空间，并打包为 CognitiveMapResult。

    Args:
        distance_matrix: (n_states, n_states) 距离矩阵。
        state_labels: 状态标签列表。
        method: 距离方法名称（如 "sr", "lops", "action_js", "transition_similarity"）。
        n_components: MDS 降维维度，默认 2。
        random_state: MDS 随机种子。

    Returns:
        CognitiveMapResult 实例，包含距离矩阵、状态标签、MDS 坐标和 stress。

    Raises:
        TypeError: distance_matrix 不是 np.ndarray。
        ValueError: distance_matrix 不是二维方阵、状态数不足 2、或 state_labels 维度不匹配。
    """
    # --- 输入验证 ---
    if not isinstance(distance_matrix, np.ndarray):
        raise TypeError(f"distance_matrix 必须是 np.ndarray，实际为 {type(distance_matrix)}")
    if distance_matrix.ndim != 2:
        raise ValueError(f"distance_matrix 必须是二维方阵，实际维度为 {distance_matrix.ndim}")
    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        raise ValueError(f"distance_matrix 必须是方阵，实际形状为 {distance_matrix.shape}")
    if n < 2:
        raise ValueError(f"至少需要 2 个状态才能构建 cognitive map，实际为 {n}")
    if len(state_labels) != n:
        raise ValueError(f"state_labels 长度 ({len(state_labels)}) 与 distance_matrix 维度 ({n}) 不匹配")

    coordinates, stress = mds_embed(
        distance_matrix,
        n_components=n_components,
        random_state=random_state,
    )

    return CognitiveMapResult(
        distance_matrix=distance_matrix,
        state_labels=state_labels,
        mds_coordinates=coordinates,
        stress=stress,
        method=method,
    )
