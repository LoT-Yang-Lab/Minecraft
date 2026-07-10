# -*- coding: utf-8 -*-
"""
Cognitive Map 数据富化模块
===========================
本模块提供 enrich_from_trials() 函数，从原始 trial 数据中重新计算可视化所需的中间统计量，
并返回一个新的 CognitiveMapResult 对象。

设计原则：
    - 不修改距离计算函数的签名（保持纯函数）
    - 按需调用：只有需要可视化时才进行富化
    - 非原地修改：返回新对象，不修改原始 CognitiveMapResult
"""

from dataclasses import replace
from typing import List

from cognitivemap.map_estimation.distances import (
    Trial,
    build_action_counts,
    build_edge_action_counts,
    build_transition_counts,
)
from cognitivemap.map_estimation.embedding import CognitiveMapResult


def enrich_from_trials(
    result: CognitiveMapResult,
    trials: List[Trial],
) -> CognitiveMapResult:
    """从原始 trial 数据中提取中间统计量，返回一个新的 CognitiveMapResult。

    新对象包含原始 result 的所有字段，并附加：
        - transition_counts: 状态间转移计数矩阵 (n_states, n_states)
        - action_counts: 每状态的动作计数矩阵 (n_states, n_actions)
        - action_labels: 动作标签列表
        - edge_actions: 每条有向边上的动作使用次数（仅 LoPS 方法设置）

    对非 LoPS 方法，不设置 edge_actions 字段。

    Args:
        result: 已构建的 CognitiveMapResult（含距离矩阵和 MDS 坐标）。
        trials: 原始 trial 列表。

    Returns:
        一个新的 CognitiveMapResult 实例，附加了中间统计数据。
    """
    states = result.state_labels

    transition_counts = build_transition_counts(trials, states)
    action_counts, action_labels = build_action_counts(trials, states)

    edge_actions = None
    if result.method == "lops":
        edge_actions = build_edge_action_counts(trials, states)

    return replace(
        result,
        transition_counts=transition_counts,
        action_counts=action_counts,
        action_labels=action_labels,
        edge_actions=edge_actions,
    )
