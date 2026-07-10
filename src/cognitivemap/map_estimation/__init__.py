# -*- coding: utf-8 -*-
"""
map_estimation 模块
===================
基于多 trial 序列数据估计 cognitive map（认知地图）。

核心流程：
    1. 选择合适的距离方法计算状态间距离矩阵
    2. 使用 MDS 将距离矩阵降维到 2D 空间
    3. 保存/加载 cognitive map 结果，或渲染 HTML 可视化

提供的距离方法：
    - compute_sr_distance: 继承表示（Successor Representation）距离
    - compute_lops_distance: 基于转移概率对数的最短路径距离
    - compute_action_js_distance: 基于动作分布的 Jensen-Shannon 距离
    - compute_transition_similarity_distance: 基于转移分布的 Jensen-Shannon 距离
"""

from cognitivemap.map_estimation.distances import (
    Trial,
    compute_action_js_distance,
    compute_lops_distance,
    compute_sr_distance,
    compute_transition_similarity_distance,
)
from cognitivemap.map_estimation.embedding import CognitiveMapResult, build_cognitive_map, mds_embed
from cognitivemap.map_estimation.enrichment import enrich_from_trials
from cognitivemap.map_estimation.io import load_cognitive_map, save_cognitive_map
from cognitivemap.map_estimation.simulation import generate_simulated_trials
from cognitivemap.map_estimation.visualization import render_cognitive_map_html

__all__ = [
    # 数据类型
    "Trial",
    "CognitiveMapResult",
    # 距离方法
    "compute_sr_distance",
    "compute_lops_distance",
    "compute_action_js_distance",
    "compute_transition_similarity_distance",
    # 降维与构建
    "mds_embed",
    "build_cognitive_map",
    # 数据富化与模拟数据生成
    "enrich_from_trials",
    # 模拟数据生成
    "generate_simulated_trials",
    # 序列化
    "save_cognitive_map",
    "load_cognitive_map",
    # 可视化
    "render_cognitive_map_html",
]
