# -*- coding: utf-8 -*-
"""
Cognitive Map 结果序列化模块
=============================
提供 cognitive map 结果的保存与加载功能，使用 pickle 格式。
"""

import os
import pickle

from cognitivemap.map_estimation.embedding import CognitiveMapResult


def save_cognitive_map(
    result: CognitiveMapResult,
    filepath: str,
) -> None:
    """将 cognitive map 结果保存为 pickle 文件。

    自动创建目标目录（如果不存在）。

    Args:
        result: CognitiveMapResult 实例。
        filepath: 目标文件路径（建议使用 .pkl 后缀）。
    """
    # 确保目标目录存在
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(result, f)


def load_cognitive_map(
    filepath: str,
) -> CognitiveMapResult:
    """从 pickle 文件加载 cognitive map 结果。

    注意：此函数使用 pickle 加载数据，只应加载可信来源的文件。

    Args:
        filepath: pickle 文件路径。

    Returns:
        CognitiveMapResult 实例。

    Raises:
        FileNotFoundError: 如果文件不存在。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"cognitive map 结果文件不存在: {filepath}")

    with open(filepath, "rb") as f:
        result = pickle.load(f)

    if not isinstance(result, CognitiveMapResult):
        raise TypeError(f"加载的对象类型为 {type(result)}，期望 CognitiveMapResult")

    return result
