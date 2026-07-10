"""
Cognitive Map Visualization Module.
Renders cognitive maps as interactive D3.js visualizations with reveal-on-demand interaction.
"""

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

# ============================================================================
# 预设 action 颜色调色板（10 色，循环分配）
# ============================================================================


class _CognitiveMapLike(Protocol):
    """HTML 渲染器所需的认知地图结果最小协议。"""

    distance_matrix: np.ndarray
    state_labels: List[Any]
    mds_coordinates: np.ndarray
    stress: float
    method: str
    action_counts: Optional[np.ndarray]
    edge_actions: Optional[Dict[Tuple[Any, Any], Dict[str, int]]]
    action_labels: Optional[List[str]]


_ACTION_PALETTE = [
    "#22c55e",  # green
    "#06b6d4",  # cyan
    "#8b5cf6",  # violet
    "#f59e0b",  # amber
    "#3b82f6",  # blue
    "#ef4444",  # red
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#6366f1",  # indigo
]


def _auto_action_colors(action_labels: List[str]) -> Dict[str, str]:
    """为 action 标签自动分配颜色。"""
    return {label: _ACTION_PALETTE[i % len(_ACTION_PALETTE)] for i, label in enumerate(sorted(action_labels))}


# ============================================================================
# 距离 tier 计算
# ============================================================================


def _distance_tier(dist: float, max_dist: float) -> str:
    """三等分距离范围，返回 'near' / 'mid' / 'far'。"""
    if max_dist <= 0.0:
        return "near"
    if dist < max_dist * 0.33:
        return "near"
    elif dist < max_dist * 0.66:
        return "mid"
    else:
        return "far"


# ============================================================================
# JS 数据构建
# ============================================================================


def _build_js_data(
    result: _CognitiveMapLike,
    threshold: float = 0.2,
    top_k: int = 2,
    token_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """将 CognitiveMapResult 转换为 JS 可用的 dict。

    Args:
        result: CognitiveMapResult 实例（需先调用 enrich_from_trials）。
        threshold: 主 action 比例阈值，默认 0.2。
        top_k: 每条边最多显示几个 action，默认 2。
        token_labels: 需要显示在过滤器中的控制单元标签；默认使用实际出现过的 action/chunk。

    Returns:
        可直接 json.dumps 的 dict，包含 nodes, edges, actions, action_colors, config。
    """
    states = result.state_labels
    n_states = len(states)
    coords = result.mds_coordinates
    state_to_idx = {s: i for i, s in enumerate(states)}

    # 归一化 MDS 坐标到 [0, 1]
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    norm_coords = np.zeros_like(coords)
    norm_coords[:, 0] = (coords[:, 0] - x_min) / x_range
    norm_coords[:, 1] = (coords[:, 1] - y_min) / y_range

    # 节点
    nodes = []
    for i, label in enumerate(states):
        nodes.append(
            {
                "id": str(label),
                "x": round(float(norm_coords[i, 0]), 6),
                "y": round(float(norm_coords[i, 1]), 6),
                "label": str(label),
            }
        )

    # 边（从 edge_actions 构建）
    edge_actions = getattr(result, "edge_actions", None) or {}
    dist_matrix = result.distance_matrix
    all_distances = []
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                all_distances.append(dist_matrix[i, j])
    max_dist = float(max(all_distances)) if all_distances else 1.0

    edges = []
    for (from_label, to_label), action_counts in edge_actions.items():
        total = sum(action_counts.values())
        if total == 0:
            continue
        # 按次数降序排列 action
        sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
        # 计算比例
        action_probs = {a: cnt / total for a, cnt in sorted_actions}
        primary_action, primary_count = sorted_actions[0]
        primary_proportion = primary_count / total

        # 查找距离
        from_idx = state_to_idx.get(from_label)
        to_idx = state_to_idx.get(to_label)
        if from_idx is not None and to_idx is not None:
            dist = float(dist_matrix[from_idx, to_idx])
        else:
            dist = 0.0

        edges.append(
            {
                "source": str(from_label),
                "target": str(to_label),
                "actions": {a: round(p, 4) for a, p in action_probs.items()},
                "action_counts": {a: int(cnt) for a, cnt in sorted_actions},
                "transition_count": int(total),
                "primary_proportion": round(primary_proportion, 4),
                "distance": round(dist, 4),
                "tier": _distance_tier(dist, max_dist),
            }
        )

    # action/chunk 元数据：统计 parse 后控制单元的全局使用频数和比例。
    action_labels = getattr(result, "action_labels", None) or []
    display_labels = sorted(set(token_labels or action_labels) | set(action_labels))
    action_counts_matrix = getattr(result, "action_counts", None)
    count_by_action = {label: 0 for label in display_labels}
    if action_counts_matrix is not None:
        for idx, label in enumerate(action_labels):
            if idx < action_counts_matrix.shape[1]:
                count_by_action[label] = int(action_counts_matrix[:, idx].sum())
    total_control_units = sum(count_by_action.values())
    action_stats = {
        label: {
            "count": count_by_action[label],
            "proportion": round(count_by_action[label] / total_control_units, 4) if total_control_units else 0.0,
            "is_chunk": "-" in label,
            "length": len(label.split("-")),
        }
        for label in display_labels
    }
    action_colors = _auto_action_colors(display_labels)

    # 配置
    config = {
        "threshold": threshold,
        "top_k": top_k,
        "method": result.method,
        "stress": round(float(result.stress), 4),
        "n_states": n_states,
        "total_control_units": int(total_control_units),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "actions": display_labels,
        "action_stats": action_stats,
        "action_colors": action_colors,
        "config": config,
    }


# ============================================================================
# HTML 渲染
# ============================================================================


def _load_template() -> str:
    """加载 HTML 模板文件。"""
    template_path = Path(__file__).resolve().parent / "templates" / "cognitive_map_template.html"
    return template_path.read_text(encoding="utf-8")


def render_cognitive_map_html(
    result: _CognitiveMapLike,
    output_path: Union[str, Path],
    title: str = "Cognitive Map",
    threshold: float = 0.2,
    top_k: int = 2,
    token_labels: Optional[List[str]] = None,
    verbose: bool = False,
) -> str:
    """将 CognitiveMapResult 渲染为交互式 D3.js HTML 可视化页面。

    采用 reveal-on-demand 交互范式：
        - 默认只显示节点（无连线）
        - hover/click 节点时显示该节点的出边
        - 边按主 action 阈值和 action 类型过滤
        - 边颜色按距离 tier 映射到橙色深浅渐变

    Args:
        result: CognitiveMapResult 实例（需先调用 enrich_from_trials）。
        output_path: 输出 HTML 文件路径。
        title: 页面标题。
        threshold: 主 action 比例阈值，默认 0.2。
        top_k: 每条边最多显示几个 action 胶囊，默认 2。
        token_labels: 需要显示在过滤器中的控制单元标签；默认使用实际出现过的 action/chunk。
        verbose: 是否打印保存路径。默认 False，避免库函数产生无条件 stdout 副作用。

    Returns:
        output_path (str): 输出文件的绝对路径。
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 构建 JS 数据
    js_data = _build_js_data(result, threshold=threshold, top_k=top_k, token_labels=token_labels)
    js_data_json = json.dumps(js_data, ensure_ascii=False)

    method_display = {
        "sr": "Successor Representation",
        "lops": "LoPS-based",
        "action_js": "Action JS Divergence",
        "transition_similarity": "Transition Similarity",
    }.get(result.method, result.method)

    html_content = _load_template().format(
        title=html.escape(title),
        method_display=html.escape(method_display),
        threshold=threshold,
        js_data_json=js_data_json,
        stress=result.stress,
    )
    output.write_text(html_content, encoding="utf-8")
    if verbose:
        print(f"Cognitive Map HTML saved to: {output.resolve()}")
    return str(output.resolve())
