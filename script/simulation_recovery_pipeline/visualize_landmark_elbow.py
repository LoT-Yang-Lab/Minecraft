# -*- coding: utf-8 -*-
"""
可视化 landmark 候选分数曲线和相邻分数差，用于诊断 elbow 选点是否稳定。

输入来自 ``script/simulated_recovery_pipeline/validate_landmark_recovery.py`` 生成的 joblib。
脚本会按真实 landmark 集合和 ``pLL`` 条件分组，展示三类图形：
1. 每个抽样数据集的候选分数降序曲线；
2. 相邻 rank 的分数差曲线；
3. score-curve elbow 与 largest-gap elbow 各自建议的 landmark 数量 ``k``。
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_INPUT = Path("data/simulated_recovery_pipeline/recovery_results/landmark_recovery_quick.joblib")
DEFAULT_OUTPUT = Path("results/figs/simulated_recovery_pipeline/landmark_recovery/elbow_shapes_quick.html")
DEFAULT_IMAGE_OUTPUT = Path("results/figs/simulated_recovery_pipeline/landmark_recovery/elbow_shapes_quick.png")


def label_landmarks(landmarks: Sequence[int]) -> str:
    """把 landmark 状态集合转换为稳定的逗号分隔标签。"""

    return ",".join(str(int(state)) for state in sorted(landmarks))


def condition_key(result: Dict[str, Any]) -> Tuple[str, float]:
    """提取 elbow 图中用于分组的真实 landmark 标签和 pLL 条件。"""

    condition = result.get("condition", {})
    landmarks = result.get("true_landmarks", condition.get("landmarks", ()))
    return label_landmarks(landmarks), float(condition.get("mass_ll", 0.0))


def sorted_scores(result: Dict[str, Any]) -> np.ndarray:
    """从单个恢复结果中取出候选状态分数，并按降序排列。"""

    state_scores = result.get("state_scores", {})
    scores = [float(row.get("score", 0.0)) for row in state_scores.values()]
    return np.asarray(sorted(scores, reverse=True), dtype=np.float64)


def pad_matrix(rows: Sequence[np.ndarray]) -> np.ndarray:
    """把不同长度的一维曲线补齐成矩阵，缺失位置使用 NaN。"""

    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    width = max(len(row) for row in rows)
    matrix = np.full((len(rows), width), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        matrix[row_idx, : len(row)] = row
    return matrix


def adjacent_gaps(scores: np.ndarray) -> np.ndarray:
    """计算相邻候选分数差，即 ``score[k] - score[k+1]``。"""

    if len(scores) < 2:
        return np.zeros(0, dtype=np.float64)
    return scores[:-1] - scores[1:]


def score_elbow_k(scores: np.ndarray, max_landmarks: int) -> int:
    """根据分数曲线到端点连线的最大偏离位置估计 elbow 的 k。"""

    if len(scores) == 0:
        return 0
    candidate_count = min(max_landmarks, len(scores))
    if candidate_count <= 2:
        return candidate_count

    y = scores[:candidate_count]
    x = np.linspace(0.0, 1.0, candidate_count)
    y_span = max(float(y[0] - y[-1]), 1e-12)
    y_norm = (y - y[-1]) / y_span
    line = 1.0 - x
    distances = y_norm - line
    return int(np.argmax(distances) + 1)


def gap_elbow_k(scores: np.ndarray, max_landmarks: int) -> int:
    """使用最大相邻分数差估计 elbow 的 k。"""

    gaps = adjacent_gaps(scores)
    if len(gaps) == 0:
        return 0
    limit = min(max_landmarks, len(gaps))
    return int(np.argmax(gaps[:limit]) + 1)


def figure_to_data_uri(fig: plt.Figure, image_output: Path | None = None) -> str:
    """把 Matplotlib 图保存为 PNG，并返回可嵌入 HTML 的 data URI。"""

    buffer = io.BytesIO()
    if image_output is not None:
        image_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(image_output, format="png", dpi=180, bbox_inches="tight")
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_shapes(results: Sequence[Dict[str, Any]], max_landmarks: int) -> plt.Figure:
    """绘制分数曲线、相邻 gap 曲线和 elbow 建议 k 的分布。"""

    groups: Dict[Tuple[str, float], List[np.ndarray]] = {}
    for result in results:
        groups.setdefault(condition_key(result), []).append(sorted_scores(result))

    group_items = sorted(groups.items(), key=lambda item: (item[0][0], item[0][1]))
    n_cols = len(group_items)
    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(max(4.2 * n_cols, 8.0), 10.4),
        squeeze=False,
        constrained_layout=True,
    )

    for col_idx, ((landmark_label, mass_ll), rows) in enumerate(group_items):
        score_matrix = pad_matrix(rows)
        gap_matrix = pad_matrix([adjacent_gaps(row) for row in rows])
        x_scores = np.arange(1, score_matrix.shape[1] + 1)
        x_gaps = np.arange(1, gap_matrix.shape[1] + 1)

        ax_score = axes[0, col_idx]
        for row in score_matrix:
            ax_score.plot(x_scores, row, color="#9ca3af", alpha=0.18, linewidth=0.8)
        mean_score = np.nanmean(score_matrix, axis=0)
        q25_score = np.nanpercentile(score_matrix, 25, axis=0)
        q75_score = np.nanpercentile(score_matrix, 75, axis=0)
        ax_score.plot(x_scores, mean_score, color="#2563eb", linewidth=2.4, label="mean")
        ax_score.fill_between(x_scores, q25_score, q75_score, color="#2563eb", alpha=0.14, label="IQR")
        ax_score.axvline(max_landmarks, color="#111827", linestyle=":", linewidth=1.0)
        ax_score.set_title(f"L={landmark_label}, pLL={mass_ll:g}")
        ax_score.set_xlabel("Score rank")
        ax_score.set_ylabel("Candidate score")
        ax_score.set_ylim(-0.02, 1.04)
        ax_score.grid(True, alpha=0.22)

        ax_gap = axes[1, col_idx]
        for row in gap_matrix:
            ax_gap.plot(x_gaps, row, color="#9ca3af", alpha=0.18, linewidth=0.8)
        mean_gap = np.nanmean(gap_matrix, axis=0)
        q25_gap = np.nanpercentile(gap_matrix, 25, axis=0)
        q75_gap = np.nanpercentile(gap_matrix, 75, axis=0)
        ax_gap.plot(x_gaps, mean_gap, color="#dc2626", linewidth=2.4, label="mean")
        ax_gap.fill_between(x_gaps, q25_gap, q75_gap, color="#dc2626", alpha=0.14, label="IQR")
        ax_gap.axvline(max_landmarks, color="#111827", linestyle=":", linewidth=1.0)
        ax_gap.set_xlabel("Gap after rank k")
        ax_gap.set_ylabel("Score[k] - Score[k+1]")
        ax_gap.set_ylim(bottom=-0.01)
        ax_gap.grid(True, alpha=0.22)

        ax_hist = axes[2, col_idx]
        score_ks = [score_elbow_k(row, max_landmarks) for row in rows]
        gap_ks = [gap_elbow_k(row, max_landmarks) for row in rows]
        bins = np.arange(1, max_landmarks + 1)
        width = 0.36
        score_counts = np.asarray([score_ks.count(int(k)) for k in bins], dtype=np.float64) / max(len(rows), 1)
        gap_counts = np.asarray([gap_ks.count(int(k)) for k in bins], dtype=np.float64) / max(len(rows), 1)
        ax_hist.bar(bins - width / 2, score_counts, width=width, color="#2563eb", alpha=0.8, label="score elbow")
        ax_hist.bar(bins + width / 2, gap_counts, width=width, color="#dc2626", alpha=0.8, label="gap elbow")
        ax_hist.set_xlabel("Suggested k")
        ax_hist.set_ylabel("Dataset fraction")
        ax_hist.set_xticks(bins)
        ax_hist.set_ylim(0.0, 1.0)
        ax_hist.grid(axis="y", alpha=0.22)
        ax_hist.legend(frameon=False, fontsize=8)

    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Landmark score and gap elbow diagnostics", fontsize=15)
    return fig


def build_html(payload: Dict[str, Any], image_uri: str, input_path: Path) -> str:
    """组装包含诊断图和运行参数的独立 HTML 报告。"""

    args = html.escape(str(payload.get("args", {})))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Landmark Elbow Shapes</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f7f8fa;
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 22px 44px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    p {{ color: #4b5563; line-height: 1.55; }}
    .figure {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 14px;
      overflow-x: auto;
    }}
    img {{ max-width: 100%; height: auto; display: block; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Landmark Elbow Shapes</h1>
  <p>
    输入: <code>{html.escape(str(input_path))}</code>。
    第一行是每个 sampled dataset 的候选分数降序曲线；第二行是相邻 rank 的 score gap；
    第三行比较 score-curve elbow 和 largest-gap elbow 建议的 k 分布。
  </p>
  <p>运行参数: {args}</p>
  <div class="figure"><img alt="landmark elbow shape diagnostics" src="{image_uri}" /></div>
</main>
</body>
</html>
"""


def build_arg_parser() -> argparse.ArgumentParser:
    """构造 elbow 诊断脚本的命令行解析器。"""

    parser = argparse.ArgumentParser(description="Visualize landmark score/gap elbow shapes.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="validation joblib path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output html path")
    parser.add_argument("--image-output", type=Path, default=DEFAULT_IMAGE_OUTPUT, help="output png path")
    parser.add_argument("--max-landmarks", type=int, default=4, help="candidate k upper bound")
    return parser


def main() -> None:
    """脚本入口：读取恢复结果、生成诊断图并写出 HTML/PNG。"""

    args = build_arg_parser().parse_args()
    payload = joblib.load(args.input)
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"no results found in {args.input}")

    fig = plot_shapes(results, max_landmarks=args.max_landmarks)
    image_uri = figure_to_data_uri(fig, image_output=args.image_output)
    html_text = build_html(payload, image_uri, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"Saved elbow shape report to: {args.output}")
    print(f"Saved elbow shape image to: {args.image_output}")


if __name__ == "__main__":
    main()
