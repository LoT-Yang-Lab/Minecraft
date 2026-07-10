# -*- coding: utf-8 -*-
"""
可视化基于生成模型的 landmark 恢复验证结果。

输入由 ``script/simulated_recovery_pipeline/validate_landmark_recovery.py`` 生成。报告重点关注：
1. 不同生成条件下 stable recovered landmark 的 precision/recall/F1；
2. ``k`` 等于真实 landmark 数量时的 top-k recall；
3. 真实 landmark 与最强假阳性状态的 bootstrap 选择率；
4. 九宫格状态热图，用来观察哪些状态更容易被推断为 landmark。
"""

from __future__ import annotations

import argparse
import base64
import html
import io

# unused import
from datetime import datetime  # noqa: F401
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GRID_STATES = tuple(range(1, 10))


def label_landmarks(landmarks: Sequence[int]) -> str:
    """把 landmark 状态集合转换为稳定标签，便于分组和图标题显示。"""

    return ",".join(str(int(state)) for state in sorted(landmarks))


def condition_profile(condition: Dict[str, Any]) -> str:
    """把 LL/LU/UL 策略和 beta 参数压缩成人类可读的 profile 标签。"""

    ll_strategy = condition.get("ll_strategy", "legacy")
    lu_strategy = condition.get("lu_strategy", "legacy")
    ul_strategy = condition.get("ul_strategy", "legacy")
    beta_ul = float(condition.get("beta_ul", 0.0))
    beta_ll = float(condition.get("beta_ll", beta_ul))
    beta_lu = float(condition.get("beta_lu", beta_ul))
    strategy_codes = {"distance": "D", "legacy": "X"}
    ll_code = strategy_codes.get(str(ll_strategy), str(ll_strategy)[:1].upper())
    lu_code = strategy_codes.get(str(lu_strategy), str(lu_strategy)[:1].upper())
    ul_code = strategy_codes.get(str(ul_strategy), str(ul_strategy)[:1].upper())
    return f"LL{ll_code}{beta_ll:g}:LU{lu_code}{beta_lu:g}:UL{ul_code}{beta_ul:g}"


def allocation_group(condition: Dict[str, Any]) -> str:
    """返回用于分组的概率分配策略名称；混合策略时退回到完整 profile。"""

    strategies = [
        str(condition.get("ll_strategy", "legacy")),
        str(condition.get("lu_strategy", "legacy")),
        str(condition.get("ul_strategy", "legacy")),
    ]
    if len(set(strategies)) == 1:
        return strategies[0]
    return condition_profile(condition)


def load_payload(path: Path) -> Dict[str, Any]:
    """读取恢复检验结果文件，并校验必要字段存在。"""

    if not path.exists():
        raise FileNotFoundError(f"验证结果不存在: {path}")
    payload = joblib.load(path)
    if "results" not in payload:
        raise ValueError(f"文件缺少 results 字段: {path}")
    return payload


def flatten_results(results: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """把嵌套 joblib 结果展开为一行一个抽样数据集的 DataFrame。"""

    records: List[Dict[str, Any]] = []
    for result in results:
        condition = result.get("condition", {})
        stable = result.get("metrics_stable", {})
        top_true_k = result.get("metrics_top_true_k", {})
        top_max = result.get("metrics_top_max", {})
        true_landmarks = tuple(int(state) for state in result.get("true_landmarks", condition.get("landmarks", ())))
        beta_ul = float(condition.get("beta_ul", 0.0))

        records.append(
            {
                "condition_id": result.get("condition_id", condition.get("condition_id", "")),
                "landmarks_label": label_landmarks(true_landmarks),
                "landmark_count": len(true_landmarks),
                "n_trials": int(condition.get("n_trials", 0)),
                "pool_n_trials": int(condition.get("pool_n_trials", condition.get("n_trials", 0))),
                "dataset_index": int(condition.get("dataset_index", condition.get("seed", 0))),
                "pool_seed": int(condition.get("pool_seed", condition.get("seed", 0))),
                "sample_seed": int(condition.get("sample_seed", condition.get("seed", 0))),
                "mass_ll": float(condition.get("mass_ll", 0.0)),
                "ll_strategy": condition.get("ll_strategy", "legacy"),
                "lu_strategy": condition.get("lu_strategy", "legacy"),
                "ul_strategy": condition.get("ul_strategy", "legacy"),
                "beta_ll": float(condition.get("beta_ll", beta_ul)),
                "beta_lu": float(condition.get("beta_lu", beta_ul)),
                "beta_ul": beta_ul,
                "planning_alpha": float(condition.get("planning_alpha", 0.0)),
                "landmark_path_length_beta": float(condition.get("landmark_path_length_beta", 0.0)),
                "max_landmark_simple_paths": int(condition.get("max_landmark_simple_paths", 0)),
                "profile_label": condition_profile(condition),
                "allocation_group": allocation_group(condition),
                "seed": int(condition.get("seed", 0)),
                "stable_precision": float(stable.get("precision", 0.0)),
                "stable_recall": float(stable.get("recall", 0.0)),
                "stable_f1": float(stable.get("f1", 0.0)),
                "stable_jaccard": float(stable.get("jaccard", 0.0)),
                "top_true_k_precision": float(top_true_k.get("precision", 0.0)),
                "top_true_k_recall": float(top_true_k.get("recall", 0.0)),
                "top_true_k_f1": float(top_true_k.get("f1", 0.0)),
                "top_max_precision": float(top_max.get("precision", 0.0)),
                "top_max_recall": float(top_max.get("recall", 0.0)),
                "top_max_f1": float(top_max.get("f1", 0.0)),
                "mean_true_selection_rate": float(result.get("mean_true_selection_rate", 0.0)),
                "max_false_selection_rate": float(result.get("max_false_selection_rate", 0.0)),
                "stable_recovered_count": int(result.get("stable_recovered_count", 0)),
                "generated_steps_mean": float(
                    result.get("generated_summary", result.get("generated", {})).get("mean_steps", 0.0)
                ),
            }
        )

    if not records:
        raise ValueError("验证结果为空，无法可视化")
    return pd.DataFrame.from_records(records)


def figure_to_data_uri(fig: plt.Figure) -> str:
    """将 Matplotlib 图像编码成可直接嵌入 HTML 的 base64 data URI。"""

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def subplot_grid(n_items: int, max_cols: int = 3) -> Tuple[int, int]:
    """根据面板数量计算尽量紧凑的 subplot 行列数。"""

    n_cols = min(max_cols, max(1, n_items))
    n_rows = int(np.ceil(n_items / n_cols))
    return n_rows, n_cols


def line_label(mass_ll: float, profile_label: str, show_profile: bool) -> str:
    """生成折线图 legend 文本，必要时附加完整生成 profile。"""

    base = f"p(L->L)={mass_ll:g}"
    if not show_profile:
        return base
    return f"{base}; {profile_label}"


def allocation_sort_key(value: str) -> Tuple[int, str]:
    """给分配策略提供稳定排序键，保证报告图表顺序可复现。"""

    order = {"distance": 0}
    return order.get(value, 9), value


def condition_group_summary(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """按真实 landmark、pLL 和策略 profile 汇总均值、标准差与样本数。"""

    agg_spec = {}
    for metric in metrics:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
        agg_spec[f"{metric}_count"] = (metric, "count")

    summary = (
        df.groupby(["landmarks_label", "mass_ll", "allocation_group", "profile_label"], as_index=False)
        .agg(**agg_spec)
        .sort_values(
            ["landmarks_label", "mass_ll", "allocation_group"],
            key=lambda column: column.map(allocation_sort_key) if column.name == "allocation_group" else column,
        )
    )
    show_allocation = df["allocation_group"].nunique() > 1
    summary["group_label"] = summary.apply(
        lambda row: (
            f"L={row['landmarks_label']}\npLL={row['mass_ll']:g}\n{row['allocation_group']}"
            if show_allocation
            else f"L={row['landmarks_label']}\npLL={row['mass_ll']:g}"
        ),
        axis=1,
    )
    return summary


def plot_metric_by_trials(df: pd.DataFrame, metric: str, title: str, ylabel: str) -> str:
    """绘制某个恢复指标随样本量变化的图，样本量固定时改用分组柱状图。"""

    if df["n_trials"].nunique() == 1:
        summary = condition_group_summary(df, [metric])
        labels = summary["group_label"].tolist()
        values = summary[f"{metric}_mean"].to_numpy(dtype=float)
        std = summary[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        count = summary[f"{metric}_count"].to_numpy(dtype=float)
        sem = std / np.sqrt(np.maximum(count, 1.0))

        fig, ax = plt.subplots(figsize=(max(6.4, 1.35 * len(labels)), 4.2))
        x = np.arange(len(labels))
        ax.bar(x, values, color="#3b82f6", alpha=0.82)
        ax.errorbar(x, values, yerr=sem, fmt="none", ecolor="#1f2937", capsize=3, linewidth=1.0)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.04)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        return figure_to_data_uri(fig)

    beta_values = sorted(df["beta_ul"].unique())
    n_rows, n_cols = subplot_grid(len(beta_values), max_cols=3)
    profile_groups = df.groupby(["mass_ll", "profile_label"]).ngroups
    panel_height = 3.8 + max(profile_groups - 3, 0) * 0.18
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, panel_height * n_rows), squeeze=False)
    show_profile = df["profile_label"].nunique() > 1

    for ax in axes.flat[len(beta_values) :]:
        ax.axis("off")

    for ax, beta in zip(axes.flat, beta_values):
        subset = df[df["beta_ul"] == beta]
        for (mass_ll, profile_label), profile_subset in subset.groupby(["mass_ll", "profile_label"], sort=True):
            line_data = (
                profile_subset.groupby("n_trials", as_index=False)[metric].agg(["mean", "std", "count"]).reset_index()
            )
            x = line_data["n_trials"].to_numpy(dtype=float)
            y = line_data["mean"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=2.0, label=line_label(mass_ll, profile_label, show_profile))
            if len(x) > 1:
                sem = line_data["std"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(
                    np.maximum(line_data["count"].to_numpy(dtype=float), 1.0)
                )
                ax.fill_between(x, np.clip(y - sem, 0, 1), np.clip(y + sem, 0, 1), alpha=0.14)

        ax.set_title(f"beta(U->L)={beta:g}", fontsize=11)
        ax.set_xlabel("Generated trials")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.04)
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def plot_stable_prf_summary(df: pd.DataFrame) -> str:
    """绘制 stable recovered landmark 的 precision、recall 和 F1 概览。"""

    metric_specs = [
        ("stable_precision", "Precision", "#2563eb", "o"),
        ("stable_recall", "Recall", "#16a34a", "s"),
        ("stable_f1", "F1", "#dc2626", "^"),
    ]
    if df["n_trials"].nunique() == 1:
        summary = condition_group_summary(df, [metric for metric, _, _, _ in metric_specs])
        labels = summary["group_label"].tolist()
        x = np.arange(len(labels))
        width = 0.24

        fig, ax = plt.subplots(figsize=(max(7.2, 1.55 * len(labels)), 4.6))
        for offset, (metric, label, color, _) in zip([-width, 0.0, width], metric_specs):
            values = summary[f"{metric}_mean"].to_numpy(dtype=float)
            std = summary[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
            count = summary[f"{metric}_count"].to_numpy(dtype=float)
            sem = std / np.sqrt(np.maximum(count, 1.0))
            positions = x + offset
            ax.bar(positions, values, width=width, color=color, alpha=0.82, label=label)
            ax.errorbar(positions, values, yerr=sem, fmt="none", ecolor="#1f2937", capsize=3, linewidth=1.0)

        n_trials = int(df["n_trials"].iloc[0])
        ax.set_title(f"Stable recovered landmarks at {n_trials} generated trials", fontsize=13)
        ax.set_ylabel("Score")
        ax.set_ylim(-0.02, 1.04)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=10, ncols=3)
        fig.tight_layout()
        return figure_to_data_uri(fig)

    grouped = df.groupby("n_trials", as_index=False).agg(
        stable_precision_mean=("stable_precision", "mean"),
        stable_precision_std=("stable_precision", "std"),
        stable_precision_count=("stable_precision", "count"),
        stable_recall_mean=("stable_recall", "mean"),
        stable_recall_std=("stable_recall", "std"),
        stable_recall_count=("stable_recall", "count"),
        stable_f1_mean=("stable_f1", "mean"),
        stable_f1_std=("stable_f1", "std"),
        stable_f1_count=("stable_f1", "count"),
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = grouped["n_trials"].to_numpy(dtype=float)
    for metric, label, color, marker in metric_specs:
        y = grouped[f"{metric}_mean"].to_numpy(dtype=float)
        std = grouped[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        count = grouped[f"{metric}_count"].to_numpy(dtype=float)
        sem = std / np.sqrt(np.maximum(count, 1.0))
        ax.plot(x, y, marker=marker, linewidth=2.2, color=color, label=label)
        if len(x) > 1:
            ax.fill_between(x, np.clip(y - sem, 0, 1), np.clip(y + sem, 0, 1), color=color, alpha=0.12)

    ax.set_title("Stable recovered landmarks: precision, recall, and F1", fontsize=13)
    ax.set_xlabel("Generated trials")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.04)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def plot_true_false_rates(df: pd.DataFrame) -> str:
    """绘制真实 landmark 选择率和最强假阳性选择率的对比。"""

    if df["n_trials"].nunique() == 1:
        summary = condition_group_summary(df, ["mean_true_selection_rate", "max_false_selection_rate"])
        labels = summary["group_label"].tolist()
        x = np.arange(len(labels))
        width = 0.34

        fig, ax = plt.subplots(figsize=(max(7.2, 1.55 * len(labels)), 4.4))
        metric_specs = [
            ("mean_true_selection_rate", "True landmarks", "#16a34a", -width / 2),
            ("max_false_selection_rate", "Strongest false positive", "#dc2626", width / 2),
        ]
        for metric, label, color, offset in metric_specs:
            values = summary[f"{metric}_mean"].to_numpy(dtype=float)
            std = summary[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
            count = summary[f"{metric}_count"].to_numpy(dtype=float)
            sem = std / np.sqrt(np.maximum(count, 1.0))
            positions = x + offset
            ax.bar(positions, values, width=width, color=color, alpha=0.82, label=label)
            ax.errorbar(positions, values, yerr=sem, fmt="none", ecolor="#1f2937", capsize=3, linewidth=1.0)

        ax.set_title("Bootstrap selection rates by condition group", fontsize=13)
        ax.set_ylabel("Selection rate")
        ax.set_ylim(-0.02, 1.04)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        return figure_to_data_uri(fig)

    beta_values = sorted(df["beta_ul"].unique())
    n_rows, n_cols = subplot_grid(len(beta_values), max_cols=3)
    profile_groups = df.groupby(["mass_ll", "profile_label"]).ngroups
    panel_height = 3.8 + max(profile_groups - 3, 0) * 0.22
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, panel_height * n_rows), squeeze=False)
    show_profile = df["profile_label"].nunique() > 1

    for ax in axes.flat[len(beta_values) :]:
        ax.axis("off")

    for ax, beta in zip(axes.flat, beta_values):
        subset = df[df["beta_ul"] == beta]
        for (mass_ll, profile_label), profile_subset in subset.groupby(["mass_ll", "profile_label"], sort=True):
            line_data = profile_subset.groupby("n_trials", as_index=False).agg(
                true_rate=("mean_true_selection_rate", "mean"),
                false_rate=("max_false_selection_rate", "mean"),
            )
            x = line_data["n_trials"].to_numpy(dtype=float)
            label = line_label(mass_ll, profile_label, show_profile)
            ax.plot(x, line_data["true_rate"], marker="o", linewidth=2.0, label=f"true {label}")
            ax.plot(
                x,
                line_data["false_rate"],
                marker="s",
                linewidth=1.8,
                linestyle="--",
                label=f"false {label}",
            )

        ax.set_title(f"beta(U->L)={beta:g}", fontsize=11)
        ax.set_xlabel("Generated trials")
        ax.set_ylabel("Bootstrap selection rate")
        ax.set_ylim(-0.02, 1.04)
        ax.grid(True, alpha=0.22)
        ax.legend(frameon=False, fontsize=8, ncols=2)

    fig.suptitle("True landmarks vs strongest false-positive state", fontsize=14, y=1.02)
    fig.tight_layout()
    return figure_to_data_uri(fig)


def selection_rates_by_state(
    results: Sequence[Dict[str, Any]],
    landmark_label: str,
    profile_label: str,
    mass_ll: float,
) -> np.ndarray:
    """计算某个条件组下每个九宫格状态的平均 bootstrap 选择率。"""

    rows = [
        row
        for row in results
        if label_landmarks(row.get("true_landmarks", ())) == landmark_label
        and condition_profile(row.get("condition", {})) == profile_label
        and abs(float(row.get("condition", {}).get("mass_ll", 0.0)) - mass_ll) < 1e-12
    ]
    values = np.zeros(len(GRID_STATES), dtype=np.float64)
    if not rows:
        return values

    for row in rows:
        scores = {int(state): score for state, score in row.get("state_scores", {}).items()}
        for idx, state in enumerate(GRID_STATES):
            values[idx] += float(scores.get(state, {}).get("selection_rate", 0.0))
    return values / len(rows)


def plot_selection_heatmaps(results: Sequence[Dict[str, Any]], df: pd.DataFrame) -> str:
    """绘制九宫格状态选择率热图，红框标出真实 landmark。"""

    show_allocation = df["allocation_group"].nunique() > 1
    label_rows = (
        df[["landmarks_label", "mass_ll", "allocation_group", "profile_label"]]
        .drop_duplicates()
        .sort_values(
            ["landmarks_label", "mass_ll", "allocation_group"],
            key=lambda column: column.map(allocation_sort_key) if column.name == "allocation_group" else column,
        )
        .to_dict("records")
    )
    n_rows, n_cols = subplot_grid(len(label_rows), max_cols=3)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 4.0 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax in axes.flat[len(label_rows) :]:
        ax.axis("off")

    image = None
    for ax, label_row in zip(axes.flat, label_rows):
        label = label_row["landmarks_label"]
        profile_label = label_row["profile_label"]
        mass_ll = float(label_row["mass_ll"])
        allocation = label_row["allocation_group"]
        true_states = {int(state) for state in label.split(",") if state}
        values = selection_rates_by_state(results, label, profile_label, mass_ll).reshape(3, 3)
        image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
        title = f"True landmarks: {label}\npLL={mass_ll:g}"
        if show_allocation:
            title = f"{title}, {allocation}"
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        for row_idx in range(3):
            for col_idx in range(3):
                state = row_idx * 3 + col_idx + 1
                value = values[row_idx, col_idx]
                text_color = "white" if value >= 0.55 else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{state}\n{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )
                if state in true_states:
                    rect = plt.Rectangle(
                        (col_idx - 0.48, row_idx - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#e63946",
                        linewidth=2.2,
                    )
                    ax.add_patch(rect)

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Mean selection rate")
    fig.suptitle("State-level landmark selection heatmaps", fontsize=14, y=1.02)
    return figure_to_data_uri(fig)


def build_summary_table(df: pd.DataFrame, max_rows: int) -> str:
    """生成 HTML 报告底部的分条件指标汇总表。"""

    group_columns = [
        "landmarks_label",
        "n_trials",
        "mass_ll",
        "beta_ll",
        "beta_lu",
        "beta_ul",
        "planning_alpha",
        "landmark_path_length_beta",
    ]
    if df["profile_label"].nunique() > 1:
        group_columns.insert(1, "profile_label")

    grouped = (
        df.groupby(group_columns, as_index=False)
        .agg(
            stable_precision=("stable_precision", "mean"),
            stable_f1=("stable_f1", "mean"),
            stable_recall=("stable_recall", "mean"),
            top_true_k_recall=("top_true_k_recall", "mean"),
            mean_true_selection_rate=("mean_true_selection_rate", "mean"),
            max_false_selection_rate=("max_false_selection_rate", "mean"),
            stable_recovered_count=("stable_recovered_count", "mean"),
            n_datasets=("dataset_index", "count"),
        )
        .sort_values(group_columns)
    )
    grouped = grouped.head(max_rows).copy()
    for column in [
        "stable_precision",
        "stable_f1",
        "stable_recall",
        "top_true_k_recall",
        "mean_true_selection_rate",
        "max_false_selection_rate",
        "stable_recovered_count",
    ]:
        grouped[column] = grouped[column].map(lambda value: f"{value:.3f}")
    return grouped.to_html(index=False, classes="summary-table", border=0)


def build_html(payload: Dict[str, Any], df: pd.DataFrame, image_uris: Dict[str, str], max_table_rows: int) -> str:
    """组装包含指标卡片、图像和表格的完整 HTML 报告。"""

    results = payload["results"]
    generated_at = html.escape(str(payload.get("generated_at", "")))
    args = payload.get("args", {})
    summary_items = [
        ("数据集结果数", len(df)),
        (
            "基础条件数",
            df[["landmarks_label", "mass_ll", "allocation_group", "profile_label"]].drop_duplicates().shape[0],
        ),
        ("真实表征数", df["landmarks_label"].nunique()),
        ("抽样 trial 数", f"{int(df['n_trials'].min())} - {int(df['n_trials'].max())}"),
        ("生成池 trial 数", f"{int(df['pool_n_trials'].min())} - {int(df['pool_n_trials'].max())}"),
        ("planning alpha", f"{df['planning_alpha'].min():g} - {df['planning_alpha'].max():g}"),
        (
            "LL path beta",
            f"{df['landmark_path_length_beta'].min():g} - {df['landmark_path_length_beta'].max():g}",
        ),
        ("平均 stable F1", f"{df['stable_f1'].mean():.3f}"),
        ("平均 top-k recall", f"{df['top_true_k_recall'].mean():.3f}"),
        ("平均真 landmark 选择率", f"{df['mean_true_selection_rate'].mean():.3f}"),
        ("平均最高假阳性选择率", f"{df['max_false_selection_rate'].mean():.3f}"),
    ]
    if df["profile_label"].nunique() > 1:
        summary_items.insert(2, ("生成 profile 数", df["profile_label"].nunique()))
    cards = "\n".join(
        f"<div class='card'><div class='card-label'>{html.escape(label)}</div>"
        f"<div class='card-value'>{html.escape(str(value))}</div></div>"
        for label, value in summary_items
    )
    summary_table = build_summary_table(df, max_rows=max_table_rows)
    config_json = html.escape(str(args))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Landmark Recovery Validation</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f7f8fa;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 22px 44px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 19px;
    }}
    p {{
      line-height: 1.6;
      color: #52606d;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin: 20px 0 26px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .card-label {{
      font-size: 12px;
      color: #64748b;
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 22px;
      font-weight: 700;
      color: #102a43;
    }}
    .figure {{
      background: #fff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0 22px;
    }}
    .figure img {{
      width: 100%;
      display: block;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: #fff;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 12px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      white-space: nowrap;
    }}
    th, td {{
      border-bottom: 1px solid #e5eaf0;
      padding: 8px 10px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: #334e68;
      background: #f0f4f8;
      font-weight: 650;
    }}
    code {{
      background: #e5eaf0;
      border-radius: 4px;
      padding: 2px 5px;
    }}
    .note {{
      font-size: 13px;
      color: #627d98;
    }}
  </style>
</head>
<body>
<main>
  <h1>Landmark Recovery Validation</h1>
  <p>
    这个报告使用 <code>generative_model</code> 生成模型先指定真实 landmark 表征，再生成轨迹，
    最后用数据驱动 landmark miner 从状态序列中恢复 landmark。红框表示真实 landmark，热图数值为 bootstrap 选择率。
  </p>
  <p class="note">结果文件生成时间: {generated_at}；本报告包含 {len(results)} 条抽样数据集结果。运行参数: {config_json}</p>
  <section class="cards">{cards}</section>

  <h2>恢复质量随样本量变化</h2>
  <div class="figure"><img alt="stable precision recall f1 by trials" src="{image_uris['stable_prf']}" /></div>

  <h2>真假 landmark 选择频率</h2>
  <div class="figure"><img alt="true false selection rates" src="{image_uris['true_false_rates']}" /></div>

  <h2>状态级选择热图</h2>
  <div class="figure"><img alt="selection heatmaps" src="{image_uris['selection_heatmaps']}" /></div>

  <h2>分条件汇总</h2>
  <div class="table-wrap">{summary_table}</div>
</main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """解析可视化脚本命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/simulated_recovery_pipeline/recovery_results/landmark_recovery.joblib"),
        help="validate_landmark_recovery.py 生成的 joblib 文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figs/simulated_recovery_pipeline/landmark_recovery/recovery_report.html"),
        help="输出 HTML 报告路径",
    )
    parser.add_argument("--max-table-rows", type=int, default=240, help="HTML 汇总表最多显示多少行")
    return parser.parse_args()


def main() -> None:
    """脚本入口：读取恢复结果、生成图像并写出 HTML 报告。"""

    args = parse_args()
    payload = load_payload(args.input)
    results = payload["results"]
    df = flatten_results(results)

    image_uris = {
        "stable_prf": plot_stable_prf_summary(df),
        "true_false_rates": plot_true_false_rates(df),
        "selection_heatmaps": plot_selection_heatmaps(results, df),
    }

    report = build_html(payload, df, image_uris, max_table_rows=args.max_table_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Saved landmark recovery report to: {args.output}")


if __name__ == "__main__":
    main()
