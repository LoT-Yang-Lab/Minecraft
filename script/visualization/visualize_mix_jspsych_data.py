#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集中生成 NC mix jsPsych 数据的被试级与组级可视化。

这个脚本迁移自 ``experiments/新数据`` 的原型分析，但作为正式入口时只负责
可视化与轻量汇总：核心 chunk / landmark / map-estimation 算法仍应优先放在
``src/cognitivemap`` 和对应 pipeline 中。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "result" / "figure" / "mix_jspsych"
DEFAULT_INPUT_DIRS = (ROOT / "data" / "raw_data",)

CORNER_CW: Dict[int, int] = {1: 3, 3: 9, 9: 7, 7: 1}
NODE_POS: Dict[int, Tuple[int, int]] = {
    1: (0, 0),
    2: (0, 1),
    3: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    7: (2, 0),
    8: (2, 1),
    9: (2, 2),
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def code_to_rc(code: int) -> Tuple[int, int] | None:
    """将 1..9 节点编号转成 3x3 row/col。"""

    if not 1 <= code <= 9:
        return None
    z = code - 1
    return z // 3, z % 3


def rc_to_code(row: int, col: int) -> int:
    """将 3x3 row/col 转成节点编号。"""

    return row * 3 + col + 1


def grid_neighbor(code: int, key: str) -> int | None:
    """返回实验图上的一步后继，包含 W 角点顺时针 shortcut。"""

    key = (key or "").lower()
    if key == "w":
        return CORNER_CW.get(code)
    rc = code_to_rc(code)
    if rc is None:
        return None
    row, col = rc
    if key == "q" and col > 0:
        return rc_to_code(row, col - 1)
    if key == "e" and col < 2:
        return rc_to_code(row, col + 1)
    if key == "a" and row > 0:
        return rc_to_code(row - 1, col)
    if key == "d" and row < 2:
        return rc_to_code(row + 1, col)
    return None


def neighbors(code: int) -> List[int]:
    """返回所有合法一步后继。"""

    out: List[int] = []
    for key in ("q", "e", "a", "d", "w"):
        nxt = grid_neighbor(code, key)
        if nxt is not None:
            out.append(nxt)
    return list(dict.fromkeys(out))


def bfs_distance(start: int, goal: int) -> int:
    """有向任务图上的最短步数。"""

    if start == goal:
        return 0
    queue: deque[Tuple[int, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        cur, dist = queue.popleft()
        for nxt in neighbors(cur):
            if nxt == goal:
                return dist + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return -1


def true_layout_normalized() -> Dict[int, Tuple[float, float]]:
    """将 3x3 真值网格归一化到 [0, 1]。"""

    return {code: (col / 2.0, row / 2.0) for code, (row, col) in NODE_POS.items()}


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    """读取 jsPsych JSON trial 数组。"""

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是 jsPsych trial 数组")
    return data


def flatten_value(row: Dict[str, Any]) -> Dict[str, Any]:
    """兼容 jsPsych call-function 的外层字段与 value 内层字段。"""

    out = dict(row)
    value = row.get("value")
    if isinstance(value, dict):
        for key, val in value.items():
            if key not in out:
                out[key] = val
    return out


def participant_id_from_records(records: Sequence[Dict[str, Any]]) -> str:
    """从 trial 数组中获取被试编号。"""

    for row in records:
        pid = row.get("participant_id")
        if pid:
            return str(pid).zfill(3)
    return "unknown"


def extract_main_trials(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """提取正式主任务试次的行为指标。"""

    rows: List[Dict[str, Any]] = []
    for row in records:
        if row.get("screen_id") not in ("mix_navigation_trial", "mix_crafting_trial"):
            continue
        flat = flatten_value(row)
        value = flat.get("value") if isinstance(flat.get("value"), dict) else flat
        if not isinstance(value, dict):
            value = flat
        domain = value.get("domain") or flat.get("mix_session_domain")
        start = value.get("start_code") or flat.get("start_code")
        goal = value.get("goal_code") or flat.get("goal_code")
        steps = value.get("steps") or []
        valid_steps = [step for step in steps if step.get("valid")]
        total_steps = len(valid_steps)
        optimal = bfs_distance(int(start), int(goal)) if start and goal else None
        rows.append(
            {
                "participant_id": str(flat.get("participant_id", "unknown")).zfill(3),
                "experiment_order": flat.get("experiment_order"),
                "domain": domain,
                "mix_session": int(flat.get("mix_session") or 0),
                "trial_id": value.get("trial_id") or flat.get("trial_id"),
                "pair_id": flat.get("pair_id"),
                "category": flat.get("category"),
                "start_code": start,
                "goal_code": goal,
                "total_steps": total_steps,
                "optimal_steps": optimal,
                "path_efficiency": (optimal / total_steps) if optimal and optimal >= 0 and total_steps else np.nan,
                "excess_steps": (total_steps - optimal) if optimal is not None and optimal >= 0 else np.nan,
                "outcome": value.get("outcome"),
                "total_time_ms": value.get("total_time_ms"),
                "mean_step_rt_ms": (
                    float(np.mean([s["rt_ms"] for s in valid_steps if s.get("rt_ms") is not None]))
                    if valid_steps
                    else np.nan
                ),
                "invalid_key_count": sum(1 for s in steps if not s.get("valid")),
            }
        )
    return pd.DataFrame(rows)


def _tail_outer_inner(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """返回固定 B/D/E 试次的反应外层和刺激内层。"""

    outer = row.get("value")
    if not isinstance(outer, dict):
        return {}, {}
    inner = outer.get("value")
    return outer, inner if isinstance(inner, dict) else {}


def extract_tail_b(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """提取固定 B：节点对可达性评分。"""

    rows: List[Dict[str, Any]] = []
    for row in records:
        if row.get("tail_subphase") != "B":
            continue
        outer, inner = _tail_outer_inner(row)
        rating = outer.get("tail_response_reachability")
        a, b = inner.get("a"), inner.get("b")
        if rating is None or a is None or b is None:
            continue
        rows.append(
            {
                "participant_id": str(row.get("participant_id", "unknown")).zfill(3),
                "tail_domain": row.get("tail_domain"),
                "trial_id": row.get("tail_trial_id"),
                "node_a": int(a),
                "node_b": int(b),
                "rating": int(rating),
                "grid_distance": outer.get("tail_grid_distance") or inner.get("grid_distance"),
                "graph_distance": bfs_distance(int(a), int(b)),
                "tail_rt_ms": outer.get("tail_rt_ms"),
            }
        )
    return pd.DataFrame(rows)


def extract_tail_d(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """提取固定 D：序列整体感评分。"""

    rows: List[Dict[str, Any]] = []
    for row in records:
        if row.get("tail_subphase") != "D":
            continue
        outer, inner = _tail_outer_inner(row)
        fluency = outer.get("tail_response_fluency")
        if fluency is None:
            continue
        rows.append(
            {
                "participant_id": str(row.get("participant_id", "unknown")).zfill(3),
                "tail_domain": row.get("tail_domain"),
                "trial_id": row.get("tail_trial_id"),
                "category": inner.get("category") or row.get("tail_category"),
                "sequence": "-".join(str(x) for x in (inner.get("sequence") or [])),
                "fluency": int(fluency),
                "tail_rt_ms": outer.get("tail_rt_ms"),
            }
        )
    return pd.DataFrame(rows)


def _hub_on_shortest_path(start: int, end: int, hub: int) -> bool:
    """hub 是否落在 start-end 的任意一条最短路径上。"""

    if hub in (start, end):
        return True
    dist = bfs_distance(start, end)
    if dist < 0:
        return False
    return bfs_distance(start, hub) + bfs_distance(hub, end) == dist


def extract_tail_e(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """提取固定 E：理想中转节点。"""

    rows: List[Dict[str, Any]] = []
    for row in records:
        if row.get("tail_subphase") != "E":
            continue
        outer, inner = _tail_outer_inner(row)
        hub = outer.get("tail_response_hub")
        start, end = inner.get("start"), inner.get("end")
        chosen = inner.get("chosen") if inner.get("chosen") is not None else hub
        if start is None or end is None or chosen is None:
            continue
        rows.append(
            {
                "participant_id": str(row.get("participant_id", "unknown")).zfill(3),
                "tail_domain": row.get("tail_domain"),
                "trial_id": row.get("tail_trial_id"),
                "start": int(start),
                "end": int(end),
                "chosen_hub": int(chosen),
                "grid_distance": inner.get("grid_distance") or outer.get("tail_grid_distance"),
                "on_shortest_path": _hub_on_shortest_path(int(start), int(end), int(chosen)),
                "tail_rt_ms": outer.get("tail_rt_ms"),
            }
        )
    return pd.DataFrame(rows)


def extract_map_reconstruction(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """提取最终地图还原坐标；新版 adaptive 数据中若没有该任务则返回空表。"""

    rows: List[Dict[str, Any]] = []
    true_layout = true_layout_normalized()
    for row in records:
        if row.get("screen_id") not in ("map_reconstruction_navigation", "map_reconstruction_crafting"):
            continue
        flat = flatten_value(row)
        value = flat.get("value") if isinstance(flat.get("value"), dict) else flat
        if not isinstance(value, dict):
            continue
        nodes = (value.get("map_reconstruction") or {}).get("nodes") or []
        for node in nodes:
            code = int(node["code"])
            true_x, true_y = true_layout[code]
            rows.append(
                {
                    "participant_id": str(flat.get("participant_id", "unknown")).zfill(3),
                    "domain": value.get("domain") or flat.get("domain"),
                    "code": code,
                    "xp": float(node["xp"]),
                    "yp": float(node["yp"]),
                    "true_x": true_x,
                    "true_y": true_y,
                    "duration_ms": value.get("duration_ms"),
                }
            )
    return pd.DataFrame(rows)


def procrustes_stress(participant_xy: np.ndarray, true_xy: np.ndarray) -> Tuple[float, np.ndarray]:
    """对齐被试摆放与真值坐标，返回 RMS stress 和对齐后坐标。"""

    p = participant_xy - participant_xy.mean(axis=0)
    t = true_xy - true_xy.mean(axis=0)
    matrix = p.T @ t
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = p @ rotation
    denom = np.trace(aligned.T @ aligned)
    scale = np.trace(t.T @ aligned) / denom if denom else 1.0
    aligned = aligned * scale
    diff = aligned - t
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1)))), aligned


def plot_path_efficiency(trials: pd.DataFrame, output_dir: Path) -> None:
    """生成路径效率 trial 趋势图和 session 汇总图。"""

    if trials.empty:
        return
    group_dir = output_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, domain, title in zip(axes, ["navigation", "crafting"], ["导航", "合成"]):
        sub = trials[trials["domain"] == domain].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        for pid, group in sub.groupby("participant_id"):
            group = group.sort_values(["mix_session", "trial_id"])
            ax.plot(range(len(group)), group["path_efficiency"], marker="o", ms=4, label=f"P{pid}", alpha=0.85)
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_title(f"{title} · 路径效率")
        ax.set_xlabel("试次序号（该域内）")
        ax.set_ylabel("最优步数 / 实际步数")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(group_dir / "path_efficiency_by_trial.png", dpi=150)
    plt.close(fig)

    summary = trials.groupby(["participant_id", "domain", "mix_session"], dropna=False)["path_efficiency"].mean()
    summary = summary.reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [f"P{r.participant_id}-{str(r.domain)[:3]}-S{int(r.mix_session)}" for r in summary.itertuples()]
    ax.bar(range(len(summary)), summary["path_efficiency"], color="#4C78A8")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_title("Session × Domain 平均路径效率")
    ax.set_ylabel("平均路径效率")
    fig.tight_layout()
    fig.savefig(group_dir / "path_efficiency_by_session.png", dpi=150)
    plt.close(fig)


def plot_rt_trends(trials: pd.DataFrame, output_dir: Path) -> None:
    """生成 RT 趋势图。"""

    if trials.empty:
        return
    group_dir = output_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, column, title in zip(axes, ["total_time_ms", "mean_step_rt_ms"], ["试次总耗时", "平均步 RT"]):
        for (pid, domain), group in trials.groupby(["participant_id", "domain"]):
            group = group.sort_values(["mix_session", "trial_id"])
            ax.plot(range(len(group)), group[column], marker=".", ms=5, label=f"P{pid}-{str(domain)[:3]}")
        ax.set_title(f"{title} (ms)")
        ax.set_xlabel("试次序号（域内）")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(group_dir / "reaction_time_trends.png", dpi=150)
    plt.close(fig)


def plot_tail_b(tail_b: pd.DataFrame, output_dir: Path) -> None:
    """生成固定 B 热图和真图距散点图。"""

    if tail_b.empty:
        return
    graph_dist = np.zeros((9, 9))
    for i in range(1, 10):
        for j in range(1, 10):
            graph_dist[i - 1, j - 1] = bfs_distance(i, j)

    for (pid, domain), group in tail_b.groupby(["participant_id", "tail_domain"]):
        p_dir = output_dir / f"P{pid}"
        p_dir.mkdir(parents=True, exist_ok=True)
        mat = np.full((9, 9), np.nan)
        for row in group.itertuples():
            mat[row.node_a - 1, row.node_b - 1] = row.rating
            mat[row.node_b - 1, row.node_a - 1] = row.rating
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        im0 = axes[0].imshow(mat, vmin=1, vmax=5, cmap="YlGn", origin="upper")
        axes[0].set_title("主观可达性 B")
        im1 = axes[1].imshow(graph_dist, cmap="Blues_r", origin="upper")
        axes[1].set_title("真实图最短步数")
        im2 = axes[2].imshow(mat - (6 - graph_dist), cmap="RdBu_r", vmin=-3, vmax=3, origin="upper")
        axes[2].set_title("主观 - 距离反向")
        for ax in axes:
            ax.set_xticks(range(9))
            ax.set_yticks(range(9))
            ax.set_xticklabels(range(1, 10))
            ax.set_yticklabels(range(1, 10))
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        fig.colorbar(im2, ax=axes[2], fraction=0.046)
        fig.suptitle(f"P{pid} · {domain} · 固定 B")
        fig.tight_layout()
        fig.savefig(p_dir / f"tail_B_heatmap_{domain}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(group["graph_distance"], group["rating"], alpha=0.75, c="#E45756")
        if group["graph_distance"].nunique() > 1:
            z = np.polyfit(group["graph_distance"], group["rating"], 1)
            xs = np.linspace(group["graph_distance"].min(), group["graph_distance"].max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), color="gray", ls="--")
        corr = group["graph_distance"].corr(group["rating"], method="spearman")
        ax.set_xlabel("真实图最短步数")
        ax.set_ylabel("主观可达性 (1难-5易)")
        ax.set_title(f"P{pid} · {domain} · Spearman r={corr:.2f}")
        ax.invert_xaxis()
        fig.tight_layout()
        fig.savefig(p_dir / f"tail_B_scatter_{domain}.png", dpi=150)
        plt.close(fig)


def plot_tail_d_e(tail_d: pd.DataFrame, tail_e: pd.DataFrame, output_dir: Path) -> None:
    """生成固定 D/E 的组级图。"""

    group_dir = output_dir / "group"
    group_dir.mkdir(parents=True, exist_ok=True)
    if not tail_d.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        df = tail_d.copy()
        df["group"] = df["tail_domain"].str[:3] + " · " + df["category"].astype(str)
        groups = df.groupby("group")["fluency"].apply(list)
        ax.boxplot(groups.tolist(), tick_labels=groups.index.tolist())
        ax.set_title("固定 D · 序列整体感")
        ax.set_xlabel("域 · 类别")
        ax.set_ylabel("整体感评分 (1-5)")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        fig.savefig(group_dir / "tail_D_fluency_boxplot.png", dpi=150)
        plt.close(fig)

    if not tail_e.empty:
        rate = tail_e.groupby(["participant_id", "tail_domain"])["on_shortest_path"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [f"P{r.participant_id}-{str(r.tail_domain)[:3]}" for r in rate.itertuples()]
        ax.bar(range(len(rate)), rate["on_shortest_path"], color="#72B7B2")
        ax.set_xticks(range(len(rate)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("固定 E · 所选 hub 落在最短路上的比例")
        ax.set_ylabel("比例")
        fig.tight_layout()
        fig.savefig(group_dir / "tail_E_hub_on_shortest_path.png", dpi=150)
        plt.close(fig)


def plot_map_reconstruction(map_df: pd.DataFrame, output_dir: Path) -> Dict[str, Dict[str, float]]:
    """生成地图还原 Procrustes 图，并返回 stress 汇总。"""

    stresses: Dict[str, Dict[str, float]] = {}
    if map_df.empty:
        return stresses
    for (pid, domain), group in map_df.groupby(["participant_id", "domain"]):
        p_dir = output_dir / f"P{pid}"
        p_dir.mkdir(parents=True, exist_ok=True)
        p_xy = group.sort_values("code")[["xp", "yp"]].to_numpy()
        t_xy = group.sort_values("code")[["true_x", "true_y"]].to_numpy()
        stress, aligned = procrustes_stress(p_xy, t_xy)
        stresses.setdefault(pid, {})[str(domain)] = stress
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(t_xy[:, 0], t_xy[:, 1], s=120, c="#4C78A8", label="真实网格")
        axes[0].scatter(p_xy[:, 0], p_xy[:, 1], s=80, c="#F58518", label="被试摆放")
        for _, row in group.sort_values("code").iterrows():
            axes[0].annotate(str(int(row.code)), (row.true_x, row.true_y), ha="center", va="center", color="white")
            axes[0].annotate(str(int(row.code)), (row.xp, row.yp), ha="center", va="center", fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_title("原始摆放")
        axes[0].legend(fontsize=8)

        axes[1].scatter(t_xy[:, 0], t_xy[:, 1], s=120, c="#4C78A8")
        axes[1].scatter(aligned[:, 0], aligned[:, 1], s=80, c="#F58518")
        for idx, code in enumerate(group.sort_values("code")["code"]):
            axes[1].plot([t_xy[idx, 0], aligned[idx, 0]], [t_xy[idx, 1], aligned[idx, 1]], color="gray", lw=1)
            axes[1].annotate(str(int(code)), (t_xy[idx, 0], t_xy[idx, 1]), ha="center", va="center", color="white")
        axes[1].invert_yaxis()
        axes[1].set_title(f"Procrustes 对齐 · stress={stress:.3f}")
        fig.suptitle(f"P{pid} · {domain} · 地图还原")
        fig.tight_layout()
        fig.savefig(p_dir / f"map_reconstruction_{domain}.png", dpi=150)
        plt.close(fig)
    return stresses


def build_summary(
    participant_id: str,
    trials: pd.DataFrame,
    tail_b: pd.DataFrame,
    tail_e: pd.DataFrame,
    stresses: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """构造轻量 JSON 汇总。"""

    summary: Dict[str, Any] = {"participant_id": participant_id}
    if not trials.empty:
        by_session_domain = []
        grouped = trials.groupby(["mix_session", "domain"], dropna=False)
        for (session, domain), group in grouped:
            by_session_domain.append(
                {
                    "mix_session": int(session),
                    "domain": domain,
                    "n": int(len(group)),
                    "path_eff": float(group["path_efficiency"].mean()),
                    "rt": float(group["total_time_ms"].mean()),
                }
            )
        summary["main_trials"] = {
            "n_trials": int(len(trials)),
            "mean_path_efficiency": float(trials["path_efficiency"].mean()),
            "mean_excess_steps": float(trials["excess_steps"].mean()),
            "success_rate": float((trials["outcome"] == "success").mean()) if "outcome" in trials else np.nan,
            "by_session_domain": by_session_domain,
        }
    if not tail_b.empty:
        summary["tail_B"] = {
            "spearman_graph_vs_rating": float(tail_b["graph_distance"].corr(tail_b["rating"], method="spearman")),
            "mean_rating": float(tail_b["rating"].mean()),
        }
    if not tail_e.empty:
        summary["tail_E"] = {"hub_on_shortest_path_rate": float(tail_e["on_shortest_path"].mean())}
    if stresses.get(participant_id):
        summary["map_reconstruction_stress"] = stresses[participant_id]
    return summary


def discover_input_files(input_dir: Path | None, input_files: Sequence[Path]) -> List[Path]:
    """发现待处理 JSON 文件。"""

    if input_files:
        return [fp.resolve() for fp in input_files]
    search_dirs = [input_dir] if input_dir is not None else list(DEFAULT_INPUT_DIRS)
    files: List[Path] = []
    for directory in search_dirs:
        if directory and directory.exists():
            files.extend(sorted(directory.glob("mix_nc_jspsych_*.json")))
    unique = list(dict.fromkeys(fp.resolve() for fp in files))
    if not unique:
        raise FileNotFoundError("没有找到 mix_nc_jspsych_*.json；请用 --input-dir 或 --input-files 指定数据。")
    return unique


def process_file(json_path: Path, output_dir: Path) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """处理单个被试 JSON，生成被试级图和汇总。"""

    records = load_json_records(json_path)
    pid = participant_id_from_records(records)
    trials = extract_main_trials(records)
    tail_b = extract_tail_b(records)
    tail_d = extract_tail_d(records)
    tail_e = extract_tail_e(records)
    map_df = extract_map_reconstruction(records)
    stresses = plot_map_reconstruction(map_df, output_dir)
    plot_tail_b(tail_b, output_dir)
    summary = build_summary(pid, trials, tail_b, tail_e, stresses)
    p_dir = output_dir / f"P{pid}"
    p_dir.mkdir(parents=True, exist_ok=True)
    with (p_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, {"trials": trials, "tail_b": tail_b, "tail_d": tail_d, "tail_e": tail_e}


def concatenate_tables(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """安全拼接一组 DataFrame。"""

    non_empty = [df for df in tables if not df.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=None, help="包含 mix_nc_jspsych_*.json 的目录")
    parser.add_argument("--input-files", type=Path, nargs="*", default=(), help="显式指定一个或多个 JSON 文件")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="图表输出目录")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_input_files(args.input_dir, args.input_files)

    summaries: List[Dict[str, Any]] = []
    all_tables: Dict[str, List[pd.DataFrame]] = {"trials": [], "tail_b": [], "tail_d": [], "tail_e": []}
    for fp in files:
        summary, tables = process_file(fp, output_dir)
        summaries.append(summary)
        for key, df in tables.items():
            all_tables[key].append(df)

    trials = concatenate_tables(all_tables["trials"])
    tail_d = concatenate_tables(all_tables["tail_d"])
    tail_e = concatenate_tables(all_tables["tail_e"])
    plot_path_efficiency(trials, output_dir)
    plot_rt_trends(trials, output_dir)
    plot_tail_d_e(tail_d, tail_e, output_dir)

    with (output_dir / "all_summaries.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"完成：处理 {len(files)} 个 JSON，图表输出到 {output_dir}")


if __name__ == "__main__":
    main()
