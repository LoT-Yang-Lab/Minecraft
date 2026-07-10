# -*- coding: utf-8 -*-
"""
Cognitive Map 构建与可视化脚本
===============================
支持两种模式：

``--mode simulated``（默认）:
    生成 3x3 网格随机游走 trial 数据，使用 4 种距离方法
    （SR / LoPS / Action JS / Transition Similarity）构建认知地图，
    并渲染为交互式 D3.js HTML 可视化。

``--mode fully-connected``:
    生成全连接图的 trial 数据（所有状态对之间都有转移边），
    使用 LoPS 方法构建认知地图并可视化。不需要单独的 fully_connected 脚本。

3x3 网格结构（节点编号 1-9）：
    ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │
    ├───┼───┼───┤
    │ 4 │ 5 │ 6 │
    ├───┼───┼───┤
    │ 7 │ 8 │ 9 │
    └───┴───┴───┘
"""

import argparse
import os
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from cognitivemap._env import init  # noqa: E402

init()

from cognitivemap.map_estimation import (  # noqa: E402
    Trial,
    build_cognitive_map,
    compute_action_js_distance,
    compute_lops_distance,
    compute_sr_distance,
    compute_transition_similarity_distance,
    enrich_from_trials,
    generate_simulated_trials,
)
from cognitivemap.map_estimation.visualization import render_cognitive_map_html  # noqa: E402

# ============================================================================
# 全连接图数据生成
# ============================================================================


def generate_fully_connected_trials(
    n_states: int = 9,
    n_trials: int = 500,
    seed: int = 123,
) -> List[Trial]:
    """生成全连接图的 trial 数据。

    每个状态都可以转移到任意其他状态，但转移概率不同：
    - 相邻编号的状态转移概率更高（模拟空间邻近）
    - 编号差越大的状态转移概率越低
    """
    rng = np.random.RandomState(seed)
    # 状态编号 1..n_states
    states_range = list(range(1, n_states + 1))
    actions = [f"a{i}" for i in range(n_states)]

    trials = []
    for _ in range(n_trials):
        current = int(rng.choice(states_range))
        state_seq = [current]
        action_seq = []

        length = rng.randint(10, 40)
        for _ in range(length):
            weights = np.zeros(n_states, dtype=np.float64)
            for j, s in enumerate(states_range):
                dist = abs(current - s)
                weights[j] = np.exp(-dist * 0.5) + rng.random() * 0.3

            # 自环概率降低
            current_idx = current - 1
            weights[current_idx] *= 0.3
            weights /= weights.sum()

            next_state = int(rng.choice(states_range, p=weights))
            action_weights = np.ones(len(actions), dtype=np.float64)
            action_weights[next_state - 1] += 3.0
            action_weights /= action_weights.sum()
            action = actions[int(rng.choice(len(actions), p=action_weights))]

            state_seq.append(next_state)
            action_seq.append(action)
            current = next_state

        trials.append(Trial(state_sequence=state_seq, action_sequence=action_seq))

    return trials


# ============================================================================
# 模式：simulated（默认）
# ============================================================================


def run_simulated(
    output_dir: str,
    n_trials: int,
    seed: int,
    threshold: float,
    top_k: int,
) -> None:
    """运行九宫格随机游走示例，并用四种距离方法构建认知地图。"""

    print("=" * 60)
    print("Cognitive Map 可视化 — simulated 模式")
    print("=" * 60)

    # 1. 生成模拟数据
    print(f"\n[1/4] 生成模拟 trial 数据 (n={n_trials}, seed={seed})...")
    trials = generate_simulated_trials(n_trials=n_trials, seed=seed)
    print(f"  生成了 {len(trials)} 个 trial")

    all_states = set()
    total_steps = 0
    for trial in trials:
        all_states.update(trial.state_sequence)
        total_steps += len(trial.state_sequence)
    print(f"  涉及唯一状态数: {len(all_states)}")
    print(f"  平均 trial 长度: {total_steps / len(trials):.1f} 步")

    # 2. 4 种距离方法
    print("\n[2/4] 计算距离矩阵并构建 cognitive map...")
    methods = {
        "sr": ("Successor Representation", compute_sr_distance),
        "lops": ("LoPS-based", compute_lops_distance),
        "action_js": ("Action JS Divergence", compute_action_js_distance),
        "transition_similarity": ("Transition Similarity", compute_transition_similarity_distance),
    }

    results: Dict[str, object] = {}
    for method_key, (method_name, distance_func) in methods.items():
        print(f"\n  方法: {method_name} ({method_key})")
        distance_matrix, state_labels = distance_func(trials)
        result = build_cognitive_map(distance_matrix=distance_matrix, state_labels=state_labels, method=method_key)
        results[method_key] = result
        print(f"    状态数: {len(state_labels)}, MDS stress: {result.stress:.4f}")

    # 3. 富化
    print("\n[3/4] 富化中间数据...")
    for method_key, result in results.items():
        results[method_key] = enrich_from_trials(result, trials)
        enriched = results[method_key]
        print(f"  {method_key}: edges={len(enriched.edge_actions) if enriched.edge_actions else 0}")

    # 4. 渲染 HTML
    print("\n[4/4] 渲染 HTML 可视化...")
    for method_key, result in results.items():
        method_name = methods[method_key][0]
        output_path = os.path.join(output_dir, f"cognitive_map_{method_key}.html")
        saved_path = render_cognitive_map_html(
            result,
            output_path,
            title=f"Cognitive Map — {method_name}",
            threshold=threshold,
            top_k=top_k,
        )
        print(f"  {method_key}: {saved_path}")

    print("\n" + "=" * 60)
    print("所有 cognitive map 可视化完成！")
    print(f"HTML 文件保存在: {output_dir}/")


# ============================================================================
# 模式：fully-connected
# ============================================================================


def run_fully_connected(
    output_dir: str,
    fc_n_states: int,
    fc_n_trials: int,
    fc_seed: int,
    threshold: float,
    top_k: int,
) -> None:
    """运行全连接图示例，并用 LoPS 距离构建认知地图。"""

    print("=" * 60)
    print("Cognitive Map 可视化 — fully-connected 模式")
    print("=" * 60)

    n_states = fc_n_states
    print(f"\n[1/3] 生成全连接图 trial 数据 ({n_states} 状态)...")
    trials = generate_fully_connected_trials(n_states=n_states, n_trials=fc_n_trials, seed=fc_seed)

    all_states = set()
    for t in trials:
        all_states.update(t.state_sequence)
    print(f"  生成了 {len(trials)} 个 trial, 涉及唯一状态数: {len(all_states)}")

    edge_counter = Counter()
    for t in trials:
        for s, ns in zip(t.state_sequence, t.state_sequence[1:]):
            edge_counter[(s, ns)] += 1
    print(f"  有向边数: {len(edge_counter)} (理论最大 {n_states * n_states})")

    print("\n[2/3] 计算 LoPS 距离 + 构建 cognitive map...")
    distance_matrix, state_labels = compute_lops_distance(trials)
    result = build_cognitive_map(distance_matrix=distance_matrix, state_labels=state_labels, method="lops")
    print(f"  状态数: {len(state_labels)}, MDS stress: {result.stress:.4f}")

    result = enrich_from_trials(result, trials)
    edge_count = len(result.edge_actions) if result.edge_actions else 0
    print(f"  edge_actions 边数: {edge_count}")

    print("\n[3/3] 渲染全连接图 HTML...")
    output_path = os.path.join(output_dir, "cognitive_map_fully_connected.html")
    saved = render_cognitive_map_html(
        result,
        output_path,
        title="Cognitive Map — Fully Connected (LoPS)",
        threshold=threshold,
        top_k=top_k,
    )
    print(f"  已保存: {saved}")

    print("\n" + "=" * 60)
    print("全连接图可视化完成！")


# ============================================================================
# CLI
# ============================================================================


def main():
    """命令行入口：根据模式选择模拟数据并渲染认知地图 HTML。"""

    parser = argparse.ArgumentParser(description="构建并可视化 Cognitive Map。")

    parser.add_argument(
        "--mode",
        choices=["simulated", "fully-connected"],
        default="simulated",
        help="数据生成模式 (default: simulated)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figs/demo/cognitive_maps",
        help="输出目录 (default: results/figs/demo/cognitive_maps)",
    )

    # simulated 模式参数
    parser.add_argument("--n-trials", type=int, default=200, help="simulated 模式的 trial 数量")
    parser.add_argument("--seed", type=int, default=42, help="simulated 模式的随机种子")

    # fully-connected 模式参数
    parser.add_argument("--fc-n-states", type=int, default=9, help="fully-connected 模式的状态数")
    parser.add_argument("--fc-n-trials", type=int, default=500, help="fully-connected 模式的 trial 数量")
    parser.add_argument("--fc-seed", type=int, default=123, help="fully-connected 模式的随机种子")

    # 可视化参数
    parser.add_argument("--threshold", type=float, default=0.2, help="主 action 比例阈值")
    parser.add_argument("--top-k", type=int, default=2, help="每条边最多显示几个 action 胶囊")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "simulated":
        run_simulated(
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            seed=args.seed,
            threshold=args.threshold,
            top_k=args.top_k,
        )
    else:
        run_fully_connected(
            output_dir=args.output_dir,
            fc_n_states=args.fc_n_states,
            fc_n_trials=args.fc_n_trials,
            fc_seed=args.fc_seed,
            threshold=args.threshold,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
