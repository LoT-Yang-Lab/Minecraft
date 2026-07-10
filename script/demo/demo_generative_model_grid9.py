# -*- coding: utf-8 -*-
"""3x3 九宫格认知导航生成模型示例。

本示例展示如何使用 ``cognitivemap.generative_model`` 中的新 agent API：

1. 构建一个 3x3 九宫格 TaskGraph。
2. 定义 primitive、chunk、landmark 三种不同表征。
3. 为每种表征创建一个 CognitiveNavigationAgent。
4. 随机生成若干 start-goal trial，并让三个 agent 分别完成。

九宫格状态编号：

    1  2  3
    4  5  6
    7  8  9

primitive action 包含 U、D、L、R 和 LOOP。
其中 U/D/L/R 是普通相邻移动；LOOP 参考题图中的橙色外环：

    1 --LOOP--> 3 --LOOP--> 9 --LOOP--> 7 --LOOP--> 1

LOOP 只在四个角状态合法。边界处不可执行的 U/D/L/R 不会被列入 legal actions。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from cognitivemap._env import init  # noqa: E402

init()

from cognitivemap.generative_model import (  # noqa: E402
    CognitiveNavigationAgent,
    LandmarkRepresentation,
    PlanningConfig,
    RepresentationBuilder,
    TaskGraph,
    TrialSpec,
)

ACTION_ORDER = ("U", "D", "L", "R", "LOOP")
GRID_STATES = tuple(range(1, 10))
LANDMARKS = (1, 5, 9)
NON_LANDMARKS = (2, 3, 4, 6, 7, 8)


@dataclass(frozen=True)
class NavigationJob:
    """一个随机生成的导航任务，只包含 start、goal 和复现 seed。"""

    trial_id: str
    start_state: int
    goal_state: int
    random_seed: int


def build_grid9_task() -> TaskGraph:
    """构建题图对应的 3x3 primitive 任务图。

    为了让 primitive candidate 顺序固定为 U/D/L/R/LOOP，这里直接构造 TaskGraph，
    而不是依赖嵌套 transition 字典中的首次出现顺序。
    """

    transitions: dict[tuple[int, str], int] = {}
    legal_actions_by_state: dict[int, list[str]] = {state: [] for state in GRID_STATES}

    for state in GRID_STATES:
        row, col = divmod(state - 1, 3)
        candidates = {
            "U": state - 3 if row > 0 else None,
            "D": state + 3 if row < 2 else None,
            "L": state - 1 if col > 0 else None,
            "R": state + 1 if col < 2 else None,
        }
        for action in ACTION_ORDER:
            if action == "LOOP":
                continue
            target = candidates[action]
            if target is None:
                continue
            legal_actions_by_state[state].append(action)
            transitions[(state, action)] = target

    loop_edges = {
        1: 3,
        3: 9,
        9: 7,
        7: 1,
    }
    for source, target in loop_edges.items():
        legal_actions_by_state[source].append("LOOP")
        transitions[(source, "LOOP")] = target

    action_costs = {key: 1.0 for key in transitions}
    return TaskGraph(
        task_id="grid9-with-loop",
        states=GRID_STATES,
        actions=ACTION_ORDER,
        legal_actions_by_state=legal_actions_by_state,
        transitions=transitions,
        action_costs=action_costs,
    )


def build_primitive_agent(task: TaskGraph, config: PlanningConfig) -> CognitiveNavigationAgent:
    """构建 primitive agent：每个 state 的合法 primitive action 均匀选择。"""

    representation = RepresentationBuilder.build_primitive(task, "primitive-uniform-grid9")
    return CognitiveNavigationAgent(representation=representation, config=config)


def build_chunk_agent(task: TaskGraph, config: PlanningConfig) -> CognitiveNavigationAgent:
    """构建 chunk agent。

    chunk inventory 按用户给定的五个 chunk 设置：
    LOOP-LOOP、L-D、U-R、D-L、R-U。
    每个 state 先列 primitive tokens，再列从该 state 可完整执行的 chunks；
    本示例使用 uniform token 权重。
    """

    chunk_inventory = {
        "loop_loop": ("LOOP", "LOOP"),
        "l_d": ("L", "D"),
        "u_r": ("U", "R"),
        "d_l": ("D", "L"),
        "r_u": ("R", "U"),
    }
    representation = RepresentationBuilder.build_chunk(
        task,
        representation_id="chunk-grid9-five-patterns",
        chunk_inventory=chunk_inventory,
    )
    return CognitiveNavigationAgent(representation=representation, config=config)


def build_landmark_agent(config: PlanningConfig) -> CognitiveNavigationAgent:
    """构建 landmark agent。

    数值参考 ``docs/method/analysis-7-1.tex`` 的
    ``Worked 3 x 3 landmark example``：

    - landmark set 为 {1, 5, 9}
    - 每个 landmark 到另外两个 landmark 的概率为 0.30
    - 每个 landmark 到每个 non-landmark 的概率为 1/15
    - non-landmark 到 landmark 的 CPD 使用文档表格
    - non-landmark 到 non-landmark 在 kernel 中固定为 0
    """

    p_ll = {source: {target: (0.0 if source == target else 0.30) for target in LANDMARKS} for source in LANDMARKS}
    p_lu = {source: {target: 1.0 / 15.0 for target in NON_LANDMARKS} for source in LANDMARKS}
    p_ul = {
        2: {1: 0.60, 5: 0.25, 9: 0.15},
        3: {1: 0.55, 5: 0.30, 9: 0.15},
        4: {1: 0.30, 5: 0.55, 9: 0.15},
        6: {1: 0.15, 5: 0.55, 9: 0.30},
        7: {1: 0.15, 5: 0.30, 9: 0.55},
        8: {1: 0.15, 5: 0.25, 9: 0.60},
    }
    representation = LandmarkRepresentation(
        representation_id="landmark-grid9-worked-example",
        landmarks=LANDMARKS,
        p_ll=p_ll,
        p_lu=p_lu,
        p_ul=p_ul,
    )
    return CognitiveNavigationAgent(representation=representation, config=config)


def generate_navigation_jobs(n_trials: int, seed: int) -> list[NavigationJob]:
    """随机生成 start != goal 的导航任务。"""

    rng = np.random.default_rng(seed)
    jobs: list[NavigationJob] = []
    for index in range(n_trials):
        start = int(rng.choice(GRID_STATES))
        possible_goals = [state for state in GRID_STATES if state != start]
        goal = int(rng.choice(possible_goals))
        trial_seed = int(rng.integers(0, 2**31 - 1))
        jobs.append(
            NavigationJob(
                trial_id=f"grid9-random-{index + 1:03d}",
                start_state=start,
                goal_state=goal,
                random_seed=trial_seed,
            )
        )
    return jobs


def make_trial(job: NavigationJob, family: str) -> TrialSpec:
    """把通用 NavigationJob 转换为某个 family 对应的 TrialSpec。"""

    return TrialSpec(
        trial_id=job.trial_id,
        start_state=job.start_state,
        goal_state=job.goal_state,
        family=family,
        random_seed=job.random_seed,
    )


def run_agents(
    task: TaskGraph,
    agents: dict[str, CognitiveNavigationAgent],
    jobs: Iterable[NavigationJob],
) -> dict[str, object]:
    """让三个 agent 对同一批 start-goal 任务分别运行。"""

    results_by_name = {}
    jobs = list(jobs)
    for name, agent in agents.items():
        family = agent.representation.family.value
        trials = [make_trial(job, family) for job in jobs]
        results_by_name[name] = agent.run_batch(task, trials)
    return results_by_name


def print_task_summary(task: TaskGraph) -> None:
    """打印九宫格任务结构摘要。"""

    print("=" * 72)
    print("3x3 九宫格认知导航 demo")
    print("=" * 72)
    print(f"Task: {task.task_id}")
    print(f"States: {list(task.states)}")
    print(f"Actions: {list(task.actions)}")
    print("LOOP: 1 -> 3 -> 9 -> 7 -> 1")
    print()
    print("每个状态的合法动作：")
    for state in task.states:
        print(f"  {state}: {task.legal_actions(state)}")


def print_result_summary(results_by_name: dict[str, object]) -> None:
    """打印每个 agent 的 trial 运行摘要。"""

    print("\n运行结果：")
    for name, results in results_by_name.items():
        print("\n" + "-" * 72)
        print(f"Agent: {name}")
        print("-" * 72)
        for result in results:
            astar_path = list(result.astar.cognitive_path) if result.astar else []
            actions = list(result.expansion.actions) if result.expansion else []
            states = list(result.expansion.states) if result.expansion else []
            status = result.status.value
            print(
                f"{result.trial.trial_id}: "
                f"{result.trial.start_state}->{result.trial.goal_state} | "
                f"status={status:<13} | "
                f"cognitive_path={astar_path} | "
                f"actions={actions} | "
                f"states={states}"
            )


def save_json_outputs(
    output_dir: Path,
    agents: dict[str, CognitiveNavigationAgent],
    results_by_name: dict[str, object],
) -> None:
    """把每个 agent 的完整运行结果保存为 JSON。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, agent in agents.items():
        output_path = output_dir / f"grid9_{name}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(agent.to_dict(results_by_name[name]), file, ensure_ascii=False, indent=2)
        print(f"  已保存: {output_path}")


def main() -> None:
    """命令行入口：运行九宫格三类 agent 示例并按需保存 JSON。"""

    parser = argparse.ArgumentParser(description="运行 3x3 九宫格认知导航生成模型 demo。")
    parser.add_argument("--n-trials", type=int, default=8, help="随机 start-goal 任务数量")
    parser.add_argument("--seed", type=int, default=2026, help="随机任务生成种子")
    parser.add_argument("--alpha", type=float, default=0.0, help="A* embedding heuristic scale，0 表示 Dijkstra")
    parser.add_argument(
        "--output-dir",
        default="data/demo/generative_model_grid9",
        help="JSON 输出目录 (default: data/demo/generative_model_grid9)",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="只打印结果，不保存 JSON",
    )
    args = parser.parse_args()

    task = build_grid9_task()
    config = PlanningConfig(alpha=args.alpha)
    agents = {
        "primitive": build_primitive_agent(task, config),
        "chunk": build_chunk_agent(task, config),
        "landmark": build_landmark_agent(config),
    }
    jobs = generate_navigation_jobs(args.n_trials, args.seed)
    results_by_name = run_agents(task, agents, jobs)

    print_task_summary(task)
    print_result_summary(results_by_name)

    if not args.no_save_json:
        print("\n保存完整 JSON 结果：")
        save_json_outputs(Path(args.output_dir), agents, results_by_name)


if __name__ == "__main__":
    main()
