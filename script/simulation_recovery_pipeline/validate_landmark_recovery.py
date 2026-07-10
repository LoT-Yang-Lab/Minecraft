# -*- coding: utf-8 -*-
"""
使用生成模型验证 landmark 推断算法的恢复能力。

这个脚本实现一条完整的模型恢复检验链路：
1. 预先指定真实 landmark 集合 ``L_true``；
2. 根据真实集合构造 ``LandmarkRepresentation`` 的 LL/LU/UL 条件概率表；
3. 用 ``CognitiveNavigationAgent`` 在九宫格任务上生成模拟轨迹；
4. 仅把生成出的 state 序列交给数据驱动的 landmark 推断算法；
5. 将推断出的 landmark 与真实集合比较，输出 precision/recall/F1 等指标。

这里的生成器刻意保持小而透明。脚本只负责构造实验条件、抽样数据集和汇总指标；
具体的路径展开、landmark 决策与 primitive 规划仍由 ``cognitivemap.generative_model`` 提供。
"""

from __future__ import annotations

import argparse
import itertools
import json

# unused import
import math  # noqa: F401
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# unused import
from typing import Iterable  # noqa: F401
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cognitivemap._env import init  # noqa: E402

init()

from cognitivemap.generative_model import (  # noqa: E402
    CognitiveNavigationAgent,
    LandmarkRepresentation,
    PlanningConfig,
    TaskGraph,
    TrialSpec,
)
from cognitivemap.inference.miner_landmark import LandmarkMiningConfig, mine_landmarks  # noqa: E402

ACTION_ORDER = ("U", "D", "L", "R", "LOOP")
GRID_STATES = tuple(range(1, 10))
ALLOCATION_STRATEGIES = ("distance",)

DEFAULT_SIMULATED_DATA_OUTPUT = Path("data/simulated_recovery_pipeline/simulated_datasets/landmark_grid9.joblib")
DEFAULT_QUICK_SIMULATED_DATA_OUTPUT = Path(
    "data/simulated_recovery_pipeline/simulated_datasets/landmark_grid9_quick.joblib"
)
DEFAULT_RECOVERY_OUTPUT = Path("data/simulated_recovery_pipeline/recovery_results/landmark_recovery.joblib")
DEFAULT_QUICK_RECOVERY_OUTPUT = Path("data/simulated_recovery_pipeline/recovery_results/landmark_recovery_quick.joblib")


@dataclass(frozen=True)
class ValidationCondition:
    """一次模拟恢复实验的完整条件配置。

    ``base_condition_id`` 标识共享同一个轨迹池的基础条件，``dataset_index`` 标识从该池中抽样出的
    第几个数据集。这样可以把“生成轨迹池”和“抽样检验数据集”解耦，便于比较样本量和随机种子的影响。
    """

    condition_id: str
    base_condition_id: str
    landmarks: Tuple[int, ...]
    n_trials: int
    pool_n_trials: int
    mass_ll: float
    ll_strategy: str
    lu_strategy: str
    ul_strategy: str
    beta_ll: float
    beta_lu: float
    beta_ul: float
    planning_alpha: float
    landmark_path_length_beta: float
    max_landmark_simple_paths: int
    seed: int
    pool_seed: int
    sample_seed: int
    dataset_index: int
    miner_bootstrap_iterations: int
    miner_max_landmarks: int
    miner_selection_threshold: float
    miner_bootstrap_sample_ratio: float


def build_grid9_task() -> TaskGraph:
    """构造九宫格加有向 LOOP 边的 primitive 任务图。

    普通动作 ``U/D/L/R`` 是网格中的相邻移动，``LOOP`` 是沿角点顺时针跳转的特殊动作：
    ``1 -> 3 -> 9 -> 7 -> 1``。该结构与生成模型 demo 保持一致，方便模拟数据和示例互相对照。
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

    loop_edges = {1: 3, 3: 9, 9: 7, 7: 1}
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


def build_landmark_representation(
    task: TaskGraph,
    landmarks: Sequence[int],
    mass_ll: float,
    ll_strategy: str,
    lu_strategy: str,
    ul_strategy: str,
    beta_ll: float,
    beta_lu: float,
    beta_ul: float,
    random_seed: int,
) -> LandmarkRepresentation:
    """根据真实 landmark 集合构造 LL/LU/UL 三块条件概率表。

    当前版本只保留距离 softmax 分配策略。对于 landmark 行，先用 ``mass_ll`` 控制“去往另一个
    landmark”和“回到普通状态”的总概率质量，再在对应目标集合内部按最短路距离分配：

    ``p(target | source) ∝ exp(-beta * d*(source, target))``
    """

    landmarks = tuple(int(state) for state in landmarks)
    if len(set(landmarks)) != len(landmarks) or len(landmarks) < 2:
        raise ValueError("landmarks must contain at least two unique states")
    if not 0 <= mass_ll <= 1:
        raise ValueError("mass_ll must be in [0, 1]")
    for name, strategy in {
        "ll_strategy": ll_strategy,
        "lu_strategy": lu_strategy,
        "ul_strategy": ul_strategy,
    }.items():
        if strategy not in ALLOCATION_STRATEGIES:
            raise ValueError(f"{name} must be one of {ALLOCATION_STRATEGIES}")
    for name, beta in {"beta_ll": beta_ll, "beta_lu": beta_lu, "beta_ul": beta_ul}.items():
        if beta < 0:
            raise ValueError(f"{name} must be non-negative")
    if beta_ul < 0:
        raise ValueError("beta_ul must be non-negative")

    landmark_set = set(landmarks)
    non_landmarks = tuple(state for state in task.states if state not in landmark_set)
    if not non_landmarks:
        raise ValueError("at least one non-landmark state is required")

    mass_lu = 1.0 - mass_ll
    rng = np.random.default_rng(random_seed)
    p_ll = {}
    p_lu = {}
    for source in landmarks:
        ll_targets = tuple(target for target in landmarks if target != source)
        ll_weights = allocate_target_probabilities(task, source, ll_targets, ll_strategy, beta_ll, rng)
        lu_weights = allocate_target_probabilities(task, source, non_landmarks, lu_strategy, beta_lu, rng)
        p_ll[source] = {target: 0.0 if target == source else mass_ll * ll_weights[target] for target in landmarks}
        p_lu[source] = {target: mass_lu * lu_weights[target] for target in non_landmarks}

    p_ul = {}
    for source in non_landmarks:
        p_ul[source] = allocate_target_probabilities(task, source, landmarks, ul_strategy, beta_ul, rng)

    return LandmarkRepresentation(
        representation_id=(
            f"synthetic-landmark-{'_'.join(map(str, landmarks))}" f"-ll-{ll_strategy}-lu-{lu_strategy}-ul-{ul_strategy}"
        ),
        landmarks=landmarks,
        p_ll=p_ll,
        p_lu=p_lu,
        p_ul=p_ul,
    )


def allocate_target_probabilities(
    task: TaskGraph,
    source: int,
    targets: Sequence[int],
    strategy: str,
    beta: float,
    rng: np.random.Generator,
) -> Dict[int, float]:
    """在候选目标集合上分配一行概率。

    距离越短的目标获得越大的概率；减去最大 logit 是标准的数值稳定写法，避免较大 ``beta`` 下指数溢出。
    ``rng`` 作为参数保留，是为了让未来重新加入随机分配策略时仍能共享同一随机源。
    """

    targets = tuple(int(target) for target in targets)
    if not targets:
        return {}
    if strategy == "distance":
        distances = np.asarray([task.shortest_distance(source, target) for target in targets], dtype=np.float64)
        if not np.all(np.isfinite(distances)):
            raise ValueError(f"distance strategy found unreachable targets from source {source}: {targets}")
        logits = -beta * distances
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
    else:
        raise ValueError(f"unknown allocation strategy: {strategy}")
    return {target: float(weight) for target, weight in zip(targets, weights)}


def generate_trial_specs(n_trials: int, seed: int) -> List[TrialSpec]:
    """生成一批起点、终点不同的模拟 trial 规格。"""

    rng = np.random.default_rng(seed)
    trials = []
    for index in range(n_trials):
        start = int(rng.choice(GRID_STATES))
        possible_goals = [state for state in GRID_STATES if state != start]
        goal = int(rng.choice(possible_goals))
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trials.append(
            TrialSpec(
                trial_id=f"synthetic-{index + 1:04d}",
                start_state=start,
                goal_state=goal,
                family="landmark",
                random_seed=trial_seed,
            )
        )
    return trials


def generate_sequence_pool(condition: ValidationCondition) -> Tuple[List[List[int]], Dict[str, Any]]:
    """为一个基础条件生成可重复抽样的轨迹池。

    轨迹池规模由 ``pool_n_trials`` 控制；后续恢复实验会从池中无放回抽取 ``n_trials`` 条轨迹。
    这样同一基础条件下的多个随机数据集可以共享相同生成机制，同时保持样本抽取差异。
    """

    task = build_grid9_task()
    representation = build_landmark_representation(
        task,
        landmarks=condition.landmarks,
        mass_ll=condition.mass_ll,
        ll_strategy=condition.ll_strategy,
        lu_strategy=condition.lu_strategy,
        ul_strategy=condition.ul_strategy,
        beta_ll=condition.beta_ll,
        beta_lu=condition.beta_lu,
        beta_ul=condition.beta_ul,
        random_seed=condition.pool_seed + 17_071,
    )
    agent = CognitiveNavigationAgent(
        representation=representation,
        config=PlanningConfig(
            alpha=condition.planning_alpha,
            landmark_path_length_beta=condition.landmark_path_length_beta,
            max_landmark_simple_paths=condition.max_landmark_simple_paths,
        ),
    )
    trials = generate_trial_specs(condition.pool_n_trials, condition.pool_seed)
    results = agent.run_batch(task, trials)

    failed = [result for result in results if result.expansion is None]
    if failed:
        raise RuntimeError(f"{len(failed)} generated trials failed in condition {condition.base_condition_id}")

    state_sequences = [list(result.expansion.states) for result in results]
    return state_sequences, summarize_generated_sequences(state_sequences, condition.landmarks)


def generate_sequence_pools(
    conditions: Sequence[ValidationCondition],
) -> Dict[str, Tuple[List[List[int]], Dict[str, Any]]]:
    """为每个基础条件只生成一次轨迹池，避免重复模拟相同配置。"""

    pools: Dict[str, Tuple[List[List[int]], Dict[str, Any]]] = {}
    for condition in conditions:
        if condition.base_condition_id not in pools:
            pools[condition.base_condition_id] = generate_sequence_pool(condition)
    return pools


def build_generator_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    """构造写入 joblib 的生成器元数据，便于之后审计模拟数据来源。"""

    return {
        "cpd_rule": "LL/LU/UL allocation by distance-softmax",
        "states": list(GRID_STATES),
        "actions": list(ACTION_ORDER),
        "allocation_strategies": list(ALLOCATION_STRATEGIES),
        "pool_n_trials": args.pool_trials,
        "sampled_datasets_per_base": resolve_n_datasets(args),
    }


def build_simulated_dataset_payload(
    args: argparse.Namespace,
    conditions: Sequence[ValidationCondition],
    pools: Mapping[str, Tuple[List[List[int]], Mapping[str, Any]]],
) -> Dict[str, Any]:
    """将模拟轨迹池整理成可持久化的 joblib payload。"""

    first_condition_by_base: Dict[str, ValidationCondition] = {}
    for condition in conditions:
        first_condition_by_base.setdefault(condition.base_condition_id, condition)

    return {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task_id": "grid9-with-loop",
        "generator": build_generator_metadata(args),
        "args": vars(args),
        "conditions": [asdict(condition) for condition in conditions],
        "sequence_pools": {
            base_condition_id: {
                "condition": asdict(first_condition_by_base[base_condition_id]),
                "state_sequences": [list(sequence) for sequence in sequences],
                "summary": dict(summary),
            }
            for base_condition_id, (sequences, summary) in sorted(pools.items())
        },
    }


def load_simulated_dataset_payload(
    path: Path,
) -> Tuple[List[ValidationCondition], Dict[str, Tuple[List[List[int]], Dict[str, Any]]], Dict[str, Any]]:
    """读取由本脚本保存的模拟轨迹池，并恢复为内部使用的数据结构。"""

    payload = joblib.load(path)
    if "conditions" not in payload or "sequence_pools" not in payload:
        raise ValueError(f"模拟数据文件缺少 conditions/sequence_pools 字段: {path}")

    conditions = []
    for raw_condition in payload["conditions"]:
        condition = dict(raw_condition)
        condition["landmarks"] = tuple(int(state) for state in condition["landmarks"])
        conditions.append(ValidationCondition(**condition))

    pools = {}
    for base_condition_id, raw_pool in payload["sequence_pools"].items():
        sequences = [list(sequence) for sequence in raw_pool["state_sequences"]]
        pools[base_condition_id] = (sequences, dict(raw_pool.get("summary", {})))

    return conditions, pools, payload


def sample_state_sequences(
    pool_sequences: Sequence[Sequence[int]],
    n_trials: int,
    sample_seed: int,
) -> Tuple[List[List[int]], List[int]]:
    """从轨迹池中无放回抽取指定数量的 state 序列。"""

    if n_trials > len(pool_sequences):
        raise ValueError(f"sample n_trials={n_trials} exceeds pool size={len(pool_sequences)}")

    rng = np.random.default_rng(sample_seed)
    sample_indices = rng.choice(len(pool_sequences), size=n_trials, replace=False)
    sampled = [list(pool_sequences[int(index)]) for index in sample_indices]
    return sampled, [int(index) for index in sample_indices]


def mine_sampled_condition(
    condition: ValidationCondition,
    pool_sequences: Sequence[Sequence[int]],
    pool_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """对一个抽样数据集运行 landmark 推断，并计算恢复指标。"""

    state_sequences, sample_indices = sample_state_sequences(
        pool_sequences,
        condition.n_trials,
        condition.sample_seed,
    )
    miner_config = LandmarkMiningConfig(
        max_landmarks=condition.miner_max_landmarks,
        bootstrap_iterations=condition.miner_bootstrap_iterations,
        bootstrap_sample_ratio=condition.miner_bootstrap_sample_ratio,
        selection_threshold=condition.miner_selection_threshold,
        random_state=condition.sample_seed + 100_003,
        n_jobs=1,
    )
    mining = mine_landmarks(state_sequences, miner_config)

    true_landmarks = set(condition.landmarks)
    stable_pred = set(int(state) for state in mining["landmarks"])
    top_true_k = set(int(state) for state in mining["top_landmarks"][: len(true_landmarks)])
    top_max = set(int(state) for state in mining["top_landmarks"])
    ranking = [int(item["state"]) for item in mining["candidate_ranking"]]
    rank_by_state = {state: rank + 1 for rank, state in enumerate(ranking)}

    metrics_stable = set_recovery_metrics(true_landmarks, stable_pred)
    metrics_top_true_k = set_recovery_metrics(true_landmarks, top_true_k)
    metrics_top_max = set_recovery_metrics(true_landmarks, top_max)
    true_rates = [
        float(mining["state_scores"][state]["selection_rate"])
        for state in true_landmarks
        if state in mining["state_scores"]
    ]
    false_rates = [
        float(row["selection_rate"])
        for state, row in mining["state_scores"].items()
        if int(state) not in true_landmarks
    ]

    sample_summary = summarize_generated_sequences(state_sequences, condition.landmarks)
    return {
        "condition_id": condition.condition_id,
        "base_condition_id": condition.base_condition_id,
        "condition": asdict(condition),
        "true_landmarks": list(condition.landmarks),
        "predicted_landmarks": sorted(stable_pred),
        "top_true_k_landmarks": sorted(top_true_k),
        "top_max_landmarks": [int(state) for state in mining["top_landmarks"]],
        "stable_recovered_count": len(stable_pred),
        "metrics_stable": metrics_stable,
        "metrics_top_true_k": metrics_top_true_k,
        "metrics_top_max": metrics_top_max,
        "mean_true_rank": float(np.mean([rank_by_state[state] for state in true_landmarks])),
        "max_true_rank": int(max(rank_by_state[state] for state in true_landmarks)),
        "mean_true_selection_rate": float(np.mean(true_rates)) if true_rates else 0.0,
        "max_false_selection_rate": float(max(false_rates)) if false_rates else 0.0,
        "generated": sample_summary,
        "generated_summary": sample_summary,
        "generated_pool": dict(pool_summary),
        "sample_indices": sample_indices,
        "state_scores": mining["state_scores"],
        "candidate_ranking": mining["candidate_ranking"],
    }


def run_condition(condition: ValidationCondition) -> Dict[str, Any]:
    """不复用外部轨迹池，直接生成并检验单个条件。"""

    pool_sequences, pool_summary = generate_sequence_pool(condition)
    return mine_sampled_condition(condition, pool_sequences, pool_summary)


def summarize_generated_sequences(state_sequences: Sequence[Sequence[int]], landmarks: Sequence[int]) -> Dict[str, Any]:
    """汇总生成轨迹中真实 landmark 的访问频率和覆盖率。"""

    landmark_set = set(landmarks)
    visit_counts = {state: 0 for state in landmarks}
    coverage_counts = {state: 0 for state in landmarks}
    for seq in state_sequences:
        seq_set = set(seq)
        for state in landmarks:
            visit_counts[state] += sum(1 for item in seq if item == state)
            coverage_counts[state] += int(state in seq_set)
    n_trials = max(len(state_sequences), 1)
    total_steps = sum(len(seq) for seq in state_sequences)
    landmark_visits = sum(visit_counts.values())
    return {
        "n_trials": len(state_sequences),
        "total_states_observed": total_steps,
        "mean_steps": total_steps / max(len(state_sequences), 1),
        "landmark_visit_fraction": landmark_visits / max(total_steps, 1),
        "true_landmark_visit_counts": visit_counts,
        "true_landmark_trial_coverage": {state: coverage_counts[state] / n_trials for state in landmark_set},
    }


def set_recovery_metrics(true_landmarks: set[int], predicted_landmarks: set[int]) -> Dict[str, float]:
    """计算集合恢复任务常用的 precision、recall、F1 和 Jaccard 指标。"""

    if not true_landmarks and not predicted_landmarks:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}
    intersection = true_landmarks & predicted_landmarks
    precision = len(intersection) / len(predicted_landmarks) if predicted_landmarks else 0.0
    recall = len(intersection) / len(true_landmarks) if true_landmarks else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    union = true_landmarks | predicted_landmarks
    jaccard = len(intersection) / len(union) if union else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
    }


def parse_number_list(raw: str, cast) -> list:
    """解析逗号分隔的数值参数，并使用 ``cast`` 转换元素类型。"""

    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_strategy_list(raw: str) -> List[str]:
    """解析概率分配策略列表，并校验当前版本支持的策略。"""

    strategies = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [strategy for strategy in strategies if strategy not in ALLOCATION_STRATEGIES]
    if invalid:
        raise ValueError(f"unknown allocation strategies {invalid}; expected {ALLOCATION_STRATEGIES}")
    return strategies


def build_strategy_tuples(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    """生成 LL/LU/UL 三块条件概率表的策略组合。"""

    if args.strategy_profiles:
        return [(strategy, strategy, strategy) for strategy in parse_strategy_list(args.strategy_profiles)]

    ll_strategy_values = parse_strategy_list(args.ll_strategy)
    lu_strategy_values = parse_strategy_list(args.lu_strategy)
    ul_strategy_values = parse_strategy_list(args.ul_strategy)
    return list(itertools.product(ll_strategy_values, lu_strategy_values, ul_strategy_values))


def parse_landmark_sets(raw: str) -> List[Tuple[int, ...]]:
    """解析 landmark 集合配置，支持预设集合和手写分号分隔格式。"""

    raw = raw.strip()
    if raw == "curated":
        return [(1, 5), (3, 7), (1, 5, 9), (2, 5, 8), (1, 3, 7, 9)]
    if raw == "all2":
        return list(itertools.combinations(GRID_STATES, 2))
    if raw == "all3":
        return list(itertools.combinations(GRID_STATES, 3))
    if raw == "all4":
        return list(itertools.combinations(GRID_STATES, 4))
    if raw == "all":
        return [
            *itertools.combinations(GRID_STATES, 2),
            *itertools.combinations(GRID_STATES, 3),
            *itertools.combinations(GRID_STATES, 4),
        ]

    landmark_sets = []
    for chunk in raw.split(";"):
        values = tuple(int(item.strip()) for item in chunk.split(",") if item.strip())
        if len(values) < 2:
            raise ValueError(f"invalid landmark set {chunk!r}; each set needs at least two states")
        landmark_sets.append(values)
    return landmark_sets


def build_conditions(args: argparse.Namespace) -> List[ValidationCondition]:
    """根据命令行参数展开全部模拟恢复实验条件。"""

    landmark_sets = parse_landmark_sets(args.landmark_sets)
    n_trials_values = parse_number_list(args.n_trials, int)
    pool_n_trials = int(args.pool_trials)
    if any(n_trials > pool_n_trials for n_trials in n_trials_values):
        raise ValueError(f"all n_trials values must be <= pool_trials={pool_n_trials}")
    mass_ll_values = parse_number_list(args.mass_ll, float)
    beta_ul_values = parse_number_list(args.beta_ul, float)
    beta_ll_values = parse_number_list(args.beta_ll, float) if args.beta_ll else None
    beta_lu_values = parse_number_list(args.beta_lu, float) if args.beta_lu else None
    strategy_tuples = build_strategy_tuples(args)
    n_datasets = resolve_n_datasets(args)

    conditions = []
    index = 0
    base_index = 0
    for beta_ul in beta_ul_values:
        local_beta_ll_values = beta_ll_values if beta_ll_values is not None else [beta_ul]
        local_beta_lu_values = beta_lu_values if beta_lu_values is not None else [beta_ul]
        for (
            landmarks,
            n_trials,
            mass_ll,
            strategy_tuple,
            beta_ll,
            beta_lu,
        ) in itertools.product(
            landmark_sets,
            n_trials_values,
            mass_ll_values,
            strategy_tuples,
            local_beta_ll_values,
            local_beta_lu_values,
        ):
            ll_strategy, lu_strategy, ul_strategy = strategy_tuple
            base_index += 1
            base_condition_id = f"base-{base_index:04d}"
            pool_seed = args.seed_offset + base_index
            for dataset_index in range(n_datasets):
                sample_seed = args.seed_offset + 1_000_000 + base_index * 100_000 + dataset_index
                index += 1
                conditions.append(
                    ValidationCondition(
                        condition_id=f"cond-{index:06d}",
                        base_condition_id=base_condition_id,
                        landmarks=tuple(sorted(landmarks)),
                        n_trials=n_trials,
                        pool_n_trials=pool_n_trials,
                        mass_ll=mass_ll,
                        ll_strategy=ll_strategy,
                        lu_strategy=lu_strategy,
                        ul_strategy=ul_strategy,
                        beta_ll=beta_ll,
                        beta_lu=beta_lu,
                        beta_ul=beta_ul,
                        planning_alpha=args.planning_alpha,
                        landmark_path_length_beta=args.landmark_path_length_beta,
                        max_landmark_simple_paths=args.max_landmark_simple_paths,
                        seed=sample_seed,
                        pool_seed=pool_seed,
                        sample_seed=sample_seed,
                        dataset_index=dataset_index,
                        miner_bootstrap_iterations=args.miner_bootstrap_iterations,
                        miner_max_landmarks=args.miner_max_landmarks,
                        miner_selection_threshold=args.miner_selection_threshold,
                        miner_bootstrap_sample_ratio=args.miner_bootstrap_sample_ratio,
                    )
                )
    return conditions


def resolve_n_datasets(args: argparse.Namespace) -> int:
    """解析每个基础条件需要抽样多少个数据集。"""

    n_datasets = args.seeds if args.seeds is not None else args.datasets
    if n_datasets is None:
        n_datasets = 100
    if n_datasets < 1:
        raise ValueError("datasets must be positive")
    return int(n_datasets)


def resolve_n_jobs(n_jobs: int) -> int:
    """把命令行传入的并行数规范化为实际 worker 数。"""

    if n_jobs == 0:
        return max(os.cpu_count() or 1, 1)
    return max(n_jobs, 1)


def run_conditions_from_pools(
    conditions: Sequence[ValidationCondition],
    pools: Mapping[str, Tuple[List[List[int]], Mapping[str, Any]]],
    n_jobs: int,
) -> List[Dict[str, Any]]:
    """在已有轨迹池上运行所有抽样数据集的恢复检验。

    并行只发生在“从轨迹池抽样并挖掘”这一层，轨迹池本身已经提前生成，因此不同 worker 不会重复
    调用生成模型，也更容易保证同一基础条件下结果可比较。
    """

    jobs = [(condition, *pools[condition.base_condition_id]) for condition in conditions]

    if n_jobs == 1:
        return [
            mine_sampled_condition(condition, pool_sequences, pool_summary)
            for condition, pool_sequences, pool_summary in jobs
        ]

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(mine_sampled_condition, condition, pool_sequences, pool_summary)
            for condition, pool_sequences, pool_summary in jobs
        ]
        for index, future in enumerate(futures, start=1):
            result = future.result()
            results.append(result)
            if index % 25 == 0 or index == len(futures):
                print(f"  completed {index}/{len(futures)} sampled datasets")
    return results


def run_conditions(conditions: Sequence[ValidationCondition], n_jobs: int) -> List[Dict[str, Any]]:
    """生成轨迹池并立即运行恢复检验的便捷入口。"""

    pools = generate_sequence_pools(conditions)
    return run_conditions_from_pools(conditions, pools, n_jobs)


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器添加模拟轨迹生成相关参数。"""

    parser.add_argument(
        "--landmark-sets",
        type=str,
        default="curated",
        help='curated, all2, all3, all4, all, or custom like "1,5,9;3,7"',
    )
    parser.add_argument("--n-trials", type=str, default="30", help="comma-separated sampled trial counts")
    parser.add_argument("--pool-trials", type=int, default=60, help="trial pool size generated per base condition")
    parser.add_argument("--mass-ll", type=str, default="0.7,0.9", help="comma-separated LL mass values")
    parser.add_argument(
        "--strategy-profiles",
        type=str,
        default="distance",
        help="coupled strategies for LL/LU/UL; only distance is supported",
    )
    parser.add_argument(
        "--ll-strategy",
        type=str,
        default="distance",
        help="LL allocation strategy; only distance is supported",
    )
    parser.add_argument(
        "--lu-strategy",
        type=str,
        default="distance",
        help="LU allocation strategy; only distance is supported",
    )
    parser.add_argument(
        "--ul-strategy",
        type=str,
        default="distance",
        help="UL allocation strategy; only distance is supported",
    )
    parser.add_argument(
        "--beta-ul",
        type=str,
        default="1.0",
        help="comma-separated UL distance beta values; LL/LU mirror this when omitted",
    )
    parser.add_argument("--beta-ll", type=str, default="", help="optional comma-separated LL distance beta values")
    parser.add_argument("--beta-lu", type=str, default="", help="optional comma-separated LU distance beta values")
    parser.add_argument(
        "--planning-alpha",
        type=float,
        default=1.0,
        help="A* heuristic scale; 0.0 disables heuristic and reduces planning to Dijkstra",
    )
    parser.add_argument(
        "--landmark-path-length-beta",
        type=float,
        default=1.0,
        help="LL simple-path expansion length penalty; larger values favor shorter primitive paths",
    )
    parser.add_argument(
        "--max-landmark-simple-paths",
        type=int,
        default=10000,
        help="max simple paths enumerated for each LL landmark expansion",
    )
    parser.add_argument("--datasets", type=int, default=None, help="sampled datasets per base condition")
    parser.add_argument("--seeds", type=int, default=None, help="legacy alias for --datasets")
    parser.add_argument("--seed-offset", type=int, default=1000, help="first random seed")
    parser.add_argument("--n-jobs", type=int, default=1, help="outer condition parallelism; 0 uses all CPUs")
    parser.add_argument("--quick", action="store_true", help="small smoke-test grid")


def add_miner_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器添加 landmark 推断算法相关参数。"""

    parser.add_argument("--miner-bootstrap-iterations", type=int, default=200, help="bootstrap iterations per dataset")
    parser.add_argument("--miner-bootstrap-sample-ratio", type=float, default=0.8, help="subsampling ratio")
    parser.add_argument("--miner-selection-threshold", type=float, default=0.7, help="selection-rate threshold")
    parser.add_argument("--miner-max-landmarks", type=int, default=4, help="max landmarks selected by miner")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造模拟恢复检验脚本的命令行解析器。"""

    parser = argparse.ArgumentParser(description="Validate landmark inference recovery on synthetic landmark data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RECOVERY_OUTPUT,
        help=f"validation result joblib path (default: {DEFAULT_RECOVERY_OUTPUT})",
    )
    parser.add_argument(
        "--simulated-data-input",
        type=Path,
        default=None,
        help="optional generated synthetic dataset joblib path; when set, validation reuses this file",
    )
    parser.add_argument(
        "--simulated-data-output",
        type=Path,
        default=DEFAULT_SIMULATED_DATA_OUTPUT,
        help=f"generated synthetic dataset joblib path (default: {DEFAULT_SIMULATED_DATA_OUTPUT})",
    )
    add_generation_arguments(parser)
    add_miner_arguments(parser)
    return parser


def apply_quick_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """应用 quick 模式默认值，用较小规模快速检查整条流程。"""

    if not args.quick:
        return args
    if getattr(args, "output", None) == DEFAULT_RECOVERY_OUTPUT:
        args.output = DEFAULT_QUICK_RECOVERY_OUTPUT
    if getattr(args, "simulated_data_output", None) == DEFAULT_SIMULATED_DATA_OUTPUT:
        args.simulated_data_output = DEFAULT_QUICK_SIMULATED_DATA_OUTPUT
    args.landmark_sets = "1,5,9;3,7"
    args.n_trials = "30"
    args.pool_trials = 60
    args.mass_ll = "0.7,0.9"
    args.beta_ul = "1.0"
    args.strategy_profiles = "distance"
    args.beta_ll = args.beta_ll or ""
    args.beta_lu = args.beta_lu or ""
    if args.datasets is None:
        args.datasets = 100
    args.miner_bootstrap_iterations = min(args.miner_bootstrap_iterations, 80)
    return args


def main() -> None:
    """脚本入口：准备模拟数据、运行恢复检验并保存结果 payload。"""

    args = apply_quick_defaults(build_arg_parser().parse_args())
    n_jobs = resolve_n_jobs(args.n_jobs)
    if args.simulated_data_input is not None:
        conditions, pools, simulated_payload = load_simulated_dataset_payload(args.simulated_data_input)
        print(f"Loaded simulated dataset from: {args.simulated_data_input}")
    else:
        conditions = build_conditions(args)
        pools = generate_sequence_pools(conditions)
        simulated_payload = build_simulated_dataset_payload(args, conditions, pools)
        args.simulated_data_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(simulated_payload, args.simulated_data_output)
        print(f"Saved simulated dataset to: {args.simulated_data_output}")

    base_condition_count = len({condition.base_condition_id for condition in conditions})
    datasets_per_base = len(conditions) // max(base_condition_count, 1)
    print(
        f"Running {base_condition_count} base conditions x "
        f"{datasets_per_base} sampled datasets = {len(conditions)} datasets with n_jobs={n_jobs}"
    )
    print(
        "Miner: "
        f"bootstrap={args.miner_bootstrap_iterations}, "
        f"sample_ratio={args.miner_bootstrap_sample_ratio}, "
        f"selection_threshold={args.miner_selection_threshold}, "
        f"max_landmarks={args.miner_max_landmarks}"
    )
    print(
        f"Sampling: pool_trials={args.pool_trials}, sampled_trials={args.n_trials}, "
        f"datasets_per_base={datasets_per_base}"
    )
    print(f"Planning: alpha={args.planning_alpha}")

    results = run_conditions_from_pools(conditions, pools, n_jobs=n_jobs)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task_id": "grid9-with-loop",
        "generator": simulated_payload.get("generator", build_generator_metadata(args)),
        "simulated_dataset": {
            "input_path": str(args.simulated_data_input) if args.simulated_data_input is not None else None,
            "output_path": str(args.simulated_data_output) if args.simulated_data_input is None else None,
            "generated_at": simulated_payload.get("generated_at"),
            "n_sequence_pools": len(simulated_payload.get("sequence_pools", {})),
        },
        "args": vars(args),
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output)

    summary = summarize_results(results)
    print("\nRecovery summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved validation results to: {args.output}")


def summarize_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """从全部条件结果中提取一组命令行可读的总体摘要指标。"""

    if not results:
        return {}
    stable_f1 = [row["metrics_stable"]["f1"] for row in results]
    stable_recall = [row["metrics_stable"]["recall"] for row in results]
    topk_recall = [row["metrics_top_true_k"]["recall"] for row in results]
    return {
        "n_conditions": float(len(results)),
        "mean_stable_f1": float(np.mean(stable_f1)),
        "mean_stable_recall": float(np.mean(stable_recall)),
        "mean_top_true_k_recall": float(np.mean(topk_recall)),
        "mean_true_selection_rate": float(np.mean([row["mean_true_selection_rate"] for row in results])),
        "mean_max_false_selection_rate": float(np.mean([row["max_false_selection_rate"] for row in results])),
    }


if __name__ == "__main__":
    main()
