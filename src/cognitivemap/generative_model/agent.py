"""认知导航模型的统一运行入口。

``CognitiveNavigationAgent`` 绑定一个已构造好的 representation 和一份 PlanningConfig。
TaskGraph 是外部环境，每次 build/plan/run 时显式传入；agent 只缓存最近一次
task + representation 组合生成的 ModelArtifacts。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cognitivemap.generative_model.cognitive_graph import (
    build_cognitive_graph,
    compute_classical_mds,
    compute_directed_distances,
    symmetrize_distances,
)
from cognitivemap.generative_model.planning import expand_cognitive_path, plan_cognitive_path
from cognitivemap.generative_model.representations import Representation
from cognitivemap.generative_model.task import TaskGraph
from cognitivemap.generative_model.transition_kernel import build_transition_kernel
from cognitivemap.generative_model.types import (
    AStarResult,
    AStarStatus,
    CognitiveNavigationError,
    EmbeddingError,
    ErrorDetail,
    ErrorStage,
    ExpansionError,
    ModelArtifacts,
    PlanningConfig,
    PlanningError,
    RunMetadata,
    RunResult,
    RunStatus,
    TrialSpec,
    to_json,
)


@dataclass
class CognitiveNavigationAgent:
    """持有单一认知表征和运行配置的导航主体。

    第一版 agent 的职责是编排主链路：kernel -> cognitive graph -> distance ->
    embedding -> A* -> expansion。它不构造 representation，也不从行为数据中学习参数。
    """

    representation: Representation
    config: PlanningConfig = field(default_factory=PlanningConfig)
    cached_task_id: str | None = field(init=False, default=None)
    cached_artifacts: ModelArtifacts | None = field(init=False, default=None)

    def build_artifacts(self, task: TaskGraph, force: bool = False) -> ModelArtifacts:
        """为给定 task 构造并缓存可复用 artifact。

        若 ``force=False`` 且 task_id 与缓存一致，直接复用缓存。
        如果调用方修改了同名 task 的内部内容，应传入 ``force=True`` 或创建新 agent。
        """

        if self.cached_artifacts is not None and self.cached_task_id == task.task_id and not force:
            return self.cached_artifacts

        # kernel/support 是并列 artifact；后续 graph 和 distance 只依赖 kernel。
        kernel, support = build_transition_kernel(task, self.representation, self.config)
        graph = build_cognitive_graph(kernel, self.config)
        directed = compute_directed_distances(graph)
        symmetric = symmetrize_distances(directed)
        embedding = compute_classical_mds(symmetric, self.config)
        artifacts = ModelArtifacts(
            task_id=task.task_id,
            representation_id=self.representation.representation_id,
            family=self.representation.family,
            transition_kernel=kernel,
            kernel_support=support,
            cognitive_graph=graph,
            directed_distance=directed,
            symmetric_distance=symmetric,
            embedding=embedding,
        )
        self.cached_task_id = task.task_id
        self.cached_artifacts = artifacts
        return artifacts

    def plan(self, task: TaskGraph, start_state: Any, goal_state: Any) -> AStarResult:
        """只在 cognitive graph 上规划路径，不展开 primitive 轨迹。"""

        artifacts = self.build_artifacts(task)
        _validate_start_goal(artifacts, start_state, goal_state)
        return plan_cognitive_path(
            artifacts.cognitive_graph,
            artifacts.embedding,
            start_state,
            goal_state,
            self.config,
        )

    def run_trial(self, task: TaskGraph, trial: TrialSpec) -> RunResult:
        """运行单个 trial：校验 family、规划 cognitive path，并执行 edge expansion。"""

        metadata = RunMetadata(seed=trial.random_seed, config=self.config)
        try:
            _validate_trial_family(self.representation, trial)
            artifacts = self.build_artifacts(task)
            astar = self.plan(task, trial.start_state, trial.goal_state)
            if astar.status == AStarStatus.NO_PATH:
                return _run_result(
                    status=RunStatus.NO_PATH,
                    task=task,
                    representation=self.representation,
                    trial=trial,
                    artifacts=artifacts,
                    astar=astar,
                    expansion=None,
                    metadata=metadata,
                    error=None,
                )

            # trial seed 是唯一随机源，保证相同候选顺序和概率下可复现。
            rng = np.random.default_rng(trial.random_seed)
            expansion = expand_cognitive_path(
                task,
                self.representation,
                artifacts.kernel_support,
                astar.cognitive_path,
                trial.goal_state,
                rng,
                self.config,
            )
            return _run_result(
                status=RunStatus.REACHED_GOAL,
                task=task,
                representation=self.representation,
                trial=trial,
                artifacts=artifacts,
                astar=astar,
                expansion=expansion,
                metadata=metadata,
                error=None,
            )
        except CognitiveNavigationError as exc:
            return _error_result(task, self.representation, trial, metadata, exc)

    def run_batch(self, task: TaskGraph, trials: list[TrialSpec]) -> list[RunResult]:
        """批量运行 trial；同一个 agent 会复用最近一次 artifact 缓存。"""

        return [self.run_trial(task, trial) for trial in trials]

    def to_dict(self, result: RunResult | list[RunResult]) -> dict[str, Any]:
        """把单个或多个 RunResult 转换为稳定 JSON-compatible dict。"""

        results = result if isinstance(result, list) else [result]
        return _results_to_dict(results)

    def to_json(self, result: RunResult | list[RunResult], *, indent: int | None = 2) -> str:
        """把单个或多个 RunResult 序列化为 JSON 字符串。"""

        return to_json(self.to_dict(result), indent=indent)


def _validate_trial_family(representation: Representation, trial: TrialSpec) -> None:
    """校验 trial.family 是否与 agent 的 representation.family 一致。"""

    if representation.family != trial.family:
        raise PlanningError(
            "trial family does not match representation family",
            context={"trial_family": trial.family.value, "representation_family": representation.family.value},
        )


def _validate_start_goal(artifacts: ModelArtifacts, start_state: Any, goal_state: Any) -> None:
    """校验 start/goal 是否出现在 cognitive graph 中。"""

    states = set(artifacts.cognitive_graph.states)
    if start_state not in states:
        raise PlanningError("trial start state is not in graph", context={"start_state": start_state})
    if goal_state not in states:
        raise PlanningError("trial goal state is not in graph", context={"goal_state": goal_state})


def _error_result(
    task: TaskGraph,
    representation: Representation,
    trial: TrialSpec,
    metadata: RunMetadata,
    error: CognitiveNavigationError,
) -> RunResult:
    """把可识别错误转换为失败 RunResult。"""

    if isinstance(error, EmbeddingError):
        status = RunStatus.EMBEDDING_ERROR
    elif isinstance(error, ExpansionError):
        status = RunStatus.EXPANSION_ERROR
    elif isinstance(error, PlanningError):
        status = RunStatus.VALIDATION_ERROR
    else:
        status = RunStatus.VALIDATION_ERROR
    return _run_result(
        status=status,
        task=task,
        representation=representation,
        trial=trial,
        artifacts=None,
        astar=None,
        expansion=None,
        metadata=metadata,
        error=error.to_detail(),
    )


def _run_result(
    *,
    status: RunStatus,
    task: TaskGraph,
    representation: Representation,
    trial: TrialSpec,
    artifacts: ModelArtifacts | None,
    astar: Any,
    expansion: Any,
    metadata: RunMetadata,
    error: ErrorDetail | None,
) -> RunResult:
    """组装 RunResult，减少正常路径和错误路径的重复代码。"""

    return RunResult(
        status=status,
        task_id=task.task_id,
        representation_id=representation.representation_id,
        family=representation.family,
        trial=trial,
        artifacts=artifacts,
        astar=astar,
        expansion=expansion,
        metadata=metadata,
        error=error,
    )


def _results_to_dict(results: list[RunResult]) -> dict[str, Any]:
    """生成文档约定的顶层序列化结构。"""

    if not results:
        return {"schema_version": 1, "trials": []}
    first = results[0]
    return {
        "schema_version": 1,
        "task_id": first.task_id,
        "representation_id": first.representation_id,
        "family": first.family.value,
        "config": _config_to_dict(first.metadata.config),
        "model_artifacts": _artifacts_to_dict(first.artifacts),
        "trials": [_trial_result_to_dict(result) for result in results],
    }


def _config_to_dict(config: PlanningConfig) -> dict[str, Any]:
    """序列化 PlanningConfig，仅包含第一版设计中的配置字段。"""

    return {
        "alpha": config.alpha,
        "probability_epsilon": config.probability_epsilon,
        "row_sum_tolerance": config.row_sum_tolerance,
        "tie_breaking": config.tie_breaking,
        "max_landmark_shortest_paths": config.max_landmark_shortest_paths,
        "max_landmark_simple_paths": config.max_landmark_simple_paths,
        "landmark_path_length_beta": config.landmark_path_length_beta,
        "mds_dim": config.mds_dim,
    }


def _artifacts_to_dict(artifacts: ModelArtifacts | None) -> dict[str, Any] | None:
    """序列化 ModelArtifacts。

    cognitive edge 中不输出 support_refs；概率来源通过 kernel_support 并列输出。
    """

    if artifacts is None:
        return None
    kernel = artifacts.transition_kernel
    graph = artifacts.cognitive_graph
    return {
        "transition_kernel": {
            "family": kernel.family.value,
            "states": list(kernel.states),
            "matrix": [list(row) for row in kernel.matrix],
            "row_index": {str(state): index for state, index in kernel.row_index.items()},
        },
        "kernel_support": {
            "entries_by_edge": {
                f"{source}->{target}": [_support_entry_to_dict(entry) for entry in entries]
                for (source, target), entries in artifacts.kernel_support.entries_by_edge.items()
            }
        },
        "cognitive_graph": {
            "family": graph.family.value,
            "states": list(graph.states),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "probability": edge.probability,
                    "cost": edge.cost,
                }
                for edge in graph.edges
            ],
            "successors": {str(state): list(values) for state, values in graph.successors.items()},
        },
        "directed_distance": {
            "states": list(artifacts.directed_distance.states),
            "matrix": [list(row) for row in artifacts.directed_distance.matrix],
            "unreachable_pairs": [list(pair) for pair in artifacts.directed_distance.unreachable_pairs],
        },
        "symmetric_distance": {
            "states": list(artifacts.symmetric_distance.states),
            "matrix": [list(row) for row in artifacts.symmetric_distance.matrix],
        },
        "embedding": {
            "states": list(artifacts.embedding.states),
            "coordinates": {str(state): list(coord) for state, coord in artifacts.embedding.coordinates.items()},
            "eigenvalues": list(artifacts.embedding.eigenvalues),
            "status": artifacts.embedding.status,
        },
    }


def _support_entry_to_dict(entry: Any) -> dict[str, Any]:
    """序列化单条 KernelSupportEntry。"""

    return {
        "source": entry.source,
        "target": entry.target,
        "kind": entry.kind.value,
        "item_id": entry.item_id,
        "probability": entry.probability,
        "endpoint": entry.endpoint,
        "action_sequence": list(entry.action_sequence),
        "state_sequence": list(entry.state_sequence),
        "cpd_block": entry.cpd_block.value if entry.cpd_block else None,
    }


def _trial_result_to_dict(result: RunResult) -> dict[str, Any]:
    """序列化单个 trial 的运行结果。"""

    expansion = result.expansion
    return {
        "trial_id": result.trial.trial_id,
        "start_state": result.trial.start_state,
        "goal_state": result.trial.goal_state,
        "random_seed": result.metadata.seed,
        "status": result.status.value,
        "cognitive_path": list(result.astar.cognitive_path) if result.astar else [],
        "astar": _astar_to_dict(result.astar),
        "edge_expansions": _edge_expansions_to_list(expansion),
        "actions": list(expansion.actions) if expansion else [],
        "states": list(expansion.states) if expansion else [],
        "termination": _termination_to_dict(expansion),
        "error": _error_to_dict(result.error),
    }


def _astar_to_dict(astar: Any) -> dict[str, Any] | None:
    """序列化 AStarResult。"""

    if astar is None:
        return None
    return {
        "status": astar.status.value,
        "start_state": astar.start_state,
        "goal_state": astar.goal_state,
        "cognitive_path": list(astar.cognitive_path),
        "path_cost": astar.path_cost,
        "expanded_nodes": list(astar.expanded_nodes),
        "visited_count": astar.visited_count,
        "tie_breaking": astar.tie_breaking,
    }


def _edge_expansions_to_list(expansion: Any) -> list[dict[str, Any]]:
    """序列化 expansion 中每条 cognitive edge 的展开记录。"""

    if expansion is None:
        return []
    return [
        {
            "edge_index": trace.edge_index,
            "source": trace.source,
            "target": trace.target,
            "family": trace.family.value,
            "candidates": [
                {
                    "item_id": candidate.item_id,
                    "probability": candidate.probability,
                    "actions": list(candidate.actions),
                    "states": list(candidate.states),
                }
                for candidate in trace.candidates
            ],
            "candidate_probabilities": list(trace.candidate_probabilities),
            "sampled_index": trace.sampled_index,
            "sampled_item_id": trace.sampled_item_id,
            "actions": list(trace.actions),
            "states": list(trace.states),
            "reached_goal_at_step": trace.reached_goal_at_step,
        }
        for trace in expansion.edge_expansions
    ]


def _termination_to_dict(expansion: Any) -> dict[str, Any] | None:
    """序列化 primitive 展开的终止信息。"""

    if expansion is None:
        return None
    return {
        "reason": expansion.termination.reason,
        "edge_index": expansion.termination.edge_index,
        "primitive_step_index": expansion.termination.primitive_step_index,
    }


def _error_to_dict(error: ErrorDetail | None) -> dict[str, Any] | None:
    """序列化错误详情。"""

    if error is None:
        return None
    return {
        "code": error.code,
        "message": error.message,
        "stage": error.stage.value if isinstance(error.stage, ErrorStage) else error.stage,
        "object_id": error.object_id,
        "context": dict(error.context),
    }
