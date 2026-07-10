"""生成模型 kernel、认知图、规划和展开行为测试。"""

import json

import numpy as np
import pytest

from cognitivemap.generative_model import (
    AStarStatus,
    CognitiveNavigationAgent,
    Family,
    LandmarkRepresentation,
    PlanningConfig,
    TaskGraph,
    TrialSpec,
    build_transition_kernel,
    expand_cognitive_path,
    plan_cognitive_path,
)


def test_primitive_kernel_sums_actions_by_endpoint(branch_task, primitive_branch_representation) -> None:
    """验证 primitive kernel 会把同 endpoint 的 action 概率相加。"""

    kernel, support = build_transition_kernel(branch_task, primitive_branch_representation)

    assert kernel.family == Family.PRIMITIVE
    assert kernel.probability(1, 2) == pytest.approx(0.4)
    assert kernel.probability(1, 3) == pytest.approx(0.6)
    assert kernel.probability(4, 1) == pytest.approx(1.0)
    assert len(support.entries_for(4, 1)) == 2


def test_chunk_kernel_uses_executable_tokens_and_endpoint_aggregation(branch_task, chunk_branch_representation) -> None:
    """验证 chunk kernel 使用可执行 token，并按 endpoint 聚合概率。"""

    kernel, support = build_transition_kernel(branch_task, chunk_branch_representation)

    assert kernel.family == Family.CHUNK
    assert kernel.probability(1, 2) == pytest.approx(0.2)
    assert kernel.probability(1, 3) == pytest.approx(0.3)
    assert kernel.probability(1, 4) == pytest.approx(0.5)
    assert [entry.item_id for entry in support.entries_for(1, 4)] == ["chunk:g_ac"]


def test_landmark_kernel_uses_cpd_blocks(line_task, landmark_line_representation) -> None:
    """验证 landmark kernel 直接使用 LL/LU/UL CPD 概率块。"""

    kernel, support = build_transition_kernel(line_task, landmark_line_representation)

    assert kernel.family == Family.LANDMARK
    assert kernel.probability(1, 3) == pytest.approx(0.5)
    assert kernel.probability(2, 2) == pytest.approx(0.0)
    assert support.entries_for(1, 3)[0].cpd_block.value == "LL"


def test_cognitive_graph_distance_embedding_and_astar(line_task, primitive_line_representation) -> None:
    """验证 kernel artifact 能生成认知图、距离、embedding 和 A* 路径。"""

    agent = CognitiveNavigationAgent(primitive_line_representation, PlanningConfig(alpha=0.0))
    artifacts = agent.build_artifacts(line_task)
    graph = artifacts.cognitive_graph

    assert graph.edge_lookup[(1, 2)].cost == pytest.approx(-np.log(1.0))
    assert not hasattr(graph.edge_lookup[(1, 2)], "support_refs")
    assert artifacts.directed_distance.matrix[0][2] == pytest.approx(-np.log(1.0) - np.log(0.75))
    assert artifacts.embedding.status == "ready"

    astar = plan_cognitive_path(graph, artifacts.embedding, 1, 3, PlanningConfig(alpha=0.0))
    assert astar.status == AStarStatus.FOUND
    assert astar.cognitive_path == (1, 2, 3)


def test_expansion_records_candidates_and_samples_token(branch_task, chunk_branch_representation) -> None:
    """验证 edge 展开会记录候选集并按随机源采样 token。"""

    _, support = build_transition_kernel(branch_task, chunk_branch_representation)
    expansion = expand_cognitive_path(
        branch_task,
        chunk_branch_representation,
        support,
        (1, 4),
        4,
        np.random.default_rng(0),
        PlanningConfig(),
    )

    assert expansion.actions == ("A", "C")
    assert expansion.states == (1, 2, 4)
    assert expansion.edge_expansions[0].sampled_item_id == "chunk:g_ac"
    assert expansion.termination.primitive_step_index == 1


def test_landmark_expansion_samples_shortest_primitive_path(line_task, landmark_line_representation) -> None:
    """验证非 LL landmark edge 使用 primitive 最短路径展开。"""

    _, support = build_transition_kernel(line_task, landmark_line_representation)
    expansion = expand_cognitive_path(
        line_task,
        landmark_line_representation,
        support,
        (1, 3),
        3,
        np.random.default_rng(0),
        PlanningConfig(),
    )

    assert expansion.actions == ("R", "R")
    assert expansion.states == (1, 2, 3)
    assert expansion.edge_expansions[0].candidates[0].item_id == "path:0"


def test_landmark_ll_expansion_uses_weighted_simple_paths() -> None:
    """验证 LL landmark edge 枚举 simple paths，并按路径长度加权。"""

    task = TaskGraph.from_transitions(
        "ll-simple-paths",
        {
            1: {"direct": 3, "via": 2},
            2: {"finish": 3},
            3: {"back": 1},
        },
    )
    representation = LandmarkRepresentation(
        representation_id="ll-simple",
        landmarks=(1, 3),
        p_ll={
            1: {1: 0.0, 3: 0.5},
            3: {1: 0.5, 3: 0.0},
        },
        p_lu={
            1: {2: 0.5},
            3: {2: 0.5},
        },
        p_ul={
            2: {1: 0.5, 3: 0.5},
        },
    )
    _, support = build_transition_kernel(task, representation)

    expansion = expand_cognitive_path(
        task,
        representation,
        support,
        (1, 3),
        3,
        np.random.default_rng(0),
        PlanningConfig(landmark_path_length_beta=1.0),
    )
    trace = expansion.edge_expansions[0]

    assert [candidate.states for candidate in trace.candidates] == [(1, 3), (1, 2, 3)]
    assert trace.candidate_probabilities[0] > trace.candidate_probabilities[1]


def test_agent_run_trial_outputs_artifacts_and_json(line_task, primitive_line_representation) -> None:
    """验证 agent 单 trial 输出可序列化结果和 model artifacts。"""

    agent = CognitiveNavigationAgent(primitive_line_representation, PlanningConfig(alpha=0.0))
    result = agent.run_trial(line_task, TrialSpec("trial-1", 1, 3, "primitive", random_seed=7))
    output = agent.to_dict(result)

    assert output["trials"][0]["status"] == "reached_goal"
    assert output["trials"][0]["astar"]["cognitive_path"] == [1, 2, 3]
    assert output["trials"][0]["actions"] == ["R", "R"]
    assert "transition_kernel" in output["model_artifacts"]
    json.dumps(output)


def test_embedding_error_is_reported_when_kernel_graph_is_not_strongly_connected(line_task) -> None:
    """验证认知图不可强连通时会返回 embedding_error 结果。"""

    from cognitivemap.generative_model import PrimitiveRepresentation

    representation = PrimitiveRepresentation(
        representation_id="one-way",
        action_weights_by_state={
            1: {"R": 1.0},
            2: {"L": 0.0, "R": 1.0},
            3: {"L": 1.0},
        },
    )
    agent = CognitiveNavigationAgent(representation, PlanningConfig(alpha=0.0))
    result = agent.run_trial(line_task, TrialSpec("trial-error", 1, 3, "primitive"))

    assert result.status.value == "embedding_error"
    assert result.error is not None
    assert result.error.stage.value == "embedding"
