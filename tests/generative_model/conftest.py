"""生成模型测试共享 fixture。

这些 fixture 构造小型 line/branch 任务图和三类表征，用于在单元测试中精确验证 kernel、
认知图、A* 规划和 primitive 展开的行为。
"""

import pytest

from cognitivemap.generative_model import (
    ChunkRepresentation,
    LandmarkRepresentation,
    PrimitiveRepresentation,
    TaskGraph,
)


@pytest.fixture
def line_task() -> TaskGraph:
    """返回一个 1-2-3 线性任务图。"""

    return TaskGraph.from_transitions(
        task_id="line-3",
        transitions={
            1: {"R": 2},
            2: {"L": 1, "R": 3},
            3: {"L": 2},
        },
    )


@pytest.fixture
def branch_task() -> TaskGraph:
    """返回一个带两条分支路径的四状态任务图。"""

    return TaskGraph.from_transitions(
        task_id="branch-4",
        transitions={
            1: {"A": 2, "B": 3},
            2: {"C": 4},
            3: {"D": 4},
            4: {"E": 1, "F": 1},
        },
    )


@pytest.fixture
def primitive_line_representation() -> PrimitiveRepresentation:
    """返回 line_task 上的 primitive action 权重。"""

    return PrimitiveRepresentation(
        representation_id="prim-line",
        action_weights_by_state={
            1: {"R": 1.0},
            2: {"L": 0.25, "R": 0.75},
            3: {"L": 1.0},
        },
    )


@pytest.fixture
def primitive_branch_representation() -> PrimitiveRepresentation:
    """返回 branch_task 上的 primitive action 权重。"""

    return PrimitiveRepresentation(
        representation_id="prim-branch",
        action_weights_by_state={
            1: {"A": 0.4, "B": 0.6},
            2: {"C": 1.0},
            3: {"D": 1.0},
            4: {"E": 0.5, "F": 0.5},
        },
    )


@pytest.fixture
def chunk_branch_representation() -> ChunkRepresentation:
    """返回 branch_task 上包含一个可执行和一个不可执行 chunk 的表征。"""

    return ChunkRepresentation(
        representation_id="chunk-branch",
        chunk_inventory={
            "g_ac": ("A", "C"),
            "g_bad": ("A", "F"),
        },
        token_weights_by_state={
            1: {
                "primitive:A": 0.2,
                "primitive:B": 0.3,
                "chunk:g_ac": 0.5,
            },
            2: {
                "primitive:C": 1.0,
            },
            3: {
                "primitive:D": 1.0,
            },
            4: {
                "primitive:E": 0.4,
                "primitive:F": 0.6,
            },
        },
    )


@pytest.fixture
def landmark_line_representation() -> LandmarkRepresentation:
    """返回 line_task 上以 1 和 3 为 landmark 的 CPD 表征。"""

    return LandmarkRepresentation(
        representation_id="land-line",
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
