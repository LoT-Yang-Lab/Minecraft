"""生成模型 TaskGraph 和 Representation 数据层测试。"""

import pytest

from cognitivemap.generative_model import (
    ChunkRepresentation,
    InvalidChunkRepresentationError,
    InvalidPrimitiveRepresentationError,
    PrimitiveToken,
    RepresentationBuilder,
    RepresentationSpec,
    TaskGraph,
)


def test_task_executes_tokens_and_enumerates_shortest_paths(line_task) -> None:
    """验证任务图能执行 primitive token，并枚举最短路径和 simple path。"""

    assert line_task.legal_actions(2) == ["R", "L"]
    trace = line_task.execute_token(1, PrimitiveToken("R"))
    assert trace.actions == ("R",)
    assert trace.states == (1, 2)
    assert line_task.shortest_distance(1, 3) == 2.0
    assert [path.states for path in line_task.all_shortest_paths(1, 3)] == [(1, 2, 3)]
    assert [path.states for path in line_task.all_simple_paths(1, 3)] == [(1, 2, 3)]
    assert line_task.all_shortest_paths(2, 2)[0].actions == ()


def test_task_enumerates_simple_paths_beyond_shortest() -> None:
    """验证 simple path 枚举会保留非最短但不重复状态的路径。"""

    task = TaskGraph.from_transitions(
        "simple-paths",
        {
            1: {"direct": 3, "via": 2},
            2: {"finish": 3},
            3: {"back": 1},
        },
    )

    assert [path.states for path in task.all_shortest_paths(1, 3)] == [(1, 3)]
    assert [path.states for path in task.all_simple_paths(1, 3)] == [(1, 3), (1, 2, 3)]


def test_task_from_transitions_derives_task_fields() -> None:
    """验证 from_transitions 能派生状态、动作、合法动作和代价。"""

    task = TaskGraph.from_transitions(
        "derived",
        {
            "s1": {"go": "s2"},
            "s2": {"loop": "s2"},
        },
        cost_overrides={"s2": {"loop": 2.5}},
    )

    assert task.states == ("s1", "s2")
    assert task.actions == ("go", "loop")
    assert task.legal_actions("s1") == ["go"]
    assert task.action_cost("s2", "loop") == 2.5


def test_primitive_representation_requires_exact_legal_action_rows(line_task) -> None:
    """验证 primitive 权重行必须严格覆盖合法动作并归一化。"""

    with pytest.raises(InvalidPrimitiveRepresentationError):
        bad = {
            1: {"R": 1.0},
            2: {"L": 0.5, "R": 0.4},
            3: {"L": 1.0},
        }
        from cognitivemap.generative_model import PrimitiveRepresentation

        PrimitiveRepresentation("bad", bad).validate(line_task)


def test_chunk_representation_filters_executable_chunks(branch_task, chunk_branch_representation) -> None:
    """验证 chunk 表征只把当前 state 可完整执行的 chunk 放入候选集。"""

    tokens = chunk_branch_representation.tokens_at(1, branch_task)
    assert [token.token_id for token in tokens] == ["primitive:A", "primitive:B", "chunk:g_ac"]

    with pytest.raises(InvalidChunkRepresentationError):
        ChunkRepresentation(
            representation_id="bad-chunk",
            chunk_inventory={"g_ac": ("A", "C")},
            token_weights_by_state={
                1: {"primitive:A": 0.5, "primitive:B": 0.5},
                2: {"primitive:C": 1.0},
                3: {"primitive:D": 1.0},
                4: {"primitive:E": 0.5, "primitive:F": 0.5},
            },
        ).validate(branch_task)


def test_representation_builder_builds_uniform_representations(branch_task) -> None:
    """验证 RepresentationBuilder 能构造 uniform primitive/chunk 表征。"""

    primitive = RepresentationBuilder.build_primitive(branch_task, "prim-uniform")
    assert primitive.weight(1, "A") == pytest.approx(0.5)

    spec = RepresentationSpec(
        family="chunk",
        representation_id="chunk-uniform",
        parameters={"chunk_inventory": {"g_ac": ("A", "C")}},
        defaults={"policy": "uniform"},
    )
    chunk = RepresentationBuilder.build(branch_task, spec)

    assert [token.token_id for token in chunk.tokens_at(1, branch_task)] == [
        "primitive:A",
        "primitive:B",
        "chunk:g_ac",
    ]
    assert chunk.weight(1, "chunk:g_ac") == pytest.approx(1 / 3)
