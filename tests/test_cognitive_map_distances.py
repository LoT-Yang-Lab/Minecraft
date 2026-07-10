"""认知地图距离函数的行为回归测试。"""

import numpy as np
import pytest

from cognitivemap.map_estimation import (
    Trial,
    compute_action_js_distance,
    compute_lops_distance,
    compute_sr_distance,
    compute_transition_similarity_distance,
)


def test_compute_lops_distance_uses_dijkstra_lengths_without_swallowing_edges() -> None:
    """验证 LoPS 距离使用 Dijkstra 路径长度并保留有效边。"""

    trials = [
        Trial(state_sequence=[1, 2, 3], action_sequence=["a", "b"]),
        Trial(state_sequence=[1, 3], action_sequence=["c"]),
        Trial(state_sequence=[2, 1], action_sequence=["d"]),
    ]

    distance_matrix, state_labels = compute_lops_distance(trials)

    expected_edge_weight = np.sqrt(-np.log(0.5))
    assert state_labels == [1, 2, 3]
    assert distance_matrix[0, 1] == pytest.approx(expected_edge_weight)
    assert distance_matrix[0, 2] == pytest.approx(expected_edge_weight)
    assert distance_matrix[1, 2] == pytest.approx(expected_edge_weight)
    assert np.allclose(distance_matrix, distance_matrix.T)
    assert np.allclose(np.diag(distance_matrix), 0.0)


def test_distance_methods_apply_state_filter_to_trial_events() -> None:
    """验证所有距离方法都会先应用 state_filter 再计算距离。"""

    trials = [
        Trial(state_sequence=[1, 2, 3], action_sequence=["a", "b"]),
        Trial(state_sequence=[1, 3], action_sequence=["c"]),
        Trial(state_sequence=[2, 1], action_sequence=["d"]),
    ]

    def filter_out_state_3(states: list[int]) -> list[int]:
        """过滤状态 3，用于测试 state_filter 管线。"""

        return [state for state in states if state != 3]

    distance_functions = (
        compute_sr_distance,
        compute_lops_distance,
        compute_action_js_distance,
        compute_transition_similarity_distance,
    )

    for distance_function in distance_functions:
        distance_matrix, state_labels = distance_function(trials, state_filter=filter_out_state_3)

        assert state_labels == [1, 2]
        assert distance_matrix.shape == (2, 2)
        assert np.allclose(np.diag(distance_matrix), 0.0)
