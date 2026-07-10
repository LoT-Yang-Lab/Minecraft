"""landmark 推断函数的基础行为测试。"""

from cognitivemap.inference.miner_landmark import (
    LandmarkMiningConfig,
    mine_landmarks,
    state_sequences_from_transition_rows,
)


def test_landmark_miner_ranks_common_bridge_state_first() -> None:
    """验证共同桥接状态在 landmark candidate 中排名最高。"""

    sequences = [
        [1, 2, 3],
        [1, 2, 4],
        [5, 2, 3],
        [5, 2, 4],
        [1, 2, 3],
    ]
    config = LandmarkMiningConfig(
        features=("coverage", "path_commonality", "betweenness"),
        max_landmarks=2,
        bootstrap_iterations=0,
        n_jobs=1,
    )

    result = mine_landmarks(sequences, config)

    assert result["top_landmarks"][0] == 2
    assert result["state_scores"][2]["score"] > result["state_scores"][1]["score"]
    assert result["state_scores"][2]["path_commonality"] > result["state_scores"][3]["path_commonality"]


def test_state_sequences_from_transition_rows_keeps_only_valid_steps() -> None:
    """验证行式 transition 记录只用 valid step 恢复状态序列。"""

    rows = [
        {"trial_id": "t1", "step": 1, "state": 1, "next_state": 1, "valid": False},
        {"trial_id": "t1", "step": 2, "state": 1, "next_state": 2, "valid": True},
        {"trial_id": "t1", "step": 3, "state": 2, "next_state": 3, "valid": True},
        {"trial_id": "t2", "step": 1, "state": 4, "next_state": 5, "valid": True},
    ]

    assert state_sequences_from_transition_rows(rows) == [[1, 2, 3], [4, 5]]
