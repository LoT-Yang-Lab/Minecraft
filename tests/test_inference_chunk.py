"""chunk 推断与 chunk parser 的行为回归测试。"""

import math

import pytest

from cognitivemap.inference.miner_chunk import TemporalPatternMiner
from cognitivemap.inference.miner_chunk_state_conditioned import DiscreteStateTemporalPatternMiner
from cognitivemap.inference.parser_chunk import parse_with_chunks


def test_parse_with_chunks_prefers_longest_match_and_returns_start_indices() -> None:
    """验证 chunk parser 使用最长匹配，并返回原序列起始下标。"""

    parsed, indices = parse_with_chunks(
        ["A", "B", "C", "A", "B"],
        ["A", "B", "A-B", "A-B-C"],
        need_index=True,
    )

    assert parsed == ["A-B-C", "A-B"]
    assert indices == [0, 3]


def test_temporal_pattern_miner_keeps_simple_repeated_pair_behavior() -> None:
    """验证简单重复二元模式会被合成为 chunk。"""

    miner = TemporalPatternMiner(
        max_iterations=3,
        js_threshold=0.0,
        convergence_window=5,
        min_joint_probability=0.0,
        min_log_bayes_factor=0.0,
        score_ratio_threshold=0.0,
    )

    result = miner.mine_patterns(
        ["A", "B", "A", "B", "A", "B"],
        ess=2,
        constr_func=lambda parent, child: True,
    )

    assert result["vocab"] == ["A-B"]
    assert result["sequence"] == ["A-B", "A-B", "A-B"]
    assert result["state_sequence"] is None
    assert result["components"]["A-B"] == ["A", "B"]
    assert result["components"]["B-A"] == ["B", "A"]
    assert result["js_history"] == pytest.approx([0.8325546111576977])


def test_temporal_pattern_miner_uses_log_bayes_factor_threshold() -> None:
    """验证候选 chunk 会受到总 log Bayes factor 阈值约束。"""

    sequence = ["A", "B", "A", "B", "A", "B"]
    miner = TemporalPatternMiner(
        min_joint_probability=0.0,
        min_log_bayes_factor=0.0,
        score_ratio_threshold=0.0,
    )
    data_child, data_parent, state_parent, elements = miner._sequence_to_temporal_onehot(sequence)
    del state_parent, elements
    element_probability = miner._get_element_frequency(sequence)
    data_p, score_parent, parent_num_states = miner._build_parent_scoring_context("A", data_parent, None, None)

    candidate = miner._score_candidate_pair(
        parent="A",
        child="B",
        data_p=data_p,
        data_child=data_child,
        score_parent=score_parent,
        parent_num_states=parent_num_states,
        state_bdeu_data=None,
        state_bdeu_num_states=None,
        element_probability=element_probability,
        scoring_token_count=len(sequence),
        ess=2,
    )

    assert candidate is not None
    assert candidate.log_bayes_factor > 0.0
    assert candidate.score == pytest.approx(candidate.log_bayes_factor / candidate.sample_count)

    strict_miner = TemporalPatternMiner(
        min_joint_probability=0.0,
        min_log_bayes_factor=candidate.log_bayes_factor + 1e-9,
        score_ratio_threshold=0.0,
    )
    strict_candidate = strict_miner._score_candidate_pair(
        parent="A",
        child="B",
        data_p=data_p,
        data_child=data_child,
        score_parent=score_parent,
        parent_num_states=parent_num_states,
        state_bdeu_data=None,
        state_bdeu_num_states=None,
        element_probability=element_probability,
        scoring_token_count=len(sequence),
        ess=2,
    )

    assert strict_candidate is None


def test_candidate_selection_score_is_normalized_by_sample_count() -> None:
    """验证候选排序分数按有效样本数归一化。"""

    sequence = ["A", "B", "A", "B", "A", "B"]
    miner = TemporalPatternMiner(
        min_joint_probability=0.0,
        min_log_bayes_factor=0.0,
        score_ratio_threshold=0.0,
    )
    data_child, data_parent, state_parent, elements = miner._sequence_to_temporal_onehot(sequence)
    del state_parent, elements
    element_probability = miner._get_element_frequency(sequence)
    data_p, score_parent, parent_num_states = miner._build_parent_scoring_context("A", data_parent, None, None)

    candidate = miner._score_candidate_pair(
        parent="A",
        child="B",
        data_p=data_p,
        data_child=data_child,
        score_parent=score_parent,
        parent_num_states=parent_num_states,
        state_bdeu_data=None,
        state_bdeu_num_states=None,
        element_probability=element_probability,
        scoring_token_count=len(sequence),
        ess=2,
    )

    assert candidate is not None
    assert candidate.sample_count == len(sequence) - 1
    assert candidate.score < candidate.log_bayes_factor


def test_temporal_pattern_miner_excludes_structural_token_from_frequency_and_pair_rows() -> None:
    """验证结构 token 不参与频率统计和相邻 pair 打分。"""

    sequence = ["A", "B", "S", "A", "B"]
    state_sequence = [1, 2, -1, 1, 2]
    miner = TemporalPatternMiner(structural_tokens=("S",))

    data_child, data_parent, state_parent, _ = miner._sequence_to_temporal_onehot(sequence, state_sequence)
    data_child, data_parent, state_parent = miner._filter_scoring_pairs(
        data_child,
        data_parent,
        state_parent,
        sequence,
    )

    assert miner._get_element_frequency(sequence) == {"A": 0.5, "B": 0.5}
    assert len(data_child) == 2
    assert data_parent["A"].tolist() == [1, 1]
    assert data_child["B"].tolist() == [1, 1]
    assert state_parent["state_0"].tolist() == [1, 1]


def test_temporal_pattern_miner_keeps_separator_as_boundary_without_scoring_it() -> None:
    """验证 trial 分隔符能作为边界保留，但不会进入 chunk 词表。"""

    miner = TemporalPatternMiner(
        max_iterations=1,
        js_threshold=0.0,
        convergence_window=5,
        min_joint_probability=0.0,
        min_log_bayes_factor=-1000.0,
        score_ratio_threshold=0.0,
        structural_tokens=("S",),
    )

    result = miner.mine_patterns(
        ["A", "B", "S", "C", "D"],
        ess=2,
        constr_func=lambda parent, child: "S" not in parent and "S" not in child,
        state_sequence=[1, 2, -1, 3, 4],
    )

    assert "S" not in result["vocab"]
    assert result["sequence"] == ["A-B", "S", "C-D"]
    assert result["state_sequence"] == [1, -1, 3]


def test_choice_chunks_uses_log_space_bayes_factor_ratio() -> None:
    """验证多候选筛选在 log 空间比较 Bayes factor 比值。"""

    miner = TemporalPatternMiner(score_ratio_threshold=0.85)
    max_log_bf = 10.0
    scores = [
        max_log_bf,
        max_log_bf + math.log(0.9),
        max_log_bf + math.log(0.8),
    ]

    assert miner._choice_chunks(scores) == [0, 1]


def test_discrete_state_miner_reuses_candidate_pipeline_and_realigns_states() -> None:
    """验证 state-conditioned miner 复用候选流程并同步压缩 state 序列。"""

    miner = DiscreteStateTemporalPatternMiner(
        max_iterations=3,
        js_threshold=0.0,
        convergence_window=5,
        min_joint_probability=0.0,
        min_log_bayes_factor=0.0,
        score_ratio_threshold=0.0,
    )

    result = miner.mine_patterns(
        ["A", "B", "A", "B", "A", "B"],
        ess=2,
        constr_func=lambda parent, child: True,
        state_sequence=[1, 2, 1, 2, 1, 2],
    )

    assert result["vocab"] == ["A-B"]
    assert result["sequence"] == ["A-B", "A-B", "A-B"]
    assert result["state_sequence"] == [1, 1, 1]
    assert result["components"]["A-B"] == ["A", "B"]
    assert result["components"]["B-A"] == ["B", "A"]
    assert result["js_history"] == pytest.approx([0.8325546111576977])
