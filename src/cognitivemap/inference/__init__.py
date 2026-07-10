# -*- coding: utf-8 -*-
"""
inference 模块
=======================
提供从行为序列推断 action chunk、chunk 解析与 landmark candidate 的工具。

核心类：
    - TemporalPatternMiner：连续 one-hot 状态版 chunk 推断器
    - DiscreteStateTemporalPatternMiner：离散状态版 chunk 推断器
    - LandmarkMiningConfig：landmark candidate 推断配置

工具函数：
    - parse_with_chunks：Trie 最长匹配 chunk 解析器
    - mine_landmarks：从 state sequences 推断 landmark candidates
"""

from cognitivemap.inference.miner_chunk import TemporalPatternMiner
from cognitivemap.inference.miner_chunk_state_conditioned import DiscreteStateTemporalPatternMiner
from cognitivemap.inference.miner_landmark import (
    LandmarkMiningConfig,
    mine_landmarks,
    score_landmark_candidates,
    state_sequences_from_transition_rows,
)
from cognitivemap.inference.parser_chunk import TokenTrie, TrieNode, parse_with_chunks

__all__ = [
    "TemporalPatternMiner",
    "DiscreteStateTemporalPatternMiner",
    "LandmarkMiningConfig",
    "parse_with_chunks",
    "mine_landmarks",
    "score_landmark_candidates",
    "state_sequences_from_transition_rows",
    "TokenTrie",
    "TrieNode",
]
