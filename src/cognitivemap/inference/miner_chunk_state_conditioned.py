"""
离散状态版时间序列 chunk 推断器。

这个版本与 miner_chunk.py 的主要区别只有两点：
1. trial 之间是否插入分隔符由外部脚本决定，这里不强制依赖 "S"；
2. 状态不再展开成 9 个二值变量，而是作为 1 个 9 值离散父节点参与 BDeu 评分。
"""

import numpy as np
import pandas as pd

from .miner_chunk import TemporalPatternMiner


class DiscreteStateTemporalPatternMiner(TemporalPatternMiner):
    """离散状态版时间序列 chunk 推断器。

    与父类 TemporalPatternMiner 的区别：
    1. 状态编码为 1 个 9 值离散变量（而非 9 个二值 one-hot 变量）
    2. trial 之间不强制依赖 "S" 分隔符
    3. 不接受 -1 作为状态值

    ``_mine_patterns`` 完全继承自父类，本类只覆盖两个 hook：
    - ``_sequence_to_temporal_onehot``：状态转为单列离散值
    - ``_prepare_state_parent_for_bdeu``：bde_score 参数配置为离散变量
    """

    def _sequence_to_temporal_onehot(self, sequence, state_sequence=None):
        """
        把动作序列转成 parent/child one-hot，并可选地保留一个与 parent 对齐的 9 值状态变量。

        返回值：
        - data_child: 动作 child one-hot
        - data_parent: 动作 parent one-hot
        - state_parent: 单列 DataFrame，内部取值为 0..8（BDeu 要求 0-indexed）
        - elements: 当前 chunk 序列里出现过的动作/grammar 符号
        """
        series = pd.Series(sequence)
        data = pd.get_dummies(series, dtype=int)
        data_child = data.iloc[1:].copy()
        data_parent = data.iloc[:-1].copy()
        elements = list(data.columns)

        state_parent = None
        if state_sequence is not None:
            if len(state_sequence) != len(sequence):
                raise ValueError("state_sequence length must match sequence length.")
            for state in state_sequence:
                if not 1 <= state <= 9:
                    raise ValueError(f"Invalid state value {state}; discrete-state miner expects integers in [1, 9].")
            # BDeu 评分要求离散变量为 0-indexed，内部偏移到 0..8。
            state_parent = pd.DataFrame({"state_parent": [s - 1 for s in state_sequence[:-1]]}, dtype=int)

        return data_child, data_parent, state_parent, elements

    def _prepare_state_parent_for_bdeu(self, state_parent_df):
        """钩子函数：离散状态版使用 1 个 9 值变量（而非 9 个二值 one-hot）。"""
        if state_parent_df is None:
            return None, None
        values = state_parent_df["state_parent"].to_numpy(dtype=np.int64).reshape(1, -1)
        return values, [9]
