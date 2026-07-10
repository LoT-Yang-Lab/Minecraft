"""
时间序列 chunk 推断器。

这个模块实现的是一种“自底向上”的 chunk 推断流程：

1. 把当前序列看作离散 token 序列。
2. 统计相邻时间步的依赖关系，也就是 X_{t-1} -> X_t。
3. 用 BDeu 评分比较“child 依赖 parent”与“child 独立”两种模型。
4. 配合支持度、联合频率、提升度、归一化 Bayes factor 等条件筛掉弱模式。
5. 从分数最高的一批候选里选出新的二元 chunk。
6. 回到原始序列上重新做最长匹配切分，形成新的 chunk_sequence。
7. 重复以上过程，直到没有新模式或分布变化足够小。

它和 parser_chunk.py 的关系是：
- 这里负责“发现哪些 chunk 值得加入词表”；
- parser_chunk.py 负责“给定词表后，怎样把原始序列真正重新切分”。
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pgm_toolkit.learning.structure_learning.scoring import bde_score
from scipy.spatial.distance import jensenshannon

from .parser_chunk import parse_with_chunks


@dataclass(frozen=True)
class _CandidateChunk:
    """一轮挖掘中通过统计过滤的二元 chunk 候选。"""

    # 用于多候选选择的分数：log BF / 有效 BDeu 样本数。
    score: float
    # 原始总 log BF，用于 min_log_bayes_factor 这类总体证据阈值。
    log_bayes_factor: float
    sample_count: int
    chunk: str
    parent: str
    child: str


class TemporalPatternMiner:
    """基于相邻 token 依赖和 BDeu 证据的自底向上 chunk miner。"""

    def __init__(
        self,
        max_iterations=10000,
        js_threshold=0.05,
        convergence_window=5,
        min_joint_probability=0.03,
        min_log_bayes_factor=np.log(3.0),
        score_ratio_threshold=0.85,
        structural_tokens=None,
    ):
        """保存挖掘超参数，并把结构 token 固定为不可变集合。"""

        # 最多迭代多少轮。每轮都会尝试在当前 chunk 序列上发现新模式。
        self.max_iterations = max_iterations
        # 如果最近 convergence_window 轮的 JS 距离均值低于该阈值，则视为收敛。
        self.js_threshold = js_threshold
        self.convergence_window = convergence_window
        # 候选 chunk 的最小联合频率下界，对应旧版 generateGrammar.py 里的固定阈值。
        self.min_joint_probability = min_joint_probability
        # 候选 chunk 的绝对证据阈值。这里使用原始总 log BF，因此 np.log(3) 表示 BF > 3。
        self.min_log_bayes_factor = min_log_bayes_factor
        # 候选 chunk 的相对保留阈值：基于单位样本 BF，而不是会随数据量放大的总 BF。
        self.score_ratio_threshold = score_ratio_threshold
        # 结构 token 只用于序列边界控制，不参与候选评分、总体概率和 JS 分布。
        self.structural_tokens = frozenset(structural_tokens or ())

    @staticmethod
    def _contains_any_token(element, tokens):
        """判断一个原子 token 或复合 chunk 是否包含指定 token。"""
        if not tokens:
            return False
        if element in tokens:
            return True
        return any(part in tokens for part in str(element).split("-"))

    def _is_structural_element(self, element):
        """判断元素是否为结构 token，或是否由结构 token 参与组成。"""
        return self._contains_any_token(element, self.structural_tokens)

    def _nonstructural_elements(self, elements):
        """返回用于统计学习的非结构元素。"""
        return [element for element in elements if not self._is_structural_element(element)]

    def _sequence_to_temporal_onehot(self, sequence, state_sequence=None):
        """
        把离散序列转成时间对齐的 one-hot 表示。

        输入 sequence = [x_0, x_1, ..., x_T] 后，得到：
        - data_parent: 对应 [x_0, x_1, ..., x_{T-1}]
        - data_child:  对应 [x_1, x_2, ..., x_T]
        - state_parent: 对应前一时刻状态 [s_0, s_1, ..., s_{T-1}] 的 9 维 one-hot

        其中状态值约定如下：
        - 合法状态为 1..9
        - -1 只用于 trial 分隔占位，例如动作里的 "S"
        - 当状态为 -1 时，这一行状态 one-hot 全部置 0，不把 -1 当成真实状态类别

        这样第 k 行的 parent 和 child 就构成一个时间相邻对：
        (X_t = parent, X_{t+1} = child)。
        """
        series = pd.Series(sequence)
        # 对每个离散 token 做 one-hot 编码。
        data = pd.get_dummies(series, dtype=int)
        # child 序列从第 1 个时间步开始。
        data_child = data.iloc[1:].copy()
        # parent 序列到倒数第 2 个时间步结束。
        data_parent = data.iloc[:-1].copy()
        # 当前序列中出现过的所有元素名，既可能是原子 token，也可能是高层 chunk。
        elements = list(data.columns)

        state_parent = None
        if state_sequence is not None:
            if len(state_sequence) != len(sequence):
                raise ValueError("state_sequence length must match sequence length.")

            # 状态固定使用 9 维 one-hot，内部索引 0..8 对应状态 1..9。
            state_data = np.zeros((len(state_sequence), 9), dtype=int)
            for idx, state in enumerate(state_sequence):
                if state == -1:
                    # 分隔占位不属于真实状态，整行保持全 0。
                    continue
                if 1 <= state <= 9:
                    state_data[idx, state - 1] = 1
                    continue
                raise ValueError(f"Invalid state value {state}; expected -1 or an integer in [1, 9].")

            state_columns = [f"state_{idx}" for idx in range(9)]
            state_frame = pd.DataFrame(state_data, columns=state_columns)
            state_parent = state_frame.iloc[:-1].copy()

        return data_child, data_parent, state_parent, elements

    def _get_element_frequency(self, sequence, ignored_tokens=None):
        """
        统计序列中各元素的经验频率，并归一化成概率分布。

        这个分布既用于：
        - 比较 re-parse 前后的整体分布差异（JS 距离）
        - 也用于理解当前 chunk 序列的词表占比
        """
        ignored_tokens = self.structural_tokens if ignored_tokens is None else frozenset(ignored_tokens)
        if ignored_tokens:
            frequency = dict(
                Counter(element for element in sequence if not self._contains_any_token(element, ignored_tokens))
            )
        else:
            frequency = dict(Counter(sequence))
        total = sum(frequency.values())
        if total == 0:
            return {}
        probability = {element: count / total for element, count in frequency.items()}
        return probability

    def _choice_chunks(self, scores):
        """
        从候选分数中选出“足够接近最优”的那一批候选。

        scores 采用单位样本 log BF，即 log_bayes_factor / 有效样本数。
        因此这里保留的是单位样本 BF / 最强单位样本 BF 大于阈值的候选，
        避免总 log BF 随数据量放大后只剩最高分候选。
        """
        scores = np.asarray(scores)

        if scores.size == 0:
            return []
        if self.score_ratio_threshold <= 0:
            return list(range(scores.size))
        max_score = scores.max()
        log_ratio_threshold = np.log(self.score_ratio_threshold)
        mask = scores - max_score > log_ratio_threshold
        idxs = np.where(mask)[0].tolist()
        return idxs

    @staticmethod
    def _get_js(element_probability, chunk_probability, elements=None):
        """
        计算两份离散分布之间的 Jensen-Shannon 距离。

        这里用于比较：
        - 旧的 chunk_sequence 分布
        - 新的 re-parse 后序列分布

        如果连续多轮的 JS 距离都很小，说明词表继续扩张带来的分布变化已经不明显，
        算法就可以停止。
        """
        # 两份分布的支持集可能不同，因此先求并集，确保它们投影到同一个坐标系里。
        # elements 只作为可选的已有词表输入；新增的零概率维度会在下面过滤掉，不影响 JS 值。
        support = list(
            dict.fromkeys(list(elements or []) + list(element_probability.keys()) + list(chunk_probability.keys()))
        )
        element_to_index = {element: idx for idx, element in enumerate(support)}
        P1 = np.zeros(len(support))
        P2 = np.zeros(len(support))

        for key, value in element_probability.items():
            P1[element_to_index[key]] = value
        for key, value in chunk_probability.items():
            P2[element_to_index[key]] = value

        # 只保留至少有一边非 0 的位置，避免在全零维度上做无意义计算。
        indices = np.where((P1 != 0) | (P2 != 0))[0]
        if len(indices) == 0:
            return 0.0
        P1 = P1[indices]
        P2 = P2[indices]
        js = jensenshannon(P1, P2)
        return js

    def _valid_scoring_pair_mask(self, sequence):
        """标记哪些相邻 pair 可以作为真实统计样本。"""
        pair_count = max(len(sequence) - 1, 0)
        if pair_count == 0:
            return np.zeros(0, dtype=bool)
        if not self.structural_tokens:
            return np.ones(pair_count, dtype=bool)
        return np.asarray(
            [
                not self._is_structural_element(parent) and not self._is_structural_element(child)
                for parent, child in zip(sequence[:-1], sequence[1:])
            ],
            dtype=bool,
        )

    def _filter_scoring_pairs(self, data_child, data_parent, state_parent, sequence):
        """排除含结构 token 的相邻行，避免分隔符污染统计评分。"""
        valid_pair_mask = self._valid_scoring_pair_mask(sequence)
        data_child = data_child.iloc[valid_pair_mask].reset_index(drop=True)
        data_parent = data_parent.iloc[valid_pair_mask].reset_index(drop=True)
        if state_parent is not None:
            state_parent = state_parent.iloc[valid_pair_mask].reset_index(drop=True)
        return data_child, data_parent, state_parent

    def _scoring_token_count(self, sequence):
        """统计真实 token 数量；结构 token 不计入概率分母。"""
        return sum(1 for element in sequence if not self._is_structural_element(element))

    def _build_parent_scoring_context(
        self,
        parent,
        data_parent,
        state_bdeu_data,
        state_bdeu_num_states,
    ):
        """准备某个 parent 在 BDeu 打分时复用的父节点矩阵。"""
        data_p = data_parent[parent].to_numpy(dtype=np.int64).reshape(1, -1)
        if int(data_p.sum()) == 0:
            return None

        score_parent = data_p
        parent_num_states = [2]
        if state_bdeu_data is not None:
            score_parent = np.vstack([data_p, state_bdeu_data])
            parent_num_states = [2] + list(state_bdeu_num_states)

        return data_p, score_parent, parent_num_states

    def _score_candidate_pair(
        self,
        parent,
        child,
        data_p,
        data_child,
        score_parent,
        parent_num_states,
        state_bdeu_data,
        state_bdeu_num_states,
        element_probability,
        scoring_token_count,
        ess,
    ):
        """对一个 parent -> child 组合打分；不满足过滤条件时返回 None。"""
        data_c = data_child[child].to_numpy(dtype=np.int64).reshape(1, -1)
        child_num_states = [2]
        sample_count = data_c.shape[1]
        if sample_count == 0:
            return None
        if int(data_c.sum()) == 0:
            return None

        score1, _, _ = bde_score(
            data_c,
            score_parent,
            child_num_states,
            parent_num_states,
            ess,
        )
        if state_bdeu_data is not None:
            score2, _, _ = bde_score(
                data_c,
                state_bdeu_data,
                child_num_states,
                list(state_bdeu_num_states),
                ess,
            )
        else:
            score2, _, _ = bde_score(data_c, [], child_num_states, [], ess)

        # 引入 state_parent 后，bde_score 的联合父节点已不是简单 2x2 表。
        # 这里仍只基于动作 parent 与 child 这对二值变量计算联合出现次数，保持旧统计口径。
        joint_count = int(np.sum((data_p[0] == 1) & (data_c[0] == 1)))

        # bde_score 返回 log marginal likelihood，因此 parent model 的支持强度是 log BF。
        log_bayes_factor = score1 - score2
        if scoring_token_count == 0:
            return None
        p_parent = element_probability.get(parent, 0.0)
        p_child = element_probability.get(child, 0.0)
        p_joint = joint_count / scoring_token_count

        if p_joint < p_parent * p_child or p_joint < self.min_joint_probability:
            return None
        if log_bayes_factor <= self.min_log_bayes_factor:
            return None

        selection_score = log_bayes_factor / sample_count
        return _CandidateChunk(
            score=selection_score,
            log_bayes_factor=log_bayes_factor,
            sample_count=sample_count,
            chunk=parent + "-" + child,
            parent=parent,
            child=child,
        )

    def _collect_candidate_chunks(
        self,
        data_child,
        data_parent,
        elements,
        element_probability,
        state_bdeu_data,
        state_bdeu_num_states,
        scoring_token_count,
        ess,
        constr_func,
    ):
        """穷举当前词表里的 parent -> child 组合，并返回通过过滤的候选。"""
        candidates = []

        for parent in elements:
            parent_context = self._build_parent_scoring_context(
                parent,
                data_parent,
                state_bdeu_data,
                state_bdeu_num_states,
            )
            if parent_context is None:
                continue
            data_p, score_parent, parent_num_states = parent_context

            for child in elements:
                if constr_func is not None and not constr_func(parent, child):
                    continue

                candidate = self._score_candidate_pair(
                    parent=parent,
                    child=child,
                    data_p=data_p,
                    data_child=data_child,
                    score_parent=score_parent,
                    parent_num_states=parent_num_states,
                    state_bdeu_data=state_bdeu_data,
                    state_bdeu_num_states=state_bdeu_num_states,
                    element_probability=element_probability,
                    scoring_token_count=scoring_token_count,
                    ess=ess,
                )
                if candidate is not None:
                    candidates.append(candidate)

        return candidates

    def _choose_candidate_chunks(self, candidates, components):
        """从候选中选出本轮真正进入词表的 chunk，并同步更新 components。"""
        indices = self._choice_chunks([candidate.score for candidate in candidates])
        chosen_chunks = []

        for idx in indices:
            candidate = candidates[idx]
            components[candidate.chunk] = [candidate.parent, candidate.child]
            chosen_chunks.append(candidate.chunk)

        return chosen_chunks

    @staticmethod
    def _reparse_sequence_and_states(sequence, state_sequence, vocab):
        """
        按当前词表重新切分原始动作序列，并同步得到 chunk 级状态序列。

        状态对齐规则：
        - 一个 chunk 覆盖多个动作时，保留该 chunk 第一个动作对应的状态；
        - 单 token fallback 时，状态原样保留；
        - trial 分隔符 S 对应的状态保持为 -1。
        """
        new_sequence, start_indices = parse_with_chunks(sequence, vocab, need_index=True)

        if state_sequence is None:
            return new_sequence, None

        new_state_sequence = []
        for chunk, start_idx in zip(new_sequence, start_indices):
            new_state_sequence.append(state_sequence[start_idx])

        return new_sequence, new_state_sequence

    def _prepare_state_parent_for_bdeu(self, state_parent_df):
        """钩子函数：将 state_parent DataFrame 转换为 bde_score 可用的格式。

        子类可覆盖此方法来改变状态变量的编码方式（如连续 one-hot vs 离散值）。

        返回 (state_data, state_num_states)，若 state_parent_df 为 None 则返回 (None, None)。
        """
        if state_parent_df is None:
            return None, None
        matrix = state_parent_df.to_numpy(dtype=np.int64).T
        num_states = [2] * matrix.shape[0]
        return matrix, num_states

    def _mine_patterns(
        self,
        chunk_sequence,
        chunk_state_sequence,
        components,
        sequence,
        state_sequence,
        ess: float = 2,
        constr_func=None,
    ):
        """
        执行一轮 chunk 推断。

        参数
        ----
        chunk_sequence:
            当前轮正在使用的序列表示。它可能已经不是最原始 token，而是上一轮切分后的 chunk 序列。
        chunk_state_sequence:
            与当前 chunk_sequence 对齐的状态序列。若当前 chunk 是由多个动作合成，则状态取该 chunk
            覆盖区间第一个动作对应的状态（即该 chunk 开始执行前的状态）。
        components:
            记录每个 chunk 由哪些子成分组成。
            例如 "A-B" -> ["A", "B"]；更高层 chunk 则会继续引用已有 chunk。
        sequence:
            最原始的 token 序列。注意：re-parse 时始终回到这条原始序列上切分。
        state_sequence:
            与原始 sequence 对齐的状态序列。重切分时会基于它重新生成下一轮的 chunk 级状态序列。
        ess:
            BDeu 评分所需的 equivalent sample size。
        constr_func:
            外部约束函数。若返回 False，则某个 parent-child 组合即便统计上很强，也不会被加入。

        返回
        ----
        new_sequence, vocab, components, js
        其中 js=None 表示本轮没有发现任何新候选，属于终止信号。
        """
        # 1. 把当前 chunk 序列转成时间相邻 one-hot 表达，并统计当前边际分布。
        data_child, data_parent, state_parent, elements = self._sequence_to_temporal_onehot(
            chunk_sequence,
            chunk_state_sequence,
        )
        data_child, data_parent, state_parent = self._filter_scoring_pairs(
            data_child,
            data_parent,
            state_parent,
            chunk_sequence,
        )
        scoring_elements = self._nonstructural_elements(elements)
        element_probability = self._get_element_frequency(chunk_sequence)
        scoring_token_count = self._scoring_token_count(chunk_sequence)
        state_bdeu_data, state_bdeu_num_states = self._prepare_state_parent_for_bdeu(state_parent)

        # 2. 穷举所有 parent -> child 组合。
        # 当前实现只学习“长度为 2 的相邻组合”，也就是 parent-child 形式。
        candidates = self._collect_candidate_chunks(
            data_child=data_child,
            data_parent=data_parent,
            elements=scoring_elements,
            element_probability=element_probability,
            state_bdeu_data=state_bdeu_data,
            state_bdeu_num_states=state_bdeu_num_states,
            scoring_token_count=scoring_token_count,
            ess=ess,
            constr_func=constr_func,
        )

        # 3. 若本轮没有任何新候选，则用 js=None 作为终止信号返回。
        if len(candidates) == 0:
            return chunk_sequence, chunk_state_sequence, scoring_elements, components, None

        # 4. 从所有候选里选择“接近最优”的那一批。
        # components 不做彻底摊平，而是保留层级结构。
        # 例如新 chunk 可能引用上轮已经学到的旧 chunk。
        chosen_chunks = self._choose_candidate_chunks(candidates, components)

        # 本轮词表 = 当前已存在元素 + 新选中的 chunk。
        # 注意 elements 来自当前 chunk_sequence，因此其中已经包含历史轮次学到的 chunk。
        vocab = scoring_elements + chosen_chunks

        # 5. 在“原始 sequence”上重新切分，而不是在当前 chunk_sequence 上局部拼接。
        # 这一步非常关键：它保证每一轮都能在统一基准上重新解释整条序列。
        new_sequence, new_state_sequence = self._reparse_sequence_and_states(sequence, state_sequence, vocab)

        # 6. 比较 re-parse 前后的分布差异，用于后续收敛判断。
        chunk_probability = self._get_element_frequency(new_sequence)
        js = self._get_js(element_probability, chunk_probability, vocab)

        return new_sequence, new_state_sequence, vocab, components, js

    def mine_patterns(self, sequence: list, ess: float = 2, constr_func=None, state_sequence=None):
        """
        对一条原始序列做多轮 chunk 推断，直到收敛或达到上限。

        参数
        ----
        sequence:
            原始 token 序列，例如 ["U", "L", "D", "R", "S", "U", "U"]。
        ess:
            BDeu 评分所用的 equivalent sample size。
        constr_func:
            约束函数。返回 False 的 parent-child 对不会被学习成 chunk。
        state_sequence:
            与原始动作序列逐位置对齐的状态序列。若提供，会把前一时刻状态 one-hot 一起作为
            条件父节点送入 bde_score；trial 分隔符对应的位置应使用 -1。

        返回
        ----
        dict，包含：
        - vocab: 最终保留下来的 chunk 词表
        - sequence: 最后一轮 re-parse 后的 chunk 序列（动作维度）
        - state_sequence: 与 chunk 序列对齐的状态序列；若未提供 state_sequence 则为 None
        - components: 每个 chunk 的组成关系
        - js_history: 每轮 JS 距离历史
        """
        if state_sequence is not None and len(state_sequence) != len(sequence):
            raise ValueError("state_sequence length must match sequence length.")

        # 初始时，每个原子元素都把自己当作一个最小组件。
        components = {element: [element] for element in set(sequence)}

        # 第一轮的 chunk_sequence 就是原始序列本身。
        chunk_sequence = list(sequence)
        chunk_state_sequence = list(state_sequence) if state_sequence is not None else None
        js_list = []
        chunks = list(dict.fromkeys(sequence))

        for _ in range(self.max_iterations):
            chunk_sequence, chunk_state_sequence, chunks, components, js = self._mine_patterns(
                chunk_sequence,
                chunk_state_sequence,
                components,
                sequence,
                state_sequence,
                ess,
                constr_func,
            )

            # js=None 表示本轮完全没有发现新模式，直接停止。
            if js is None:
                break

            js_list.append(js)

            # 如果最近若干轮的平均 JS 已经足够小，说明词表继续扩张的收益不大。
            if (
                len(js_list) >= self.convergence_window
                and np.mean(js_list[-self.convergence_window :]) <= self.js_threshold
            ):
                break

        # 输出阶段再做一次“最终词表清理”。
        # 这里沿用当前实现逻辑：只保留在最终 chunks 列表中且频率大于 0 的项。
        # 需要注意，chunks 是最后一轮 _mine_patterns 返回的词表，而不是所有历史候选的合集。
        chunk_probability = self._get_element_frequency(chunks)
        vocab = []
        for chunk in chunks:
            if chunk in chunk_probability and chunk_probability[chunk] > 0:
                vocab.append(chunk)

        return {
            "vocab": vocab,
            "sequence": chunk_sequence,
            "state_sequence": chunk_state_sequence,
            "components": components,
            "js_history": js_list,
        }
