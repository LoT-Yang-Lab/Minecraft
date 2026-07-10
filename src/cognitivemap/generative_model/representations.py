"""静态认知表征与表征校验。

本模块只处理“给定 task 后如何表达 agent 的认知转移偏好”。
它不从行为数据中学习参数，也不恢复 representation；所有权重、chunk inventory
和 landmark CPD 都来自显式输入或简单默认规则。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from cognitivemap.generative_model.task import TaskGraph
from cognitivemap.generative_model.types import (
    ActionId,
    ChunkToken,
    CPDBlock,
    Family,
    InvalidChunkRepresentationError,
    InvalidLandmarkRepresentationError,
    InvalidPrimitiveRepresentationError,
    InvalidRepresentationError,
    PrimitiveToken,
    StateId,
    Token,
)


@dataclass(frozen=True)
class RepresentationSpec:
    """构造表征用的声明式输入。

    Spec 不是运行期表征；它只是 builder 的输入容器。
    builder 会结合 TaskGraph 派生候选 token 集合并输出具体 Representation。
    """

    family: Family | str
    representation_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """规范化 family 与可变 mapping，避免外部后续修改影响 spec。"""

        object.__setattr__(self, "family", self.family if isinstance(self.family, Family) else Family(self.family))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "defaults", dict(self.defaults))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PrimitiveRepresentation:
    """primitive action 权重表征。

    对每个 state，权重行必须严格覆盖该 state 的合法 primitive actions，
    且概率和在容差内等于 1。
    """

    representation_id: str
    action_weights_by_state: Mapping[StateId, Mapping[ActionId, float]]
    family: Family = Family.PRIMITIVE

    def __post_init__(self) -> None:
        """把 primitive action 权重统一转换为普通 dict 和 float。"""

        object.__setattr__(
            self,
            "action_weights_by_state",
            {
                state: {action: float(weight) for action, weight in row.items()}
                for state, row in self.action_weights_by_state.items()
            },
        )

    def tokens_at(self, state: StateId, task: TaskGraph) -> list[PrimitiveToken]:
        """返回该 state 下按 task action 顺序排列的 primitive token。"""

        return [PrimitiveToken(action=action) for action in task.actions_at(state)]

    def weight(self, state: StateId, token: PrimitiveToken | ActionId) -> float:
        """查询 primitive action/token 在某个 state 下的概率权重。"""

        action = token.action if isinstance(token, PrimitiveToken) else token
        return self.action_weights_by_state[state][action]

    def validate(self, task: TaskGraph, tolerance: float = 1e-9) -> None:
        """校验 primitive 权重是否和 task 的合法动作完全一致。"""

        _ensure_states_match(
            self.action_weights_by_state, task, self.representation_id, InvalidPrimitiveRepresentationError
        )
        for state in task.states:
            legal = set(task.actions_at(state))
            row = self.action_weights_by_state[state]
            if set(row) != legal:
                raise InvalidPrimitiveRepresentationError(
                    "primitive weights must cover exactly legal actions",
                    object_id=self.representation_id,
                    context={"state": state, "expected": sorted(legal), "actual": sorted(row)},
                )
            _assert_probability_row(
                row.values(),
                tolerance,
                self.representation_id,
                {"state": state},
                InvalidPrimitiveRepresentationError,
            )


@dataclass(frozen=True)
class ChunkRepresentation:
    """action chunk token 权重表征。

    候选集合由两部分组成：先列 primitive tokens，再列从当前 state 可完整执行的 chunks。
    每个 state 的 token 权重必须严格覆盖这个 state-specific candidate set。
    """

    representation_id: str
    chunk_inventory: Mapping[str, tuple[ActionId, ...] | list[ActionId]]
    token_weights_by_state: Mapping[StateId, Mapping[str, float]]
    family: Family = Family.CHUNK

    def __post_init__(self) -> None:
        """规范化 chunk inventory 和逐 state token 权重，并拒绝空 chunk。"""

        inventory = {chunk_id: tuple(actions) for chunk_id, actions in self.chunk_inventory.items()}
        if any(not actions for actions in inventory.values()):
            empty = [chunk_id for chunk_id, actions in inventory.items() if not actions]
            raise InvalidChunkRepresentationError(
                "chunks must contain at least one primitive action",
                object_id=self.representation_id,
                context={"chunks": empty},
            )
        object.__setattr__(self, "chunk_inventory", inventory)
        object.__setattr__(
            self,
            "token_weights_by_state",
            {
                state: {token_id: float(weight) for token_id, weight in row.items()}
                for state, row in self.token_weights_by_state.items()
            },
        )

    def primitive_tokens_at(self, state: StateId, task: TaskGraph) -> list[PrimitiveToken]:
        """返回该 state 下的 primitive token 候选。"""

        return [PrimitiveToken(action=action) for action in task.actions_at(state)]

    def executable_chunks_at(self, state: StateId, task: TaskGraph) -> list[ChunkToken]:
        """返回从该 state 能完整执行的 chunk token。

        不能完整执行的 chunk 不进入候选集合，也不参与概率归一化。
        """

        chunks: list[ChunkToken] = []
        for chunk_id, actions in self.chunk_inventory.items():
            token = ChunkToken(chunk_id=chunk_id, actions=actions)
            try:
                task.execute_token(state, token)
            except Exception:
                continue
            chunks.append(token)
        return chunks

    def tokens_at(self, state: StateId, task: TaskGraph) -> list[Token]:
        """返回 chunk 表征下的完整候选 token 顺序。"""

        return [*self.primitive_tokens_at(state, task), *self.executable_chunks_at(state, task)]

    def weight(self, state: StateId, token: Token | str) -> float:
        """查询 primitive/chunk token 在某个 state 下的概率权重。"""

        token_id = token.token_id if isinstance(token, (PrimitiveToken, ChunkToken)) else token
        return self.token_weights_by_state[state][token_id]

    def validate(self, task: TaskGraph, tolerance: float = 1e-9) -> None:
        """校验 chunk inventory、可执行 token 集合和 token 权重行。"""

        vocabulary = set(task.actions)
        for chunk_id, actions in self.chunk_inventory.items():
            unknown = [action for action in actions if action not in vocabulary]
            if unknown:
                raise InvalidChunkRepresentationError(
                    "chunk contains unknown primitive actions",
                    object_id=self.representation_id,
                    context={"chunk_id": chunk_id, "unknown_actions": unknown},
                )

        _ensure_states_match(self.token_weights_by_state, task, self.representation_id, InvalidChunkRepresentationError)
        for state in task.states:
            expected = {token.token_id for token in self.tokens_at(state, task)}
            row = self.token_weights_by_state[state]
            if set(row) != expected:
                raise InvalidChunkRepresentationError(
                    "chunk token weights must cover exactly executable tokens",
                    object_id=self.representation_id,
                    context={"state": state, "expected": sorted(expected), "actual": sorted(row)},
                )
            _assert_probability_row(
                row.values(),
                tolerance,
                self.representation_id,
                {"state": state},
                InvalidChunkRepresentationError,
            )


@dataclass(frozen=True)
class LandmarkRepresentation:
    """具体 state landmark 的 CPD block 表征。

    第一版 landmark 是明确给出的 state set。CPD 分为 LL、LU、UL 三块：
    landmark 到 landmark、landmark 到 non-landmark、non-landmark 到 landmark。
    non-landmark 到 non-landmark 在 kernel 中固定为 0。
    """

    representation_id: str
    landmarks: tuple[StateId, ...] | list[StateId]
    p_ll: Mapping[StateId, Mapping[StateId, float]]
    p_lu: Mapping[StateId, Mapping[StateId, float]]
    p_ul: Mapping[StateId, Mapping[StateId, float]]
    family: Family = Family.LANDMARK

    def __post_init__(self) -> None:
        """规范化 landmark 集合与三块 CPD 概率表。"""

        object.__setattr__(self, "landmarks", tuple(self.landmarks))
        object.__setattr__(self, "p_ll", _float_nested_mapping(self.p_ll))
        object.__setattr__(self, "p_lu", _float_nested_mapping(self.p_lu))
        object.__setattr__(self, "p_ul", _float_nested_mapping(self.p_ul))

    def non_landmarks(self, task: TaskGraph) -> tuple[StateId, ...]:
        """返回 task states 中不属于 landmark 集合的状态。"""

        landmark_set = set(self.landmarks)
        return tuple(state for state in task.states if state not in landmark_set)

    def block_for(self, source: StateId, target: StateId, task: TaskGraph) -> tuple[CPDBlock | None, float]:
        """根据 source/target 是否为 landmark 查询对应 CPD block 和概率。"""

        landmark_set = set(self.landmarks)
        if source in landmark_set and target in landmark_set:
            return CPDBlock.LL, self.p_ll[source][target]
        if source in landmark_set and target not in landmark_set:
            return CPDBlock.LU, self.p_lu[source][target]
        if source not in landmark_set and target in landmark_set:
            return CPDBlock.UL, self.p_ul[source][target]
        return None, 0.0

    def validate(self, task: TaskGraph, tolerance: float = 1e-9) -> None:
        """校验 landmark 集合与三块 CPD 的行列覆盖和概率归一化。"""

        if len(set(self.landmarks)) != len(self.landmarks) or len(self.landmarks) < 2:
            raise InvalidLandmarkRepresentationError(
                "landmarks must contain at least two unique states",
                object_id=self.representation_id,
            )
        unknown = [state for state in self.landmarks if state not in task.states]
        if unknown:
            raise InvalidLandmarkRepresentationError(
                "landmarks must belong to task states",
                object_id=self.representation_id,
                context={"unknown_landmarks": unknown},
            )

        landmarks = set(self.landmarks)
        non_landmarks = set(self.non_landmarks(task))
        _ensure_exact_rows(self.p_ll, landmarks, landmarks, self.representation_id, "p_ll")
        _ensure_exact_rows(self.p_lu, landmarks, non_landmarks, self.representation_id, "p_lu")
        _ensure_exact_rows(self.p_ul, non_landmarks, landmarks, self.representation_id, "p_ul")

        for source in self.landmarks:
            values = [*self.p_ll[source].values(), *self.p_lu[source].values()]
            _assert_probability_row(
                values,
                tolerance,
                self.representation_id,
                {"source": source, "blocks": "LL+LU"},
                InvalidLandmarkRepresentationError,
            )
        for source in self.non_landmarks(task):
            _assert_probability_row(
                self.p_ul[source].values(),
                tolerance,
                self.representation_id,
                {"source": source, "block": "UL"},
                InvalidLandmarkRepresentationError,
            )


Representation = PrimitiveRepresentation | ChunkRepresentation | LandmarkRepresentation


class RepresentationBuilder:
    """从 task 和显式参数构造三类静态表征。

    Builder 的职责是“展开默认规则并校验兼容性”，不是从数据里学习参数。
    Agent 只接收已经构造好的 representation，不负责构造或修正表征。
    """

    @staticmethod
    def build_primitive(
        task: TaskGraph,
        representation_id: str,
        action_weights: Mapping[StateId, Mapping[ActionId, float]] | None = None,
        policy: str = "uniform",
    ) -> PrimitiveRepresentation:
        """构造 primitive representation。

        若未提供 action_weights，则使用 uniform policy 给每个合法 action 相同概率。
        """

        weights = action_weights or _uniform_primitive_weights(task, representation_id, policy)
        representation = PrimitiveRepresentation(
            representation_id=representation_id,
            action_weights_by_state=weights,
        )
        representation.validate(task)
        return representation

    @staticmethod
    def build_chunk(
        task: TaskGraph,
        representation_id: str,
        chunk_inventory: Mapping[str, tuple[ActionId, ...] | list[ActionId]],
        token_weights: Mapping[StateId, Mapping[str, float]] | None = None,
        policy: str = "uniform",
    ) -> ChunkRepresentation:
        """构造 chunk representation。

        若未提供 token_weights，则先根据 task 过滤 executable chunks，
        再对每个 state 的候选 token 集合使用 uniform policy。
        """

        provisional = ChunkRepresentation(
            representation_id=representation_id,
            chunk_inventory=chunk_inventory,
            token_weights_by_state=token_weights or {},
        )
        weights = token_weights or _uniform_chunk_weights(task, provisional, representation_id, policy)
        representation = ChunkRepresentation(
            representation_id=representation_id,
            chunk_inventory=chunk_inventory,
            token_weights_by_state=weights,
        )
        representation.validate(task)
        return representation

    @staticmethod
    def build_landmark(
        task: TaskGraph,
        representation_id: str,
        landmarks: tuple[StateId, ...] | list[StateId],
        cpd_blocks: Mapping[str, Mapping[StateId, Mapping[StateId, float]]],
    ) -> LandmarkRepresentation:
        """构造 landmark representation。

        landmark 第一版只支持具体 state set；CPD blocks 必须显式给出。
        """

        representation = LandmarkRepresentation(
            representation_id=representation_id,
            landmarks=landmarks,
            p_ll=cpd_blocks["p_ll"],
            p_lu=cpd_blocks["p_lu"],
            p_ul=cpd_blocks["p_ul"],
        )
        representation.validate(task)
        return representation

    @staticmethod
    def build(task: TaskGraph, spec: RepresentationSpec) -> Representation:
        """根据 RepresentationSpec 分派到具体 builder。"""

        parameters = dict(spec.parameters)
        defaults = dict(spec.defaults)
        if spec.family == Family.PRIMITIVE:
            return RepresentationBuilder.build_primitive(
                task,
                spec.representation_id,
                action_weights=parameters.get("action_weights_by_state") or parameters.get("action_weights"),
                policy=defaults.get("policy", "uniform"),
            )
        if spec.family == Family.CHUNK:
            return RepresentationBuilder.build_chunk(
                task,
                spec.representation_id,
                chunk_inventory=parameters["chunk_inventory"],
                token_weights=parameters.get("token_weights_by_state") or parameters.get("token_weights"),
                policy=defaults.get("policy", "uniform"),
            )
        if spec.family == Family.LANDMARK:
            cpd_blocks = parameters.get("cpd_blocks", parameters)
            return RepresentationBuilder.build_landmark(
                task,
                spec.representation_id,
                landmarks=parameters["landmarks"],
                cpd_blocks=cpd_blocks,
            )
        raise InvalidRepresentationError(f"unsupported representation family: {spec.family!r}")

    @staticmethod
    def validate_compatible(task: TaskGraph, representation: Representation, tolerance: float = 1e-9) -> None:
        """校验 representation 是否可用于给定 TaskGraph。"""

        validate_representation(task, representation, tolerance)


def validate_representation(task: TaskGraph, representation: Representation, tolerance: float = 1e-9) -> None:
    """根据 representation 的具体类型执行兼容性校验。"""

    if isinstance(representation, PrimitiveRepresentation):
        representation.validate(task, tolerance)
    elif isinstance(representation, ChunkRepresentation):
        representation.validate(task, tolerance)
    elif isinstance(representation, LandmarkRepresentation):
        representation.validate(task, tolerance)
    else:
        raise InvalidRepresentationError(f"unsupported representation type: {type(representation)!r}")


def _float_nested_mapping(mapping: Mapping[StateId, Mapping[StateId, float]]) -> dict[StateId, dict[StateId, float]]:
    """把嵌套概率 mapping 规范化为 float 字典。"""

    return {source: {target: float(value) for target, value in row.items()} for source, row in mapping.items()}


def _uniform_primitive_weights(
    task: TaskGraph,
    representation_id: str,
    policy: str,
) -> dict[StateId, dict[ActionId, float]]:
    """为 primitive representation 生成每个 state 的 uniform action 权重。"""

    if policy != "uniform":
        raise InvalidPrimitiveRepresentationError(
            "unsupported primitive policy",
            object_id=representation_id,
            context={"policy": policy},
        )
    weights: dict[StateId, dict[ActionId, float]] = {}
    for state in task.states:
        actions = task.legal_actions(state)
        if not actions:
            raise InvalidPrimitiveRepresentationError(
                "uniform primitive policy requires at least one legal action per state",
                object_id=representation_id,
                context={"state": state},
            )
        probability = 1.0 / len(actions)
        weights[state] = {action: probability for action in actions}
    return weights


def _uniform_chunk_weights(
    task: TaskGraph,
    representation: ChunkRepresentation,
    representation_id: str,
    policy: str,
) -> dict[StateId, dict[str, float]]:
    """为 chunk representation 生成每个 state 的 uniform token 权重。"""

    if policy != "uniform":
        raise InvalidChunkRepresentationError(
            "unsupported chunk policy",
            object_id=representation_id,
            context={"policy": policy},
        )
    weights: dict[StateId, dict[str, float]] = {}
    for state in task.states:
        tokens = representation.tokens_at(state, task)
        if not tokens:
            raise InvalidChunkRepresentationError(
                "uniform chunk policy requires at least one executable token per state",
                object_id=representation_id,
                context={"state": state},
            )
        probability = 1.0 / len(tokens)
        weights[state] = {token.token_id: probability for token in tokens}
    return weights


def _ensure_states_match(
    rows: Mapping[StateId, object],
    task: TaskGraph,
    representation_id: str,
    error_type: type[InvalidRepresentationError],
) -> None:
    """校验概率表是否为 task 中每个 state 都提供一行，且没有额外 state。"""

    if set(rows) != set(task.states):
        raise error_type(
            "representation rows must cover exactly task states",
            object_id=representation_id,
            context={"expected": list(task.states), "actual": list(rows)},
        )


def _ensure_exact_rows(
    block: Mapping[StateId, Mapping[StateId, float]],
    expected_sources: set[StateId],
    expected_targets: set[StateId],
    representation_id: str,
    block_name: str,
) -> None:
    """校验 CPD block 的 source 行和 target 列是否精确匹配期望集合。"""

    if set(block) != expected_sources:
        raise InvalidLandmarkRepresentationError(
            "CPD block source rows are invalid",
            object_id=representation_id,
            context={
                "block": block_name,
                "expected": sorted(expected_sources, key=repr),
                "actual": sorted(block, key=repr),
            },
        )
    for source, row in block.items():
        if set(row) != expected_targets:
            raise InvalidLandmarkRepresentationError(
                "CPD block target columns are invalid",
                object_id=representation_id,
                context={
                    "block": block_name,
                    "source": source,
                    "expected": sorted(expected_targets, key=repr),
                    "actual": sorted(row, key=repr),
                },
            )


def _assert_probability_row(
    values: object,
    tolerance: float,
    representation_id: str,
    context: dict[str, object],
    error_type: type[InvalidRepresentationError],
) -> None:
    """校验一行概率是否非负、有限且在容差内归一化为 1。"""

    probabilities = [float(value) for value in values]
    if any(value < 0 or not math.isfinite(value) for value in probabilities):
        raise error_type(
            "probabilities must be non-negative finite values",
            object_id=representation_id,
            context=context,
        )
    row_sum = sum(probabilities)
    if abs(row_sum - 1.0) > tolerance:
        raise error_type(
            "probability row must sum to 1",
            object_id=representation_id,
            context={**context, "row_sum": row_sum},
        )
