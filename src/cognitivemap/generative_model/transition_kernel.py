"""从认知表征构造 transition kernel 与 kernel support。

Kernel 是统一的状态转移概率矩阵；Support 是并列的概率来源记录。
图搜索、距离和 embedding 只依赖 kernel；路径展开和可追溯输出再使用 support。
"""

from __future__ import annotations

import math
from collections import defaultdict

from cognitivemap.generative_model.representations import (
    ChunkRepresentation,
    LandmarkRepresentation,
    PrimitiveRepresentation,
    Representation,
    validate_representation,
)
from cognitivemap.generative_model.task import TaskGraph
from cognitivemap.generative_model.types import (
    CPDBlock,
    Family,
    InvalidKernelError,
    KernelSupport,
    KernelSupportEntry,
    PlanningConfig,
    PrimitiveToken,
    StateId,
    SupportKind,
    TransitionKernel,
)


def build_transition_kernel(
    task: TaskGraph,
    representation: Representation,
    config: PlanningConfig | None = None,
) -> tuple[TransitionKernel, KernelSupport]:
    """按 representation family 分派，构造行随机 transition kernel 与 support。

    返回值严格是 ``(TransitionKernel, KernelSupport)``，support 不嵌入 kernel。
    构造完成后同时校验 kernel 形状/行和，以及 support 与非零 edge 的一致性。
    """

    config = config or PlanningConfig()
    validate_representation(task, representation, config.row_sum_tolerance)
    if isinstance(representation, PrimitiveRepresentation):
        kernel, support = build_primitive_kernel(task, representation)
    elif isinstance(representation, ChunkRepresentation):
        kernel, support = build_chunk_kernel(task, representation)
    elif isinstance(representation, LandmarkRepresentation):
        kernel, support = build_landmark_kernel(task, representation, config)
    else:
        raise TypeError(f"unsupported representation type: {type(representation)!r}")
    validate_kernel(kernel, config)
    validate_kernel_support(kernel, support, config)
    return kernel, support


def build_primitive_kernel(
    task: TaskGraph,
    representation: PrimitiveRepresentation,
) -> tuple[TransitionKernel, KernelSupport]:
    """构造 primitive kernel。

    每个合法 primitive action 贡献其权重到对应 endpoint。
    如果多个 action 到达同一个 target，它们的概率在同一个 kernel edge 上相加。
    """

    matrix = _zero_matrix(len(task.states))
    support_entries: dict[tuple[StateId, StateId], list[KernelSupportEntry]] = defaultdict(list)

    for source in task.states:
        source_index = task.state_index[source]
        for action in task.legal_actions(source):
            target = task.next_state(source, action)
            probability = representation.weight(source, action)
            if probability <= 0:
                continue
            target_index = task.state_index[target]
            # kernel 只聚合 endpoint 概率；具体 action 来源写入 support。
            matrix[source_index][target_index] += probability
            trace = task.execute_token(source, PrimitiveToken(action))
            _append_support(
                support_entries,
                source=source,
                target=target,
                kind=SupportKind.PRIMITIVE_ACTION,
                item_id=f"primitive:{action}",
                probability=probability,
                endpoint=target,
                action_sequence=trace.actions,
                state_sequence=trace.states,
                cpd_block=None,
            )

    return _make_kernel(Family.PRIMITIVE, task, matrix, support_entries)


def build_chunk_kernel(
    task: TaskGraph,
    representation: ChunkRepresentation,
) -> tuple[TransitionKernel, KernelSupport]:
    """构造 chunk kernel。

    候选 token 是 primitive tokens 加上从 source 可完整执行的 chunk tokens。
    每个 token 先执行成 primitive trace，再按 trace endpoint 汇总概率。
    """

    matrix = _zero_matrix(len(task.states))
    support_entries: dict[tuple[StateId, StateId], list[KernelSupportEntry]] = defaultdict(list)

    for source in task.states:
        source_index = task.state_index[source]
        for token in representation.tokens_at(source, task):
            trace = task.execute_token(source, token)
            target = trace.endpoint
            probability = representation.weight(source, token)
            if probability <= 0:
                continue
            # chunk 的 cognitive transition 可以跨多个 primitive step。
            matrix[source_index][task.state_index[target]] += probability
            kind = SupportKind.PRIMITIVE_ACTION if token.kind == "primitive" else SupportKind.CHUNK_TOKEN
            _append_support(
                support_entries,
                source=source,
                target=target,
                kind=kind,
                item_id=token.token_id,
                probability=probability,
                endpoint=target,
                action_sequence=trace.actions,
                state_sequence=trace.states,
                cpd_block=None,
            )

    return _make_kernel(Family.CHUNK, task, matrix, support_entries)


def build_landmark_kernel(
    task: TaskGraph,
    representation: LandmarkRepresentation,
    config: PlanningConfig | None = None,
) -> tuple[TransitionKernel, KernelSupport]:
    """从 LL/LU/UL 三类 CPD block 构造 landmark kernel。

    landmark 表征在 kernel 阶段只给出 cognitive transition 概率；
    对应的 primitive shortest path 在 expansion 阶段按 task 图生成。
    """

    config = config or PlanningConfig()
    matrix = _zero_matrix(len(task.states))
    support_entries: dict[tuple[StateId, StateId], list[KernelSupportEntry]] = defaultdict(list)

    for source in task.states:
        source_index = task.state_index[source]
        for target in task.states:
            block, probability = representation.block_for(source, target, task)
            matrix[source_index][task.state_index[target]] = probability
            if probability > config.probability_epsilon:
                _append_support(
                    support_entries,
                    source=source,
                    target=target,
                    kind=SupportKind.CPD_ENTRY,
                    item_id=(
                        f"{block.value}:{source}->{target}"
                        if isinstance(block, CPDBlock)
                        else f"cpd:{source}->{target}"
                    ),
                    probability=probability,
                    endpoint=target,
                    action_sequence=(),
                    state_sequence=(),
                    cpd_block=block,
                )

    return _make_kernel(Family.LANDMARK, task, matrix, support_entries)


def validate_kernel(kernel: TransitionKernel, config: PlanningConfig | None = None) -> None:
    """校验 transition kernel 的矩阵形状、概率非负有限和行归一化。"""

    config = config or PlanningConfig()
    n_states = len(kernel.states)
    if len(kernel.matrix) != n_states:
        raise InvalidKernelError("kernel matrix row count must match states")
    for row_index, row in enumerate(kernel.matrix):
        if len(row) != n_states:
            raise InvalidKernelError("kernel matrix must be square", context={"row_index": row_index})
        if any(value < -config.probability_epsilon or not math.isfinite(value) for value in row):
            raise InvalidKernelError("kernel probabilities must be non-negative finite values")
        row_sum = sum(row)
        if abs(row_sum - 1.0) > config.row_sum_tolerance:
            raise InvalidKernelError("kernel rows must sum to 1", context={"row_index": row_index, "row_sum": row_sum})


def validate_kernel_support(
    kernel: TransitionKernel,
    support: KernelSupport,
    config: PlanningConfig | None = None,
) -> None:
    """校验 KernelSupport 是否和 kernel 的非零 edge 一致。

    每个非零 edge 必须有 support；零概率 edge 不应有 support；
    同一 edge 下 support probability 之和必须等于 kernel edge probability。
    """

    config = config or PlanningConfig()
    for source_index, source in enumerate(kernel.states):
        for target_index, target in enumerate(kernel.states):
            probability = kernel.matrix[source_index][target_index]
            entries = support.entries_for(source, target)
            if probability > config.probability_epsilon:
                if not entries:
                    raise InvalidKernelError(
                        "nonzero kernel edge is missing support", context={"source": source, "target": target}
                    )
                support_sum = sum(entry.probability for entry in entries)
                if abs(support_sum - probability) > config.row_sum_tolerance:
                    raise InvalidKernelError(
                        "support probabilities must sum to edge probability",
                        context={
                            "source": source,
                            "target": target,
                            "support_sum": support_sum,
                            "probability": probability,
                        },
                    )
            elif entries:
                raise InvalidKernelError(
                    "zero kernel edge should not have support", context={"source": source, "target": target}
                )
            for entry in entries:
                if entry.source != source or entry.target != target:
                    raise InvalidKernelError(
                        "support entry source/target must match edge key",
                        context={
                            "source": source,
                            "target": target,
                            "entry_source": entry.source,
                            "entry_target": entry.target,
                        },
                    )
                if entry.probability < -config.probability_epsilon or not math.isfinite(entry.probability):
                    raise InvalidKernelError(
                        "support probabilities must be non-negative finite values",
                        context={"source": source, "target": target, "probability": entry.probability},
                    )


def _zero_matrix(size: int) -> list[list[float]]:
    """创建 size x size 的零矩阵。"""

    return [[0.0 for _ in range(size)] for _ in range(size)]


def _append_support(
    support_entries: dict[tuple[StateId, StateId], list[KernelSupportEntry]],
    *,
    source: StateId,
    target: StateId,
    kind: SupportKind,
    item_id: str,
    probability: float,
    endpoint: StateId,
    action_sequence: tuple[str, ...],
    state_sequence: tuple[StateId, ...],
    cpd_block: CPDBlock | None,
) -> None:
    """向某条 edge 追加一条概率来源记录。"""

    edge = (source, target)
    support_entries[edge].append(
        KernelSupportEntry(
            source=source,
            target=target,
            kind=kind,
            item_id=item_id,
            probability=probability,
            endpoint=endpoint,
            action_sequence=action_sequence,
            state_sequence=state_sequence,
            cpd_block=cpd_block,
        )
    )


def _make_kernel(
    family: Family,
    task: TaskGraph,
    matrix: list[list[float]],
    support_entries: dict[tuple[StateId, StateId], list[KernelSupportEntry]],
) -> tuple[TransitionKernel, KernelSupport]:
    """把累计矩阵和 support entries 打包成并列 artifact。"""

    support = KernelSupport(entries_by_edge={edge: tuple(entries) for edge, entries in support_entries.items()})
    kernel = TransitionKernel(
        family=family,
        states=task.states,
        matrix=tuple(tuple(row) for row in matrix),
        row_index=task.state_index,
    )
    return kernel, support
