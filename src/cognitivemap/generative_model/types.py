"""认知导航生成模型的公共类型定义。

本文件只放跨模块共享的数据结构、枚举和异常类型，不实现具体算法。
这样做的目的是让 task、representation、kernel、graph、planning 和 agent
都依赖同一套稳定的语义边界，避免各模块重复定义字段含义。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, TypeAlias

StateId: TypeAlias = int | str
ActionId: TypeAlias = str
Probability: TypeAlias = float


class Family(str, Enum):
    """第一版支持的三类认知表征家族。"""

    PRIMITIVE = "primitive"
    CHUNK = "chunk"
    LANDMARK = "landmark"


class SupportKind(str, Enum):
    """KernelSupport 中每条概率来源记录的类型。"""

    PRIMITIVE_ACTION = "primitive_action"
    CHUNK_TOKEN = "chunk_token"
    CPD_ENTRY = "cpd_entry"


class CPDBlock(str, Enum):
    """Landmark 表征中三类条件概率块的标签。"""

    LL = "LL"
    LU = "LU"
    UL = "UL"


class RunStatus(str, Enum):
    """单个 trial 的高层运行状态。"""

    REACHED_GOAL = "reached_goal"
    NO_PATH = "no_path"
    VALIDATION_ERROR = "validation_error"
    EMBEDDING_ERROR = "embedding_error"
    EXPANSION_ERROR = "expansion_error"


class AStarStatus(str, Enum):
    """认知图上 A* 搜索的状态。"""

    FOUND = "found"
    NO_PATH = "no_path"


class ExpansionStatus(str, Enum):
    """认知路径展开为 primitive 轨迹后的状态。"""

    REACHED_GOAL = "reached_goal"
    EXPANSION_ERROR = "expansion_error"


class ErrorStage(str, Enum):
    """错误发生的处理阶段，用于失败结果的可诊断输出。"""

    VALIDATION = "validation"
    KERNEL = "kernel"
    GRAPH = "graph"
    DISTANCE = "distance"
    EMBEDDING = "embedding"
    PLANNING = "planning"
    EXPANSION = "expansion"
    SERIALIZATION = "serialization"


class CognitiveNavigationError(Exception):
    """认知导航模块的可识别错误基类。

    所有高层入口只捕获这一类错误并写入 ``RunResult.error``。
    普通 Python 错误仍然向外抛出，避免把实现 bug 伪装成模型输入错误。
    """

    stage = ErrorStage.VALIDATION
    code = "generative_model_error"

    def __init__(
        self,
        message: str,
        *,
        object_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """保存错误消息、对象标识和结构化上下文。"""

        super().__init__(message)
        self.message = message
        self.object_id = object_id
        self.context = dict(context or {})

    def to_detail(self) -> "ErrorDetail":
        """转换为可序列化的错误详情对象。"""

        return ErrorDetail(
            code=self.code,
            message=self.message,
            stage=self.stage,
            object_id=self.object_id,
            context=self.context,
        )


class InvalidInputError(CognitiveNavigationError):
    """输入校验错误的基类。"""

    code = "invalid_input"


class InvalidTaskGraphError(InvalidInputError):
    """TaskGraph 不满足确定性 primitive 任务图不变量时抛出。"""

    code = "invalid_task_graph"


class InvalidRepresentationError(InvalidInputError):
    """表征校验错误的基类。"""

    code = "invalid_representation"


class InvalidPrimitiveRepresentationError(InvalidRepresentationError):
    """primitive action 权重无效时抛出。"""

    code = "invalid_primitive_representation"


class InvalidChunkRepresentationError(InvalidRepresentationError):
    """chunk inventory 或 token 权重无效时抛出。"""

    code = "invalid_chunk_representation"


class InvalidLandmarkRepresentationError(InvalidRepresentationError):
    """landmark 集合或 CPD block 无效时抛出。"""

    code = "invalid_landmark_representation"


class InvalidTrialSpecError(InvalidInputError):
    """trial 的 start/goal/family 等元信息无效时抛出。"""

    code = "invalid_trial_spec"


class InvalidKernelError(CognitiveNavigationError):
    """transition kernel 或 kernel support 违反不变量时抛出。"""

    stage = ErrorStage.KERNEL
    code = "invalid_kernel"


class EmbeddingError(CognitiveNavigationError):
    """距离矩阵无法安全执行 MDS embedding 时抛出。"""

    stage = ErrorStage.EMBEDDING
    code = "embedding_error"


class PlanningError(CognitiveNavigationError):
    """A* 搜索或 trial 运行输入无效时抛出。"""

    stage = ErrorStage.PLANNING
    code = "planning_error"


class ExpansionError(CognitiveNavigationError):
    """cognitive edge 无法展开成 primitive 轨迹时抛出。"""

    stage = ErrorStage.EXPANSION
    code = "expansion_error"


class SerializationError(CognitiveNavigationError):
    """结果无法序列化时抛出。"""

    stage = ErrorStage.SERIALIZATION
    code = "serialization_error"


@dataclass(frozen=True)
class PrimitiveToken:
    """primitive 表征中的单步动作 token。

    ``token_id`` 使用稳定字符串 ``primitive:<action>``，用于 chunk 表征中和
    chunk token 放在同一个候选集合里。
    """

    action: ActionId
    kind: str = field(init=False, default="primitive")

    @property
    def token_id(self) -> str:
        """返回 primitive token 的稳定标识。"""

        return f"primitive:{self.action}"


@dataclass(frozen=True)
class ChunkToken:
    """chunk 表征中的动作序列 token。

    chunk 必须至少包含一个 primitive action；是否能从某个 state 完整执行，
    由 ``TaskGraph.execute_token`` 在具体 state 上判断。
    """

    chunk_id: str
    actions: tuple[ActionId, ...]
    kind: str = field(init=False, default="chunk")

    def __post_init__(self) -> None:
        """规范化 action 序列，并拒绝空 chunk。"""

        object.__setattr__(self, "actions", tuple(self.actions))
        if not self.actions:
            raise InvalidChunkRepresentationError(f"chunk {self.chunk_id!r} has empty action sequence")

    @property
    def token_id(self) -> str:
        """返回 chunk token 的稳定标识。"""

        return f"chunk:{self.chunk_id}"


Token: TypeAlias = PrimitiveToken | ChunkToken


@dataclass(frozen=True)
class TokenTrace:
    """执行 token 或最短路径后得到的 primitive 轨迹。

    ``states`` 必须比 ``actions`` 多一个元素，因为它同时记录起点和每步动作后的状态。
    """

    actions: tuple[ActionId, ...]
    states: tuple[StateId, ...]

    def __post_init__(self) -> None:
        """规范化 trace 字段，并校验 state/action 长度关系。"""

        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "states", tuple(self.states))
        if len(self.states) != len(self.actions) + 1:
            raise ValueError("trace states length must equal actions length + 1")

    @property
    def endpoint(self) -> StateId:
        """返回 trace 的最终状态。"""

        return self.states[-1]


@dataclass(frozen=True)
class PlanningConfig:
    """几何构造、A* 搜索和路径展开共用的运行配置。

    第一版保持配置很小：不在这里放模型参数或表征参数，只控制数值容差、
    heuristic scale、landmark 展开上限和 MDS 维度。
    """

    alpha: float = 0.0
    probability_epsilon: float = 1e-12
    row_sum_tolerance: float = 1e-9
    tie_breaking: str = "deterministic_state_order"
    max_landmark_shortest_paths: int = 10000
    max_landmark_simple_paths: int = 10000
    landmark_path_length_beta: float = 1.0
    mds_dim: int = 2

    def __post_init__(self) -> None:
        """校验规划配置处于第一版支持的数值范围。"""

        if self.alpha < 0:
            raise InvalidInputError("alpha must be non-negative")
        if self.probability_epsilon < 0:
            raise InvalidInputError("probability_epsilon must be non-negative")
        if self.row_sum_tolerance <= 0:
            raise InvalidInputError("row_sum_tolerance must be positive")
        if self.max_landmark_shortest_paths <= 0:
            raise InvalidInputError("max_landmark_shortest_paths must be positive")
        if self.max_landmark_simple_paths <= 0:
            raise InvalidInputError("max_landmark_simple_paths must be positive")
        if self.landmark_path_length_beta < 0:
            raise InvalidInputError("landmark_path_length_beta must be non-negative")
        if self.mds_dim != 2:
            raise InvalidInputError("current implementation supports mds_dim=2")


@dataclass(frozen=True)
class TrialSpec:
    """单个导航 trial 的输入规格。

    ``family`` 必须和 agent 持有的 representation family 一致。
    ``random_seed`` 只控制该 trial 的展开采样，不参与 artifact 构造。
    """

    trial_id: str
    start_state: StateId
    goal_state: StateId
    family: Family | str
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """把字符串 family 规范化为枚举值。"""

        object.__setattr__(self, "family", self.family if isinstance(self.family, Family) else Family(self.family))


@dataclass(frozen=True)
class KernelSupportEntry:
    """某个 kernel edge 概率的一条来源记录。

    primitive/chunk 中它对应一个 action 或 token 对 edge 概率的贡献；
    landmark 中它对应一个 CPD entry。landmark 的 primitive action/state trace
    在 kernel 阶段为空，真正展开时才由 shortest path 生成。
    """

    source: StateId
    target: StateId
    kind: SupportKind | str
    item_id: str
    probability: float
    endpoint: StateId
    action_sequence: tuple[ActionId, ...] = ()
    state_sequence: tuple[StateId, ...] = ()
    cpd_block: CPDBlock | str | None = None

    def __post_init__(self) -> None:
        """规范化枚举字段和 tuple 字段，便于后续不可变访问。"""

        object.__setattr__(self, "kind", self.kind if isinstance(self.kind, SupportKind) else SupportKind(self.kind))
        if self.cpd_block is not None and not isinstance(self.cpd_block, CPDBlock):
            object.__setattr__(self, "cpd_block", CPDBlock(self.cpd_block))
        object.__setattr__(self, "action_sequence", tuple(self.action_sequence))
        object.__setattr__(self, "state_sequence", tuple(self.state_sequence))


@dataclass(frozen=True)
class KernelSupport:
    """按 ``(source, target)`` 分组的 kernel 概率来源。

    设计上它和 ``TransitionKernel`` 并列存在，而不是嵌入 kernel。
    核心图算法只需要 kernel；诊断、展开和序列化再查询 support。
    """

    entries_by_edge: Mapping[tuple[StateId, StateId], tuple[KernelSupportEntry, ...]]

    def __post_init__(self) -> None:
        """将每条 edge 的来源列表固定为 tuple。"""

        normalized = {edge: tuple(entries) for edge, entries in self.entries_by_edge.items()}
        object.__setattr__(self, "entries_by_edge", normalized)

    def entries_for(self, source: StateId, target: StateId) -> tuple[KernelSupportEntry, ...]:
        """查询某条 cognitive edge 对应的全部概率来源记录。"""

        return self.entries_by_edge.get((source, target), ())


@dataclass(frozen=True)
class TransitionKernel:
    """由某种认知表征诱导出的状态转移概率矩阵。

    每一行对应一个 source state，必须在容差内归一化为 1。
    该结构只保存概率矩阵本身，不保存 support 或 provenance 字段。
    """

    family: Family | str
    states: tuple[StateId, ...]
    matrix: tuple[tuple[float, ...], ...]
    row_index: Mapping[StateId, int]

    def __post_init__(self) -> None:
        """规范化 kernel family、状态序列、矩阵和索引。"""

        object.__setattr__(self, "family", self.family if isinstance(self.family, Family) else Family(self.family))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "matrix", tuple(tuple(float(value) for value in row) for row in self.matrix))
        object.__setattr__(self, "row_index", dict(self.row_index))

    def probability(self, source: StateId, target: StateId) -> float:
        """返回 source 到 target 的 kernel 转移概率。"""

        return self.matrix[self.row_index[source]][self.row_index[target]]


@dataclass(frozen=True)
class CognitiveEdge:
    """认知图中的有向边。

    第一版只保留核心图搜索需要的字段：概率和 ``-log(probability)`` cost。
    概率来源统一通过 ``KernelSupport`` 查询，避免 edge 结构过重。
    """

    source: StateId
    target: StateId
    probability: float
    cost: float


@dataclass(frozen=True)
class CognitiveGraph:
    """由 transition kernel 过滤非零概率后得到的认知 cost 图。"""

    family: Family | str
    states: tuple[StateId, ...]
    edges: tuple[CognitiveEdge, ...]
    successors: Mapping[StateId, tuple[StateId, ...]]

    def __post_init__(self) -> None:
        """规范化图的 family、edge 列表和 successor 列表。"""

        object.__setattr__(self, "family", self.family if isinstance(self.family, Family) else Family(self.family))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "successors", {state: tuple(values) for state, values in self.successors.items()})

    @property
    def edge_lookup(self) -> dict[tuple[StateId, StateId], CognitiveEdge]:
        """构造 ``(source, target) -> edge`` 的快速查询表。"""

        return {(edge.source, edge.target): edge for edge in self.edges}

    def edge_cost(self, source: StateId, target: StateId) -> float:
        """返回 cognitive edge 的搜索 cost。"""

        return self.edge_lookup[(source, target)].cost


@dataclass(frozen=True)
class DirectedDistance:
    """认知图上的全源最短有向距离矩阵。"""

    states: tuple[StateId, ...]
    matrix: tuple[tuple[float, ...], ...]
    unreachable_pairs: tuple[tuple[StateId, StateId], ...]


@dataclass(frozen=True)
class SymmetricDistance:
    """由有向距离对称化得到的 MDS 输入矩阵。"""

    states: tuple[StateId, ...]
    matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class Embedding:
    """classical MDS 生成的二维状态嵌入。"""

    states: tuple[StateId, ...]
    coordinates: Mapping[StateId, tuple[float, float]]
    eigenvalues: tuple[float, ...]
    status: str = "ready"

    def __post_init__(self) -> None:
        """规范化 embedding 坐标和特征值为普通容器。"""

        object.__setattr__(self, "coordinates", dict(self.coordinates))
        object.__setattr__(self, "eigenvalues", tuple(float(value) for value in self.eigenvalues))


@dataclass(frozen=True)
class ModelArtifacts:
    """把某个 representation 应用于某个 task 后得到的可复用中间产物。

    artifact 不属于单个 trial；同一 task 和同一 representation 下，
    多个 trial 可以复用 kernel、support、认知图、距离矩阵和 embedding。
    """

    task_id: str
    representation_id: str
    family: Family | str
    transition_kernel: TransitionKernel
    kernel_support: KernelSupport
    cognitive_graph: CognitiveGraph
    directed_distance: DirectedDistance
    symmetric_distance: SymmetricDistance
    embedding: Embedding

    def __post_init__(self) -> None:
        """把 artifact family 统一为枚举值。"""

        object.__setattr__(self, "family", self.family if isinstance(self.family, Family) else Family(self.family))


@dataclass(frozen=True)
class AStarResult:
    """认知图上的 A* 搜索结果。"""

    status: AStarStatus | str
    start_state: StateId
    goal_state: StateId
    cognitive_path: tuple[StateId, ...]
    path_cost: float | None
    expanded_nodes: tuple[StateId, ...]
    visited_count: int
    tie_breaking: str

    def __post_init__(self) -> None:
        """规范化 A* 状态和路径字段。"""

        object.__setattr__(
            self, "status", self.status if isinstance(self.status, AStarStatus) else AStarStatus(self.status)
        )
        object.__setattr__(self, "cognitive_path", tuple(self.cognitive_path))
        object.__setattr__(self, "expanded_nodes", tuple(self.expanded_nodes))


@dataclass(frozen=True)
class ExpansionCandidate:
    """展开某条 cognitive edge 时的候选 primitive 轨迹。"""

    item_id: str
    probability: float
    actions: tuple[ActionId, ...]
    states: tuple[StateId, ...]

    def __post_init__(self) -> None:
        """规范化候选展开的 action/state 序列。"""

        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "states", tuple(self.states))


@dataclass(frozen=True)
class ExpansionTrace:
    """一条 cognitive edge 的展开过程记录。

    该结构保留候选集合、候选概率、采样索引和最终采样项，
    使随机展开可以通过 seed 和候选顺序复现。
    """

    edge_index: int
    source: StateId
    target: StateId
    family: Family | str
    candidates: tuple[ExpansionCandidate, ...]
    candidate_probabilities: tuple[float, ...]
    sampled_index: int
    sampled_item_id: str
    actions: tuple[ActionId, ...]
    states: tuple[StateId, ...]
    reached_goal_at_step: int | None

    def __post_init__(self) -> None:
        """规范化 edge 展开记录中的枚举、候选和轨迹字段。"""

        object.__setattr__(self, "family", self.family if isinstance(self.family, Family) else Family(self.family))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(
            self,
            "candidate_probabilities",
            tuple(float(value) for value in self.candidate_probabilities),
        )
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "states", tuple(self.states))


@dataclass(frozen=True)
class TerminationRecord:
    """primitive 展开停止的位置和原因。"""

    reason: str
    edge_index: int | None
    primitive_step_index: int | None


@dataclass(frozen=True)
class ExpansionResult:
    """将完整 cognitive path 展开为 primitive actions/states 后的结果。"""

    status: ExpansionStatus | str
    cognitive_path: tuple[StateId, ...]
    edge_expansions: tuple[ExpansionTrace, ...]
    actions: tuple[ActionId, ...]
    states: tuple[StateId, ...]
    termination: TerminationRecord

    def __post_init__(self) -> None:
        """规范化展开状态和所有序列字段。"""

        status = self.status if isinstance(self.status, ExpansionStatus) else ExpansionStatus(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "cognitive_path", tuple(self.cognitive_path))
        object.__setattr__(self, "edge_expansions", tuple(self.edge_expansions))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "states", tuple(self.states))


@dataclass(frozen=True)
class ErrorDetail:
    """可序列化的错误详情。"""

    code: str
    message: str
    stage: ErrorStage | str
    object_id: str | None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """规范化错误阶段和上下文字典。"""

        object.__setattr__(self, "stage", self.stage if isinstance(self.stage, ErrorStage) else ErrorStage(self.stage))
        object.__setattr__(self, "context", dict(self.context))


@dataclass(frozen=True)
class RunMetadata:
    """运行元信息，目前主要记录 trial seed 和配置。"""

    seed: int | None
    config: PlanningConfig


@dataclass(frozen=True)
class RunResult:
    """单个 trial 的完整运行结果。"""

    status: RunStatus | str
    task_id: str
    representation_id: str
    family: Family | str
    trial: TrialSpec
    artifacts: ModelArtifacts | None
    astar: AStarResult | None
    expansion: ExpansionResult | None
    metadata: RunMetadata
    error: ErrorDetail | None

    def __post_init__(self) -> None:
        """规范化运行状态和表征 family。"""

        object.__setattr__(
            self, "status", self.status if isinstance(self.status, RunStatus) else RunStatus(self.status)
        )
        object.__setattr__(self, "family", self.family if isinstance(self.family, Family) else Family(self.family))


def to_jsonable(value: Any) -> Any:
    """把 dataclass、Enum、tuple 和 mapping 递归转换为 JSON 兼容对象。"""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(to_jsonable(key)): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def to_json(value: Any, *, indent: int | None = 2) -> str:
    """把支持的对象序列化为 JSON 字符串。"""

    return json.dumps(to_jsonable(value), ensure_ascii=False, indent=indent)
