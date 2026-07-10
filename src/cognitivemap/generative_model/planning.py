"""认知图 A* 搜索与 cognitive edge 展开。

planning 阶段先在 cognitive graph 上搜索状态级路径，再按 representation family
把每条 cognitive edge 展开成 primitive action/state 轨迹。
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from cognitivemap.generative_model.representations import (
    ChunkRepresentation,
    LandmarkRepresentation,
    PrimitiveRepresentation,
    Representation,
)
from cognitivemap.generative_model.task import TaskGraph
from cognitivemap.generative_model.types import (
    AStarResult,
    AStarStatus,
    CognitiveGraph,
    Embedding,
    ExpansionCandidate,
    ExpansionError,
    ExpansionResult,
    ExpansionStatus,
    ExpansionTrace,
    Family,
    KernelSupport,
    PlanningConfig,
    PlanningError,
    StateId,
    TerminationRecord,
    TokenTrace,
)


def plan_cognitive_path(
    graph: CognitiveGraph,
    embedding: Embedding,
    start: StateId,
    goal: StateId,
    config: PlanningConfig | None = None,
) -> AStarResult:
    """在有向 cognitive cost graph 上执行 A* 搜索。

    ``alpha=0`` 时 heuristic 恒为 0，算法退化为 Dijkstra。
    tie-breaking 固定使用 ``(f, h, g, state_order, push_seq)``，
    这样相同输入和相同 graph 下路径选择可复现。
    """

    config = config or PlanningConfig()
    state_order = {state: index for index, state in enumerate(graph.states)}
    if start not in state_order:
        raise PlanningError("A* start state is not in cognitive graph", context={"start": start})
    if goal not in state_order:
        raise PlanningError("A* goal state is not in cognitive graph", context={"goal": goal})
    if start == goal:
        return AStarResult(
            status=AStarStatus.FOUND,
            start_state=start,
            goal_state=goal,
            cognitive_path=(start,),
            path_cost=0.0,
            expanded_nodes=(),
            visited_count=0,
            tie_breaking=config.tie_breaking,
        )

    edge_lookup = graph.edge_lookup
    best_g: dict[StateId, float] = {start: 0.0}
    predecessor: dict[StateId, StateId] = {}
    expanded_nodes: list[StateId] = []
    push_seq = 0

    h_start = _heuristic(embedding, start, goal, config)
    open_heap: list[tuple[float, float, float, int, int, StateId]] = [
        (h_start, h_start, 0.0, state_order[start], push_seq, start)
    ]

    while open_heap:
        _, _, current_g, _, _, state = heapq.heappop(open_heap)
        if current_g > best_g.get(state, math.inf) + config.row_sum_tolerance:
            continue
        if state == goal:
            return AStarResult(
                status=AStarStatus.FOUND,
                start_state=start,
                goal_state=goal,
                cognitive_path=tuple(_reconstruct_path(predecessor, start, goal)),
                path_cost=current_g,
                expanded_nodes=tuple(expanded_nodes),
                visited_count=len(expanded_nodes),
                tie_breaking=config.tie_breaking,
            )

        expanded_nodes.append(state)
        for next_state in graph.successors.get(state, ()):
            edge = edge_lookup[(state, next_state)]
            new_g = best_g[state] + edge.cost
            # 数值相等时保留先进入的 predecessor，保证 deterministic tie-breaking。
            if new_g < best_g.get(next_state, math.inf) - config.row_sum_tolerance:
                best_g[next_state] = new_g
                predecessor[next_state] = state
                h_value = _heuristic(embedding, next_state, goal, config)
                push_seq += 1
                heapq.heappush(
                    open_heap,
                    (
                        new_g + h_value,
                        h_value,
                        new_g,
                        state_order[next_state],
                        push_seq,
                        next_state,
                    ),
                )

    return AStarResult(
        status=AStarStatus.NO_PATH,
        start_state=start,
        goal_state=goal,
        cognitive_path=(),
        path_cost=None,
        expanded_nodes=tuple(expanded_nodes),
        visited_count=len(expanded_nodes),
        tie_breaking=config.tie_breaking,
    )


def expand_cognitive_path(
    task: TaskGraph,
    representation: Representation,
    support: KernelSupport,
    cognitive_path: tuple[StateId, ...] | list[StateId],
    goal: StateId,
    rng: np.random.Generator,
    config: PlanningConfig | None = None,
) -> ExpansionResult:
    """把 cognitive path 展开为 primitive actions 和 states。

    每条 cognitive edge 独立展开；展开过程中如果某个 primitive state 已到达 goal，
    立即停止后续 edge 展开，并记录停止位置。
    """

    config = config or PlanningConfig()
    cognitive_path = tuple(cognitive_path)
    if not cognitive_path:
        raise ExpansionError("cannot expand an empty cognitive path")

    actions: list[str] = []
    states: list[StateId] = [cognitive_path[0]]
    edge_expansions: list[ExpansionTrace] = []

    if cognitive_path[0] == goal:
        return ExpansionResult(
            status=ExpansionStatus.REACHED_GOAL,
            cognitive_path=cognitive_path,
            edge_expansions=(),
            actions=(),
            states=tuple(states),
            termination=TerminationRecord(reason="goal_reached", edge_index=None, primitive_step_index=None),
        )

    for edge_index, (source, target) in enumerate(zip(cognitive_path[:-1], cognitive_path[1:])):
        trace = _expand_edge(
            task=task,
            representation=representation,
            support=support,
            source=source,
            target=target,
            edge_index=edge_index,
            rng=rng,
            config=config,
        )
        reached_goal_at_step = None
        # 按 primitive step 追加轨迹，这样可以在 chunk 或 landmark 中途命中 goal 时提前终止。
        for offset, (action, next_state) in enumerate(zip(trace.actions, trace.states[1:])):
            actions.append(action)
            states.append(next_state)
            if next_state == goal:
                reached_goal_at_step = len(actions) - 1
                break

        if reached_goal_at_step is not None:
            trace = _replace_trace_goal_step(trace, reached_goal_at_step)
            edge_expansions.append(trace)
            return ExpansionResult(
                status=ExpansionStatus.REACHED_GOAL,
                cognitive_path=cognitive_path,
                edge_expansions=tuple(edge_expansions),
                actions=tuple(actions),
                states=tuple(states),
                termination=TerminationRecord(
                    reason="goal_reached",
                    edge_index=edge_index,
                    primitive_step_index=reached_goal_at_step,
                ),
            )

        if not trace.actions and trace.states[-1] == goal:
            trace = _replace_trace_goal_step(trace, None)
            edge_expansions.append(trace)
            return ExpansionResult(
                status=ExpansionStatus.REACHED_GOAL,
                cognitive_path=cognitive_path,
                edge_expansions=tuple(edge_expansions),
                actions=tuple(actions),
                states=tuple(states),
                termination=TerminationRecord(reason="goal_reached", edge_index=edge_index, primitive_step_index=None),
            )

        if trace.actions:
            # 非空 trace 的 primitive states 已在上面的逐步追加循环中写入。
            pass
        elif trace.states[-1] != states[-1]:
            states.append(trace.states[-1])

        edge_expansions.append(trace)

    return ExpansionResult(
        status=ExpansionStatus.REACHED_GOAL,
        cognitive_path=cognitive_path,
        edge_expansions=tuple(edge_expansions),
        actions=tuple(actions),
        states=tuple(states),
        termination=TerminationRecord(
            reason="goal_reached" if states[-1] == goal else "path_exhausted",
            edge_index=len(edge_expansions) - 1 if edge_expansions else None,
            primitive_step_index=len(actions) - 1 if states[-1] == goal and actions else None,
        ),
    )


def _expand_edge(
    *,
    task: TaskGraph,
    representation: Representation,
    support: KernelSupport,
    source: StateId,
    target: StateId,
    edge_index: int,
    rng: np.random.Generator,
    config: PlanningConfig,
) -> ExpansionTrace:
    """按 representation family 生成某条 cognitive edge 的展开候选并采样。"""

    if isinstance(representation, PrimitiveRepresentation):
        candidates = _primitive_expansion_candidates(support, source, target)
        family = Family.PRIMITIVE
    elif isinstance(representation, ChunkRepresentation):
        candidates = _chunk_expansion_candidates(support, source, target)
        family = Family.CHUNK
    elif isinstance(representation, LandmarkRepresentation):
        candidates = _landmark_expansion_candidates(task, representation, source, target, config)
        family = Family.LANDMARK
    else:
        raise ExpansionError(f"unsupported representation type: {type(representation)!r}")

    if not candidates:
        raise ExpansionError("cognitive edge has no expansion candidates", context={"source": source, "target": target})

    probabilities = _normalize([candidate.probability for candidate in candidates])
    sampled_index = int(rng.choice(len(candidates), p=np.asarray(probabilities, dtype=float)))
    sampled = candidates[sampled_index]
    return ExpansionTrace(
        edge_index=edge_index,
        source=source,
        target=target,
        family=family,
        candidates=tuple(candidates),
        candidate_probabilities=tuple(probabilities),
        sampled_index=sampled_index,
        sampled_item_id=sampled.item_id,
        actions=sampled.actions,
        states=sampled.states,
        reached_goal_at_step=None,
    )


def _primitive_expansion_candidates(
    support: KernelSupport,
    source: StateId,
    target: StateId,
) -> list[ExpansionCandidate]:
    """从 support 中取出 primitive action 候选。"""

    return [
        ExpansionCandidate(
            item_id=entry.item_id,
            probability=entry.probability,
            actions=entry.action_sequence,
            states=entry.state_sequence,
        )
        for entry in support.entries_for(source, target)
        if entry.kind.value == "primitive_action"
    ]


def _chunk_expansion_candidates(
    support: KernelSupport,
    source: StateId,
    target: StateId,
) -> list[ExpansionCandidate]:
    """从 support 中取出 primitive token 和 chunk token 候选。"""

    return [
        ExpansionCandidate(
            item_id=entry.item_id,
            probability=entry.probability,
            actions=entry.action_sequence,
            states=entry.state_sequence,
        )
        for entry in support.entries_for(source, target)
        if entry.kind.value in {"primitive_action", "chunk_token"}
    ]


def _landmark_expansion_candidates(
    task: TaskGraph,
    representation: LandmarkRepresentation,
    source: StateId,
    target: StateId,
    config: PlanningConfig,
) -> list[ExpansionCandidate]:
    """为 landmark edge 枚举 primitive 展开候选。

    landmark kernel 阶段不保存 primitive trace，所以这里从 TaskGraph 重新枚举。
    LL edge 使用所有不重复 state 的 simple paths，并按路径长度加权采样；
    LU/UL edge 保持最短路径展开，避免非 landmark 出入口产生过多绕路。
    """

    landmark_set = set(representation.landmarks)
    is_ll_edge = source in landmark_set and target in landmark_set and source != target
    if is_ll_edge:
        paths = task.all_simple_paths(source, target, limit=config.max_landmark_simple_paths)
        probabilities = _landmark_path_length_probabilities(task, paths, config.landmark_path_length_beta)
    else:
        paths = task.all_shortest_paths(source, target, limit=config.max_landmark_shortest_paths)
        probabilities = [1.0 / len(paths)] * len(paths) if paths else []

    if not paths:
        raise ExpansionError(
            "landmark cognitive edge is unreachable in primitive task graph",
            context={"source": source, "target": target},
        )
    return [
        ExpansionCandidate(
            item_id=f"path:{index}",
            probability=probability,
            actions=path.actions,
            states=path.states,
        )
        for index, (path, probability) in enumerate(zip(paths, probabilities))
    ]


def _landmark_path_length_probabilities(
    task: TaskGraph,
    paths: list[TokenTrace],
    beta: float,
) -> list[float]:
    """把 simple path 长度转换为采样概率，路径越短概率越高。"""

    if not paths:
        return []
    costs = np.asarray([_trace_cost(task, path) for path in paths], dtype=float)
    logits = -float(beta) * costs
    logits = logits - logits.max()
    weights = np.exp(logits)
    return _normalize(weights.tolist())


def _trace_cost(task: TaskGraph, trace: TokenTrace) -> float:
    """按 task 的 primitive action cost 累加一条 trace 的总代价。"""

    cost = 0.0
    for state, action in zip(trace.states, trace.actions):
        cost += task.action_cost(state, action)
    return cost


def _normalize(weights: list[float]) -> list[float]:
    """把候选权重归一化成采样概率。"""

    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        raise ExpansionError("expansion candidate probabilities are not normalizable")
    return [weight / total for weight in weights]


def _heuristic(embedding: Embedding, state: StateId, goal: StateId, config: PlanningConfig) -> float:
    """计算 A* heuristic：embedding 空间欧氏距离乘以 alpha。"""

    if config.alpha == 0.0:
        return 0.0
    x1, y1 = embedding.coordinates[state]
    x2, y2 = embedding.coordinates[goal]
    return config.alpha * math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _reconstruct_path(predecessor: dict[StateId, StateId], start: StateId, goal: StateId) -> list[StateId]:
    """根据 predecessor 表回溯 cognitive path。"""

    path = [goal]
    current = goal
    while current != start:
        current = predecessor[current]
        path.append(current)
    return list(reversed(path))


def _replace_trace_goal_step(trace: ExpansionTrace, reached_goal_at_step: int | None) -> ExpansionTrace:
    """返回一份只更新 goal 命中位置的 ExpansionTrace。"""

    return ExpansionTrace(
        edge_index=trace.edge_index,
        source=trace.source,
        target=trace.target,
        family=trace.family,
        candidates=trace.candidates,
        candidate_probabilities=trace.candidate_probabilities,
        sampled_index=trace.sampled_index,
        sampled_item_id=trace.sampled_item_id,
        actions=trace.actions,
        states=trace.states,
        reached_goal_at_step=reached_goal_at_step,
    )
