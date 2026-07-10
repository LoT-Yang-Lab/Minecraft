"""primitive 任务图与 primitive 最短路径工具。

TaskGraph 是外部环境的完整定义：状态、动作、合法动作、确定性转移和动作代价。
第一版不假设空间网格、方向语义或 region membership；任何 action label 都只通过
显式 transition 决定其效果。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

from cognitivemap.generative_model.types import (
    ActionId,
    ChunkToken,
    ExpansionError,
    InvalidTaskGraphError,
    PrimitiveToken,
    StateId,
    Token,
    TokenTrace,
)


@dataclass(frozen=True)
class TaskGraph:
    """确定性的 primitive 导航任务图。

    该类保存运行所需的完整结构。用户侧推荐使用 ``from_transitions`` 传入嵌套字典，
    由类方法派生 states、actions、legal actions 和默认动作代价。
    """

    task_id: str
    states: tuple[StateId, ...] | list[StateId]
    actions: tuple[ActionId, ...] | list[ActionId]
    legal_actions_by_state: Mapping[StateId, tuple[ActionId, ...] | list[ActionId]]
    transitions: Mapping[tuple[StateId, ActionId], StateId]
    action_costs: Mapping[tuple[StateId, ActionId], float]

    def __post_init__(self) -> None:
        """把输入容器规范化为不可变顺序，并立即校验任务图不变量。"""

        states = tuple(self.states)
        actions = tuple(self.actions)
        input_legal_actions = {state: tuple(values) for state, values in self.legal_actions_by_state.items()}
        legal_actions_by_state = {state: input_legal_actions.get(state, ()) for state in states}
        for state in input_legal_actions:
            if state not in legal_actions_by_state:
                legal_actions_by_state[state] = input_legal_actions[state]
        transitions = dict(self.transitions)
        action_costs = {key: float(value) for key, value in self.action_costs.items()}

        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "legal_actions_by_state", legal_actions_by_state)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "action_costs", action_costs)

        self._validate()

    @classmethod
    def from_transitions(
        cls,
        task_id: str,
        transitions: Mapping[StateId, Mapping[ActionId, StateId]],
        default_cost: float = 1.0,
        cost_overrides: Mapping[StateId, Mapping[ActionId, float]] | None = None,
    ) -> "TaskGraph":
        """从嵌套 transition 字典构造完整 TaskGraph。

        输入格式是 ``{source_state: {action: target_state}}``。
        states 和 actions 都按首次出现顺序保留，这保证候选顺序稳定且便于复现采样。
        """

        if not transitions:
            raise InvalidTaskGraphError("transitions must be non-empty", object_id=task_id)
        if not math.isfinite(default_cost) or default_cost <= 0:
            raise InvalidTaskGraphError("default_cost must be positive and finite", object_id=task_id)

        states: list[StateId] = []
        actions: list[ActionId] = []
        legal_actions_by_state: dict[StateId, tuple[ActionId, ...]] = {}
        flat_transitions: dict[tuple[StateId, ActionId], StateId] = {}
        action_costs: dict[tuple[StateId, ActionId], float] = {}
        overrides = cost_overrides or {}

        # 用局部 helper 保留“首次出现顺序”，避免 set 排序造成跨类型 state/action 比较问题。
        def add_state(state: StateId) -> None:
            """按首次出现顺序加入状态。"""

            if state not in states:
                states.append(state)

        def add_action(action: ActionId) -> None:
            """按首次出现顺序加入动作。"""

            if action not in actions:
                actions.append(action)

        # 遍历嵌套 transition，展开为内部使用的 ``(state, action) -> target`` 映射。
        for source, row in transitions.items():
            add_state(source)
            legal_row: list[ActionId] = []
            for action, target in row.items():
                add_action(action)
                add_state(target)
                legal_row.append(action)
                flat_transitions[(source, action)] = target
                override_row = overrides.get(source, {})
                action_costs[(source, action)] = float(override_row.get(action, default_cost))
            legal_actions_by_state[source] = tuple(legal_row)

        for state in states:
            legal_actions_by_state.setdefault(state, ())

        return cls(
            task_id=task_id,
            states=tuple(states),
            actions=tuple(actions),
            legal_actions_by_state=legal_actions_by_state,
            transitions=flat_transitions,
            action_costs=action_costs,
        )

    @property
    def state_index(self) -> dict[StateId, int]:
        """返回状态到稳定行号的映射。"""

        return {state: index for index, state in enumerate(self.states)}

    def legal_actions(self, state: StateId) -> list[ActionId]:
        """返回某个 state 的合法动作。

        返回顺序遵循全局 ``TaskGraph.actions`` 顺序，而 ``actions`` 来自首次出现顺序。
        这也是 primitive token candidate 的稳定顺序。
        """

        self._require_state(state)
        legal = set(self.legal_actions_by_state.get(state, ()))
        return [action for action in self.actions if action in legal]

    def actions_at(self, state: StateId) -> list[ActionId]:
        """``legal_actions`` 的兼容别名。"""

        return self.legal_actions(state)

    def next_state(self, state: StateId, action: ActionId) -> StateId:
        """返回合法 primitive action 的确定性目标状态。"""

        key = (state, action)
        if key not in self.transitions:
            raise KeyError(f"undefined transition for {(state, action)!r}")
        return self.transitions[key]

    def action_cost(self, state: StateId, action: ActionId) -> float:
        """返回 primitive action 的正有限代价。"""

        key = (state, action)
        if key not in self.action_costs:
            raise KeyError(f"undefined action cost for {(state, action)!r}")
        return self.action_costs[key]

    def execute_token(self, start: StateId, token: Token) -> TokenTrace:
        """从 ``start`` 开始执行 primitive 或 chunk token。

        primitive token 只执行一步；chunk token 逐个执行其 action sequence。
        任一步不合法都会抛出 ExpansionError，因此 chunk 的可执行性是 state-specific 的。
        """

        self._require_state(start)
        if isinstance(token, PrimitiveToken):
            actions = (token.action,)
        elif isinstance(token, ChunkToken):
            actions = token.actions
        else:
            raise TypeError(f"unsupported token type: {type(token)!r}")

        current = start
        states = [current]
        # 逐步检查合法性，而不是只检查 chunk 的最后 endpoint。
        for action in actions:
            if action not in self.legal_actions(current):
                raise ExpansionError(
                    f"token {getattr(token, 'token_id', token)!r} is not executable from {start!r}",
                    context={"state": current, "action": action},
                )
            current = self.next_state(current, action)
            states.append(current)
        return TokenTrace(actions=actions, states=tuple(states))

    def shortest_distance(self, source: StateId, target: StateId) -> float:
        """返回 primitive 图上的最短路径距离；不可达时返回 ``math.inf``。"""

        return self._dijkstra(source).get(target, math.inf)

    def all_shortest_paths(self, source: StateId, target: StateId, limit: int = 10000) -> list[TokenTrace]:
        """枚举两个 state 之间的所有 primitive 最短路径。

        对 ``source == target`` 返回零长度路径。其它情况先跑一次 Dijkstra 得到最短距离，
        再沿着满足最短路递推关系的边 DFS 枚举。最终按 state sequence 和 action sequence
        排序，保证 landmark 展开采样时的候选顺序可复现。
        """

        self._require_state(source)
        self._require_state(target)
        if limit <= 0:
            raise ExpansionError("shortest path enumeration limit must be positive")
        if source == target:
            return [TokenTrace(actions=(), states=(source,))]

        distances = self._dijkstra(source)
        target_distance = distances.get(target, math.inf)
        if not math.isfinite(target_distance):
            return []

        results: list[TokenTrace] = []

        def dfs(state: StateId, actions: list[ActionId], states: list[StateId]) -> None:
            """沿最短路递推关系递归枚举路径。"""

            if len(results) >= limit:
                raise ExpansionError(
                    "number of shortest paths exceeds configured limit",
                    context={"source": source, "target": target, "limit": limit},
                )
            if state == target:
                results.append(TokenTrace(actions=tuple(actions), states=tuple(states)))
                return

            current_distance = distances[state]
            for action in self.legal_actions(state):
                next_state = self.next_state(state, action)
                step_cost = self.action_cost(state, action)
                expected = current_distance + step_cost
                # 只沿着从 source 出发仍保持最短路距离的边继续 DFS。
                if abs(distances.get(next_state, math.inf) - expected) > 1e-9:
                    continue
                if expected - target_distance > 1e-9:
                    continue
                actions.append(action)
                states.append(next_state)
                dfs(next_state, actions, states)
                states.pop()
                actions.pop()

        dfs(source, [], [source])
        results.sort(key=lambda trace: (tuple(repr(state) for state in trace.states), trace.actions))
        return results

    def all_simple_paths(self, source: StateId, target: StateId, limit: int = 10000) -> list[TokenTrace]:
        """枚举两个 state 之间所有不重复 state 的 primitive simple paths。

        对 ``source == target`` 返回零长度路径。其它情况用 DFS 枚举不重复访问 state 的路径。
        该方法用于 landmark edge 展开；调用方应设置合理 ``limit``，避免大图中 simple path 数量爆炸。
        """

        self._require_state(source)
        self._require_state(target)
        if limit <= 0:
            raise ExpansionError("simple path enumeration limit must be positive")
        if source == target:
            return [TokenTrace(actions=(), states=(source,))]

        results: list[TokenTrace] = []
        visited = {source}

        def dfs(state: StateId, actions: list[ActionId], states: list[StateId]) -> None:
            """递归枚举不重复访问状态的 simple path。"""

            if len(results) >= limit:
                raise ExpansionError(
                    "number of simple paths exceeds configured limit",
                    context={"source": source, "target": target, "limit": limit},
                )
            if state == target:
                results.append(TokenTrace(actions=tuple(actions), states=tuple(states)))
                return

            for action in self.legal_actions(state):
                next_state = self.next_state(state, action)
                if next_state in visited:
                    continue
                visited.add(next_state)
                actions.append(action)
                states.append(next_state)
                dfs(next_state, actions, states)
                states.pop()
                actions.pop()
                visited.remove(next_state)

        dfs(source, [], [source])
        results.sort(
            key=lambda trace: (len(trace.actions), tuple(repr(state) for state in trace.states), trace.actions)
        )
        return results

    def _dijkstra(self, source: StateId) -> dict[StateId, float]:
        """在 primitive 任务图上运行 Dijkstra，返回 source 到所有可达 state 的距离。"""

        self._require_state(source)
        distances: dict[StateId, float] = {source: 0.0}
        queue: list[tuple[float, int, StateId]] = [(0.0, 0, source)]
        sequence = 0

        while queue:
            distance, _, state = heapq.heappop(queue)
            if distance > distances[state]:
                continue

            for action in self.legal_actions(state):
                next_state = self.next_state(state, action)
                new_distance = distance + self.action_cost(state, action)
                if new_distance < distances.get(next_state, math.inf):
                    distances[next_state] = new_distance
                    sequence += 1
                    heapq.heappush(queue, (new_distance, sequence, next_state))
        return distances

    def _require_state(self, state: StateId) -> None:
        """校验 state 是否存在于任务图中。"""

        if state not in self.state_index:
            raise KeyError(f"unknown state: {state!r}")

    def _validate(self) -> None:
        """检查 TaskGraph 的确定性转移和动作代价不变量。"""

        if not self.states:
            raise InvalidTaskGraphError("TaskGraph.states must be non-empty", object_id=self.task_id)
        if len(set(self.states)) != len(self.states):
            raise InvalidTaskGraphError("TaskGraph.states must be unique", object_id=self.task_id)
        if len(set(self.actions)) != len(self.actions):
            raise InvalidTaskGraphError("TaskGraph.actions must be unique", object_id=self.task_id)

        state_set = set(self.states)
        action_set = set(self.actions)

        for state, legal_actions in self.legal_actions_by_state.items():
            if state not in state_set:
                raise InvalidTaskGraphError(
                    "legal_actions contains unknown state",
                    object_id=self.task_id,
                    context={"state": state},
                )
            if len(set(legal_actions)) != len(legal_actions):
                raise InvalidTaskGraphError(
                    "legal_actions row contains duplicate actions",
                    object_id=self.task_id,
                    context={"state": state},
                )
            for action in legal_actions:
                if action not in action_set:
                    raise InvalidTaskGraphError(
                        "legal action is not in action vocabulary",
                        object_id=self.task_id,
                        context={"state": state, "action": action},
                    )

        legal_keys = {(state, action) for state, actions in self.legal_actions_by_state.items() for action in actions}
        for key, target in self.transitions.items():
            source, action = key
            if key not in legal_keys:
                raise InvalidTaskGraphError(
                    "transition is defined for a non-legal action",
                    object_id=self.task_id,
                    context={"source": source, "action": action},
                )
            if target not in state_set:
                raise InvalidTaskGraphError(
                    "transition target is not in states",
                    object_id=self.task_id,
                    context={"source": source, "action": action, "target": target},
                )

        for key in legal_keys:
            if key not in self.transitions:
                raise InvalidTaskGraphError(
                    "legal action is missing transition",
                    object_id=self.task_id,
                    context={"source": key[0], "action": key[1]},
                )
            if key not in self.action_costs:
                raise InvalidTaskGraphError(
                    "legal action is missing action cost",
                    object_id=self.task_id,
                    context={"source": key[0], "action": key[1]},
                )

        for key, cost in self.action_costs.items():
            if key not in legal_keys:
                raise InvalidTaskGraphError(
                    "action cost is defined for a non-legal action",
                    object_id=self.task_id,
                    context={"source": key[0], "action": key[1]},
                )
            if not math.isfinite(cost) or cost <= 0:
                raise InvalidTaskGraphError(
                    "action cost must be positive and finite",
                    object_id=self.task_id,
                    context={"source": key[0], "action": key[1], "cost": cost},
                )
