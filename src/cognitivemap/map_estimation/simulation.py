# -*- coding: utf-8 -*-
"""
模拟 Trial 数据生成模块
========================
提供 3x3 网格导航的模拟数据生成函数，供脚本和测试复用。

3x3 网格结构（节点编号 1-9）：
    ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │
    ├───┼───┼───┤
    │ 4 │ 5 │ 6 │
    ├───┼───┼───┤
    │ 7 │ 8 │ 9 │
    └───┴───┴───┘

合法动作：
    - U (Up):    向上移动一行
    - D (Down):  向下移动一行
    - L (Left):  向左移动一列
    - R (Right): 向右移动一列
    - P (Pause): 停留在当前位置
"""

from typing import Dict, List, Tuple

import numpy as np

from cognitivemap.map_estimation.distances import Trial

# ============================================================================
# 3x3 网格定义
# ============================================================================

# 网格的行列数
GRID_ROWS = 3
GRID_COLS = 3

# 每个节点 (row, col) 的合法移动方向
# 格式：{动作名: (dr, dc)}
MOVES = {
    "U": (-1, 0),  # 向上
    "D": (1, 0),  # 向下
    "L": (0, -1),  # 向左
    "R": (0, 1),  # 向右
    "P": (0, 0),  # 停留
}


def state_to_rc(state: int) -> Tuple[int, int]:
    """将状态编号转换为网格中的 (row, col)。状态编号为 1-9。"""
    s = state - 1
    return s // GRID_COLS, s % GRID_COLS


def rc_to_state(row: int, col: int) -> int:
    """将网格中的 (row, col) 转换为状态编号（1-9）。"""
    return row * GRID_COLS + col + 1


def in_bounds(row: int, col: int) -> bool:
    """检查 (row, col) 是否在网格范围内。"""
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS


def get_available_actions(state: int) -> Dict[str, int]:
    """获取当前状态下所有合法的动作及其对应的下一个状态。

    Args:
        state: 当前状态编号。

    Returns:
        {动作名: 下一状态编号} 字典。
    """
    row, col = state_to_rc(state)
    available = {}
    for action, (dr, dc) in MOVES.items():
        nr, nc = row + dr, col + dc
        if in_bounds(nr, nc):
            available[action] = rc_to_state(nr, nc)
    return available


# ============================================================================
# 模拟 Trial 数据生成
# ============================================================================


def generate_random_trial(
    rng: np.random.RandomState,
    min_length: int = 5,
    max_length: int = 20,
    pause_prob: float = 0.1,
) -> Trial:
    """生成一个随机的导航 trial。

    随机选择起点，然后通过随机游走生成路径。每一步随机选择一个合法动作。
    路径长度在 [min_length, max_length] 之间随机选择。

    Args:
        rng: 随机数生成器。
        min_length: 最小路径长度（状态数）。
        max_length: 最大路径长度（状态数）。
        pause_prob: 选择 P（停留）动作的概率。

    Returns:
        一个 Trial 实例。
    """
    # 随机选择起点（状态编号 1-9）
    current_state = rng.randint(1, GRID_ROWS * GRID_COLS + 1)
    state_sequence = [current_state]
    action_sequence = []

    # 随机决定路径长度
    target_length = rng.randint(min_length, max_length + 1)

    for _ in range(target_length - 1):
        available = get_available_actions(current_state)

        # 非 P 动作的列表
        non_pause_actions = [a for a in available if a != "P"]
        if not non_pause_actions:
            # 如果没有非 P 动作（不应发生），使用所有动作
            non_pause_actions = list(available.keys())

        # 以 pause_prob 的概率选择 P，否则随机选择一个非 P 动作
        if rng.random() < pause_prob and "P" in available:
            chosen_action = "P"
        else:
            chosen_action = non_pause_actions[rng.randint(0, len(non_pause_actions))]

        next_state = available[chosen_action]
        state_sequence.append(next_state)
        action_sequence.append(chosen_action)
        current_state = next_state

    return Trial(
        state_sequence=state_sequence,
        action_sequence=action_sequence,
    )


def generate_goal_directed_trial(
    rng: np.random.RandomState,
    start_state: int,
    goal_state: int,
    noise_prob: float = 0.2,
    max_steps: int = 50,
) -> Trial:
    """生成一个目标导向的导航 trial。

    从起点出发，以偏向目标的方向移动，同时加入噪声以模拟真实行为。

    Args:
        rng: 随机数生成器。
        start_state: 起始状态。
        goal_state: 目标状态。
        noise_prob: 随机选择动作（而非朝向目标）的概率。
        max_steps: 最大步数限制。

    Returns:
        一个 Trial 实例。
    """
    current_state = start_state
    state_sequence = [current_state]
    action_sequence = []

    goal_row, goal_col = state_to_rc(goal_state)

    for _ in range(max_steps):
        if current_state == goal_state:
            break

        available = get_available_actions(current_state)

        # 计算每个动作的"目标导向得分"：选择使曼哈顿距离减小的动作
        best_actions = []
        best_dist = float("inf")

        for action, next_state in available.items():
            if action == "P":
                continue
            nr, nc = state_to_rc(next_state)
            dist = abs(nr - goal_row) + abs(nc - goal_col)
            if dist < best_dist:
                best_dist = dist
                best_actions = [action]
            elif dist == best_dist:
                best_actions.append(action)

        if not best_actions:
            best_actions = list(available.keys())

        # 以 noise_prob 的概率随机选择，否则选择最优动作
        if rng.random() < noise_prob:
            non_pause = [a for a in available if a != "P"] or list(available.keys())
            chosen_action = non_pause[rng.randint(0, len(non_pause))]
        else:
            chosen_action = best_actions[rng.randint(0, len(best_actions))]

        next_state = available[chosen_action]
        state_sequence.append(next_state)
        action_sequence.append(chosen_action)
        current_state = next_state

    return Trial(
        state_sequence=state_sequence,
        action_sequence=action_sequence,
    )


def generate_simulated_trials(
    n_trials: int = 200,
    seed: int = 42,
) -> List[Trial]:
    """生成模拟的 3x3 网格导航 trial 数据集。

    混合两种 trial 类型：
        1. 随机游走 trial（约 40%）：模拟探索行为。
        2. 目标导向 trial（约 60%）：模拟从随机起点到随机终点的导航。

    Args:
        n_trials: 生成的 trial 总数。
        seed: 随机种子。

    Returns:
        Trial 列表。
    """
    rng = np.random.RandomState(seed)
    trials = []

    for i in range(n_trials):
        if rng.random() < 0.4:
            # 随机游走 trial
            trial = generate_random_trial(
                rng,
                min_length=5,
                max_length=20,
                pause_prob=0.1,
            )
        else:
            # 目标导向 trial
            start = rng.randint(1, 10)
            goal = rng.randint(1, 10)
            # 确保起点和终点不同
            while goal == start:
                goal = rng.randint(1, 10)
            trial = generate_goal_directed_trial(
                rng,
                start_state=start,
                goal_state=goal,
                noise_prob=0.2,
                max_steps=30,
            )
        trials.append(trial)

    return trials
