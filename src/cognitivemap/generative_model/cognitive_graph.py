"""认知图、认知距离和二维嵌入工具。

这一层把 transition kernel 转换为可搜索的有向 cost 图，
再计算全源最短距离、对称化距离矩阵和经典 MDS 坐标。
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from cognitivemap.generative_model.types import (
    CognitiveEdge,
    CognitiveGraph,
    DirectedDistance,
    Embedding,
    EmbeddingError,
    PlanningConfig,
    StateId,
    SymmetricDistance,
    TransitionKernel,
)


def build_cognitive_graph(kernel: TransitionKernel, config: PlanningConfig | None = None) -> CognitiveGraph:
    """从非零 kernel 概率构造有向 cognitive graph。

    每条 edge 的 cost 定义为 ``-log(probability)``。
    support 不写入 edge；需要解释概率来源时从 ModelArtifacts.kernel_support 查询。
    """

    config = config or PlanningConfig()
    edges: list[CognitiveEdge] = []
    successors: dict[StateId, list[StateId]] = defaultdict(list)

    for source in kernel.states:
        for target in kernel.states:
            probability = kernel.probability(source, target)
            if probability <= config.probability_epsilon:
                continue
            # probability_epsilon 只过滤数值噪声，不改变输入概率校验语义。
            edge = CognitiveEdge(
                source=source,
                target=target,
                probability=probability,
                cost=-math.log(probability),
            )
            edges.append(edge)
            successors[source].append(target)

    ordered_successors = {
        state: tuple(target for target in kernel.states if target in set(successors.get(state, ())))
        for state in kernel.states
    }
    return CognitiveGraph(
        family=kernel.family,
        states=kernel.states,
        edges=tuple(edges),
        successors=ordered_successors,
    )


def compute_directed_distances(graph: CognitiveGraph) -> DirectedDistance:
    """使用 Floyd-Warshall 计算 cognitive graph 上的全源有向最短距离。

    不可达 pair 保持为 ``math.inf``，后续 MDS 会 fail-fast，
    不会用任意常数替换不可达距离。
    """

    n_states = len(graph.states)
    index = {state: idx for idx, state in enumerate(graph.states)}
    dist = [[math.inf for _ in range(n_states)] for _ in range(n_states)]
    for state in graph.states:
        dist[index[state]][index[state]] = 0.0
    for edge in graph.edges:
        i = index[edge.source]
        j = index[edge.target]
        dist[i][j] = min(dist[i][j], edge.cost)

    # 小规模任务下 Floyd-Warshall 语义直接，便于测试和追踪不可达 pair。
    for k in range(n_states):
        for i in range(n_states):
            dik = dist[i][k]
            if not math.isfinite(dik):
                continue
            for j in range(n_states):
                candidate = dik + dist[k][j]
                if candidate < dist[i][j]:
                    dist[i][j] = candidate

    unreachable = tuple(
        (source, target)
        for source_index, source in enumerate(graph.states)
        for target_index, target in enumerate(graph.states)
        if source != target and not math.isfinite(dist[source_index][target_index])
    )
    return DirectedDistance(
        states=graph.states,
        matrix=tuple(tuple(row) for row in dist),
        unreachable_pairs=unreachable,
    )


def symmetrize_distances(directed: DirectedDistance) -> SymmetricDistance:
    """把有向距离对称化为 ``(d(i,j) + d(j,i)) / 2``。"""

    n_states = len(directed.states)
    matrix = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            row.append((directed.matrix[i][j] + directed.matrix[j][i]) / 2.0)
        matrix.append(tuple(row))
    return SymmetricDistance(states=directed.states, matrix=tuple(matrix))


def compute_classical_mds(symmetric: SymmetricDistance, config: PlanningConfig | None = None) -> Embedding:
    """计算二维经典 MDS 嵌入。

    输入矩阵必须有限、对称且对角线为 0。若不满足条件直接抛出 EmbeddingError，
    这是为了避免在不可达图上生成看似可用但语义错误的坐标。
    """

    config = config or PlanningConfig()
    matrix = np.asarray(symmetric.matrix, dtype=float)
    _validate_mds_input(matrix, config)

    n_states = matrix.shape[0]
    squared = matrix**2
    # 经典 MDS：B = -1/2 * J D^2 J，然后对 Gram 矩阵做特征分解。
    centering = np.eye(n_states) - np.ones((n_states, n_states)) / n_states
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    top = order[: config.mds_dim]
    top_values = np.maximum(eigenvalues[top], 0.0)
    coordinates_matrix = eigenvectors[:, top] * np.sqrt(top_values)
    if coordinates_matrix.shape[1] < config.mds_dim:
        padding = np.zeros((n_states, config.mds_dim - coordinates_matrix.shape[1]))
        coordinates_matrix = np.hstack([coordinates_matrix, padding])

    coordinates = {
        state: (float(coordinates_matrix[index, 0]), float(coordinates_matrix[index, 1]))
        for index, state in enumerate(symmetric.states)
    }
    return Embedding(
        states=symmetric.states,
        coordinates=coordinates,
        eigenvalues=tuple(float(eigenvalues[index]) for index in order),
        status="ready",
    )


def build_geometry_artifacts(
    kernel: TransitionKernel,
    config: PlanningConfig | None = None,
) -> tuple[CognitiveGraph, DirectedDistance, SymmetricDistance, Embedding]:
    """一次性构建 graph、directed distance、symmetric distance 和 embedding。"""

    graph = build_cognitive_graph(kernel, config)
    directed = compute_directed_distances(graph)
    symmetric = symmetrize_distances(directed)
    embedding = compute_classical_mds(symmetric, config)
    return graph, directed, symmetric, embedding


def _validate_mds_input(matrix: np.ndarray, config: PlanningConfig) -> None:
    """校验 MDS 输入矩阵，失败时抛出可识别的 EmbeddingError。"""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise EmbeddingError("MDS input must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise EmbeddingError("MDS input contains non-finite distances")
    if not np.allclose(matrix, matrix.T, atol=config.row_sum_tolerance):
        raise EmbeddingError("MDS input must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=config.row_sum_tolerance):
        raise EmbeddingError("MDS input diagonal must be zero")
