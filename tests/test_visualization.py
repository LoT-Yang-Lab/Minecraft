"""认知地图 HTML 可视化序列化与渲染测试。"""

import numpy as np

from cognitivemap.map_estimation import CognitiveMapResult
from cognitivemap.map_estimation.visualization import _build_js_data, render_cognitive_map_html


def test_build_js_data_basic() -> None:
    """测试 _build_js_data 能正确转换 CognitiveMapResult 为 JS dict。"""
    _ = 3
    dist_matrix = np.array(
        [
            [0.0, 0.5, 2.0],
            [0.5, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ]
    )
    state_labels = [0, 1, 2]
    mds_coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [3.0, 2.0],
        ]
    )
    result = CognitiveMapResult(
        distance_matrix=dist_matrix,
        state_labels=state_labels,
        mds_coordinates=mds_coords,
        stress=0.05,
        method="lops",
        edge_actions={
            (0, 1): {"R": 10, "U": 3},
            (1, 2): {"D": 8},
        },
        action_labels=["D", "R", "U"],
    )

    data = _build_js_data(result, threshold=0.2, top_k=2)

    assert "nodes" in data
    assert "edges" in data
    assert "actions" in data
    assert "action_colors" in data
    assert "config" in data

    assert len(data["nodes"]) == 3
    assert data["nodes"][0]["id"] == "0"
    assert 0.0 <= data["nodes"][0]["x"] <= 1.0
    assert 0.0 <= data["nodes"][0]["y"] <= 1.0

    assert len(data["edges"]) == 2
    # edge (0,1) 的主导 action 应为 R。
    edge_01 = [e for e in data["edges"] if e["source"] == "0" and e["target"] == "1"][0]
    assert edge_01["primary_proportion"] == round(10 / 13, 4)
    assert "R" in edge_01["actions"]
    assert "U" in edge_01["actions"]
    assert edge_01["tier"] in ("near", "mid", "far")

    assert set(data["actions"]) == {"D", "R", "U"}
    assert all(a in data["action_colors"] for a in data["actions"])
    assert data["config"]["method"] == "lops"
    assert data["config"]["stress"] == 0.05


def test_build_js_data_no_edge_actions() -> None:
    """测试没有 edge_actions 时返回空的 edges 列表。"""
    result = CognitiveMapResult(
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        state_labels=[0, 1],
        mds_coordinates=np.array([[0.0, 0.0], [1.0, 1.0]]),
        stress=0.1,
        method="sr",
    )

    data = _build_js_data(result)
    assert data["edges"] == []
    assert data["actions"] == []


def test_render_cognitive_map_html_writes_graph(tmp_path) -> None:
    """测试 render_cognitive_map_html 能正常生成包含预期内容的 HTML 文件。"""
    _ = 3
    dist_matrix = np.array(
        [
            [0.0, 0.5, 2.0],
            [0.5, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ]
    )
    state_labels = [0, 1, 2]
    mds_coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [3.0, 2.0],
        ]
    )
    result = CognitiveMapResult(
        distance_matrix=dist_matrix,
        state_labels=state_labels,
        mds_coordinates=mds_coords,
        stress=0.05,
        method="sr",
    )

    output = tmp_path / "cognitive_map.html"
    saved_path = render_cognitive_map_html(
        result=result,
        output_path=output,
        title="Test Map",
    )

    assert saved_path == str(output)
    content = output.read_text(encoding="utf-8")
    assert "Test Map" in content
    assert "Successor Representation" in content
    assert "d3.v7.min.js" in content  # D3.js CDN
    assert "MDS Stress" in content
    assert "reveal transitions" in content.lower()


def test_render_cognitive_map_html_does_not_print_by_default(tmp_path, capsys) -> None:
    """测试默认渲染 HTML 时不会向 stdout 打印额外内容。"""

    result = CognitiveMapResult(
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        state_labels=[0, 1],
        mds_coordinates=np.array([[0.0, 0.0], [1.0, 1.0]]),
        stress=0.1,
        method="sr",
    )

    render_cognitive_map_html(result, tmp_path / "silent_map.html")

    captured = capsys.readouterr()
    assert captured.out == ""


def test_render_cognitive_map_html_lops_shows_edge_actions(tmp_path) -> None:
    """测试 LoPS 方法的图中包含边上的 action 标签数据。"""
    _ = 3
    dist_matrix = np.array(
        [
            [0.0, 0.5, 2.0],
            [0.5, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ]
    )
    state_labels = [0, 1, 2]
    mds_coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [3.0, 2.0],
        ]
    )
    result = CognitiveMapResult(
        distance_matrix=dist_matrix,
        state_labels=state_labels,
        mds_coordinates=mds_coords,
        stress=0.05,
        method="lops",
        edge_actions={
            (0, 1): {"R": 10, "U": 3},
            (1, 2): {"D": 8},
        },
        action_labels=["D", "R", "U"],
    )

    output = tmp_path / "cognitive_map_lops.html"
    saved_path = render_cognitive_map_html(
        result=result,
        output_path=output,
        title="LoPS Test Map",
    )

    assert saved_path == str(output)
    content = output.read_text(encoding="utf-8")
    assert "LoPS Test Map" in content
    assert "LoPS-based" in content
    # 检查边数据被嵌入
    assert '"R"' in content
    assert '"U"' in content
    assert "10" in content or "0.7692" in content  # 10/13 ≈ 0.7692


def test_render_cognitive_map_html_respects_threshold(tmp_path) -> None:
    """测试 threshold 参数被正确嵌入 HTML。"""
    result = CognitiveMapResult(
        distance_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        state_labels=[0, 1],
        mds_coordinates=np.array([[0.0, 0.0], [1.0, 1.0]]),
        stress=0.1,
        method="sr",
    )

    output = tmp_path / "map_threshold.html"
    render_cognitive_map_html(result, output, threshold=0.5)
    content = output.read_text(encoding="utf-8")
    assert "0.50" in content
