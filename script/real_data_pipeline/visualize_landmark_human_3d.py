# -*- coding: utf-8 -*-
"""
将人类数据中推断出的 landmark 渲染为交互式 3D 转移概率 HTML。

这个可视化完全由数据驱动：
- 节点位置来自经验转移概率距离的 3D MDS 嵌入；
- 有向边来自经验转移概率 ``p(s' | s)``；
- 被推断为 landmark 的状态会以更大、更醒目的金色节点显示。
"""

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cognitivemap._env import init  # noqa: E402

init()


VALID_DOMAINS = ("navigation", "crafting")


def parse_participants(raw: Optional[str]) -> Optional[List[str]]:
    """解析被试筛选参数，返回 None 表示使用全部被试。"""

    if not raw:
        return None
    raw = raw.strip()
    if raw.lower() == "all":
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def parse_domains(raw: Optional[str]) -> List[str]:
    """解析任务域筛选参数，并校验 domain 名称。"""

    if not raw or raw.strip().lower() == "all":
        return list(VALID_DOMAINS)
    domains = [domain.strip() for domain in raw.split(",") if domain.strip()]
    invalid = [domain for domain in domains if domain not in VALID_DOMAINS]
    if invalid:
        raise ValueError(f"Unknown domains: {invalid}; expected navigation, crafting, or all")
    return domains


def nested_counts_to_matrix(transition_counts: Dict) -> tuple[List[int], np.ndarray]:
    """把嵌套转移计数字典转换为状态标签和方阵。"""

    states = set()
    for source, row in transition_counts.items():
        states.add(int(source))
        for target in row:
            states.add(int(target))
    state_labels = sorted(states)
    state_to_idx = {state: idx for idx, state in enumerate(state_labels)}
    matrix = np.zeros((len(state_labels), len(state_labels)), dtype=np.float64)

    for source, row in transition_counts.items():
        source_idx = state_to_idx[int(source)]
        for target, count in row.items():
            matrix[source_idx, state_to_idx[int(target)]] = float(count)
    return state_labels, matrix


def transition_probabilities(counts: np.ndarray) -> np.ndarray:
    """将转移计数矩阵按行归一化为经验转移概率矩阵。"""

    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=np.float64), where=row_sums > 0)


def probability_distance_matrix(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """从有向经验转移概率构造对称距离矩阵。

    先把双向概率平均为亲和度，再用 ``sqrt(-log(p))`` 转成距离；Floyd-Warshall 闭包让间接高概率路径
    也能影响 MDS 嵌入，避免稀疏边导致距离矩阵不可用。
    """

    affinity = (probabilities + probabilities.T) / 2.0
    distance = np.full_like(affinity, np.inf, dtype=np.float64)
    mask = affinity > 0
    distance[mask] = np.maximum(np.sqrt(-np.log(np.clip(affinity[mask], eps, 1.0))), 1e-6)
    np.fill_diagonal(distance, 0.0)

    # Floyd-Warshall 在 MDS 前补全间接路径距离。
    n_states = distance.shape[0]
    for via in range(n_states):
        distance = np.minimum(distance, distance[:, [via]] + distance[[via], :])

    finite = distance[np.isfinite(distance) & (distance > 0)]
    fallback = float(finite.max() * 1.25) if finite.size else 1.0
    distance[~np.isfinite(distance)] = fallback
    np.fill_diagonal(distance, 0.0)
    return distance


def embed_states_3d(distance_matrix: np.ndarray, random_state: int) -> tuple[np.ndarray, float]:
    """对状态距离矩阵做 3D MDS 嵌入，并归一化到单位尺度附近。"""

    n_states = distance_matrix.shape[0]
    if n_states == 1:
        return np.zeros((1, 3), dtype=np.float64), 0.0

    n_components = min(3, n_states)
    model = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        normalized_stress="auto",
        n_init=8,
    )
    coords = model.fit_transform(distance_matrix)
    if n_components < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - n_components)), mode="constant")

    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(coords, axis=1).max()
    if scale > 0:
        coords = coords / scale
    return coords, float(model.stress_)


def build_payload(result: Dict, random_state: int) -> Dict:
    """将单个被试的 landmark 推断结果转换为前端 Three.js 所需 payload。"""

    state_labels, counts = nested_counts_to_matrix(result["transition_counts"])
    probabilities = transition_probabilities(counts)
    distance_matrix = probability_distance_matrix(probabilities)
    coords, stress = embed_states_3d(distance_matrix, random_state=random_state)
    landmark_set = {int(state) for state in result.get("landmarks", [])}
    state_scores = {int(state): value for state, value in result.get("state_scores", {}).items()}

    nodes = []
    for idx, state in enumerate(state_labels):
        scores = state_scores.get(state, {})
        nodes.append(
            {
                "id": str(state),
                "label": str(state),
                "x": round(float(coords[idx, 0]), 6),
                "y": round(float(coords[idx, 1]), 6),
                "z": round(float(coords[idx, 2]), 6),
                "is_landmark": state in landmark_set,
                "score": round(float(scores.get("score", 0.0)), 6),
                "selection_rate": round(float(scores.get("selection_rate", 0.0)), 6),
                "coverage": round(float(scores.get("coverage", 0.0)), 6),
                "path_commonality": round(float(scores.get("path_commonality", 0.0)), 6),
                "betweenness": round(float(scores.get("betweenness", 0.0)), 6),
            }
        )

    edges = []
    for source_idx, source in enumerate(state_labels):
        for target_idx, target in enumerate(state_labels):
            count = int(counts[source_idx, target_idx])
            probability = float(probabilities[source_idx, target_idx])
            if count == 0 or probability <= 0:
                continue
            edges.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "count": count,
                    "probability": round(probability, 6),
                }
            )

    return {
        "participant": result.get("participant"),
        "domain": result.get("domain"),
        "landmarks": [str(state) for state in result.get("landmarks", [])],
        "top_landmarks": [str(state) for state in result.get("top_landmarks", [])],
        "nodes": nodes,
        "edges": edges,
        "config": {
            "embedding_stress": round(stress, 6),
            "n_states": len(state_labels),
            "n_edges": len(edges),
            "total_transitions": int(counts.sum()),
        },
    }


def render_html(payload: Dict, output_path: Path, title: str, min_probability: float) -> None:
    """把 payload 注入 HTML 模板并写出交互式 3D 页面。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
    content = content.replace("__DATA_JSON__", json.dumps(payload, ensure_ascii=False))
    content = content.replace("__MIN_PROBABILITY__", json.dumps(float(min_probability)))
    output_path.write_text(content, encoding="utf-8")


def visualize_results(
    inference_path: Path,
    output_dir: Path,
    participants: Optional[List[str]],
    domains: List[str],
    random_state: int,
    min_probability: float,
) -> None:
    """按 domain 和被试批量生成 3D landmark HTML 文件。"""

    results = joblib.load(inference_path)
    for domain in domains:
        if domain not in results:
            raise KeyError(f"domain {domain} not found in {inference_path}")
        available = sorted(results[domain])
        selected = available if participants is None else sorted(participants)
        missing = [pid for pid in selected if pid not in results[domain]]
        if missing:
            raise KeyError(f"{domain} participants not found: {missing}; available={available}")

        for pid in selected:
            result = dict(results[domain][pid])
            result.setdefault("participant", pid)
            result.setdefault("domain", domain)
            payload = build_payload(result, random_state=random_state)
            output_path = output_dir / domain / f"participant_{pid}_{domain}_landmarks_3d.html"
            title = f"Human Landmarks 3D - P{pid} {domain}"
            render_html(payload, output_path, title=title, min_probability=min_probability)
            print(f"{domain} participant {pid}: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """构造 3D landmark 可视化脚本的命令行解析器。"""

    parser = argparse.ArgumentParser(description="Render mined human landmarks as 3D transition HTML.")
    parser.add_argument(
        "--inference-path",
        type=Path,
        default=Path("data/real_data_pipeline/inference_results/landmarks.joblib"),
        help="landmark inference result file (default: data/real_data_pipeline/inference_results/landmarks.joblib)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figs/real_data_pipeline/landmarks_3d"),
        help="HTML output directory (default: results/figs/real_data_pipeline/landmarks_3d)",
    )
    parser.add_argument("--participants", type=str, default="all", help='选择被试: "all", "003", "003,005"')
    parser.add_argument(
        "--domains",
        type=str,
        default="navigation,crafting",
        help='选择任务: "all", "navigation", "crafting", "navigation,crafting"',
    )
    parser.add_argument("--random-state", type=int, default=42, help="MDS random seed")
    parser.add_argument("--min-probability", type=float, default=0.0, help="initial edge probability threshold")
    return parser


def main() -> None:
    """命令行入口：读取 landmark 推断结果并渲染 HTML。"""

    args = build_arg_parser().parse_args()
    visualize_results(
        inference_path=args.inference_path,
        output_dir=args.output_dir,
        participants=parse_participants(args.participants),
        domains=parse_domains(args.domains),
        random_state=args.random_state,
        min_probability=args.min_probability,
    )
    print(f"\n全部完成，HTML 文件保存在: {args.output_dir}/")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>__TITLE__</title>
    <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
    <style>
        :root {
            --bg: #f7fafc;
            --panel: #ffffff;
            --border: #dbe3ee;
            --text: #243042;
            --muted: #64748b;
            --node: #2563eb;
            --landmark: #f59e0b;
            --edge: #475569;
            --selected: #dc2626;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            height: 100vh;
            overflow: hidden;
            display: grid;
            grid-template-rows: auto 1fr;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            color: var(--text);
            background: var(--bg);
        }
        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 12px 18px;
            background: var(--panel);
            border-bottom: 1px solid var(--border);
        }
        h1 {
            margin: 0;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0;
        }
        .metrics {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            color: var(--muted);
            font-size: 12px;
        }
        .badge {
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 4px 8px;
            background: #f8fafc;
            white-space: nowrap;
        }
        .shell {
            min-height: 0;
            display: grid;
            grid-template-columns: minmax(0, 1fr) 330px;
        }
        #scene {
            position: relative;
            min-width: 0;
            min-height: 0;
        }
        canvas {
            display: block;
            width: 100%;
            height: 100%;
        }
        aside {
            border-left: 1px solid var(--border);
            background: var(--panel);
            padding: 16px;
            overflow-y: auto;
        }
        .section {
            margin-bottom: 18px;
        }
        .section h2 {
            margin: 0 0 8px 0;
            font-size: 13px;
            letter-spacing: 0;
        }
        .legend-row, .transition-row, .landmark-row {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            padding: 6px 0;
            border-bottom: 1px solid #eef2f7;
            font-size: 12px;
        }
        .legend-key {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        label {
            display: grid;
            gap: 6px;
            font-size: 12px;
            color: var(--muted);
        }
        input[type=range] {
            width: 100%;
        }
        #tooltip {
            position: absolute;
            pointer-events: none;
            display: none;
            padding: 6px 8px;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.88);
            color: #ffffff;
            font-size: 12px;
            z-index: 2;
        }
        @media (max-width: 820px) {
            .shell { grid-template-columns: 1fr; grid-template-rows: minmax(0, 1fr) 260px; }
            aside { border-left: none; border-top: 1px solid var(--border); }
            header { align-items: flex-start; flex-direction: column; }
        }
    </style>
</head>
<body>
<header>
    <h1>__TITLE__</h1>
    <div class="metrics">
        <span class="badge">States: <span id="metric-states"></span></span>
        <span class="badge">Edges: <span id="metric-edges"></span></span>
        <span class="badge">Transitions: <span id="metric-transitions"></span></span>
        <span class="badge">MDS stress: <span id="metric-stress"></span></span>
    </div>
</header>
<main class="shell">
    <div id="scene"><div id="tooltip"></div></div>
    <aside>
        <div class="section">
            <h2>Edge Threshold</h2>
            <label>
                p(s'|s) >= <span id="threshold-value"></span>
                <input id="threshold" type="range" min="0" max="1" step="0.01">
            </label>
        </div>
        <div class="section">
            <h2>Legend</h2>
            <div class="legend-row">
                <span class="legend-key"><span class="dot" style="background: var(--landmark);"></span>Landmark</span>
                <span id="landmark-list"></span>
            </div>
            <div class="legend-row">
                <span class="legend-key"><span class="dot" style="background: var(--node);"></span>State</span>
                <span>3D MDS</span>
            </div>
        </div>
        <div class="section">
            <h2>Mined Landmarks</h2>
            <div id="landmark-table"></div>
        </div>
        <div class="section">
            <h2 id="selected-title">Outgoing Transitions</h2>
            <div id="transition-table"></div>
        </div>
    </aside>
</main>
<script>
const DATA = __DATA_JSON__;
window.LANDMARK_DATA = DATA;
const INITIAL_THRESHOLD = __MIN_PROBABILITY__;

const sceneHost = document.getElementById("scene");
const tooltip = document.getElementById("tooltip");
const thresholdInput = document.getElementById("threshold");
const thresholdValue = document.getElementById("threshold-value");
const transitionTable = document.getElementById("transition-table");
const selectedTitle = document.getElementById("selected-title");

document.getElementById("metric-states").innerText = DATA.config.n_states;
document.getElementById("metric-edges").innerText = DATA.config.n_edges;
document.getElementById("metric-transitions").innerText = DATA.config.total_transitions;
document.getElementById("metric-stress").innerText = DATA.config.embedding_stress.toFixed(4);
document.getElementById("landmark-list").innerText = DATA.landmarks.join(", ") || "none";

thresholdInput.value = INITIAL_THRESHOLD;
thresholdValue.innerText = Number(INITIAL_THRESHOLD).toFixed(2);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0xf7fafc, 1);
sceneHost.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 0, 4.8);

const root = new THREE.Group();
root.rotation.x = -0.55;
root.rotation.y = 0.72;
scene.add(root);

scene.add(new THREE.AmbientLight(0xffffff, 0.78));
const light = new THREE.DirectionalLight(0xffffff, 0.9);
light.position.set(2, 3, 4);
scene.add(light);

const nodeById = new Map(DATA.nodes.map(node => [node.id, node]));
const objectById = new Map();
let selectedNodeId = DATA.landmarks[0] || (DATA.nodes[0] && DATA.nodes[0].id);
const edgeObjects = [];

function makeTextSprite(text, color = "#243042", scale = 0.18) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 256;
    canvas.height = 96;
    ctx.font = "700 34px -apple-system, BlinkMacSystemFont, Segoe UI, Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(255, 255, 255, 0.88)";
    ctx.fillRect(0, 18, 256, 60);
    ctx.fillStyle = color;
    ctx.fillText(text, 128, 48);
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(scale * 2.4, scale * 0.9, 1);
    return sprite;
}

function vectorFor(node) {
    return new THREE.Vector3(node.x * 1.75, node.y * 1.75, node.z * 1.75);
}

function setObjectOpacity(object, opacity) {
    object.traverse(child => {
        if (child.material) {
            child.material.transparent = true;
            child.material.opacity = opacity;
        }
    });
}

function makeArrow(edge) {
    const source = vectorFor(nodeById.get(edge.source));
    const target = vectorFor(nodeById.get(edge.target));
    const delta = target.clone().sub(source);
    const length = delta.length();
    const color = new THREE.Color(0x475569);
    if (length < 1e-6) {
        const curve = new THREE.TorusGeometry(0.11, 0.008 + edge.probability * 0.012, 8, 40);
        const material = new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.24 + edge.probability * 0.56,
        });
        const torus = new THREE.Mesh(curve, material);
        torus.position.copy(source);
        torus.userData = edge;
        return torus;
    }
    const direction = delta.normalize();
    const arrow = new THREE.ArrowHelper(
        direction,
        source,
        length * 0.9,
        color,
        0.10 + edge.probability * 0.12,
        0.045 + edge.probability * 0.05,
    );
    arrow.position.add(direction.clone().multiplyScalar(0.09));
    setObjectOpacity(arrow, 0.20 + edge.probability * 0.62);
    arrow.userData = edge;
    return arrow;
}

DATA.edges.forEach(edge => {
    const arrow = makeArrow(edge);
    root.add(arrow);
    edgeObjects.push(arrow);
});

DATA.nodes.forEach(node => {
    const isLandmark = node.is_landmark;
    const geometry = new THREE.SphereGeometry(isLandmark ? 0.105 : 0.075, 32, 20);
    const material = new THREE.MeshStandardMaterial({
        color: isLandmark ? 0xf59e0b : 0x2563eb,
        roughness: 0.42,
        metalness: isLandmark ? 0.08 : 0.02,
        emissive: isLandmark ? 0x3b2500 : 0x000000,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(vectorFor(node));
    mesh.userData = { kind: "node", id: node.id };
    objectById.set(node.id, mesh);
    root.add(mesh);

    const label = makeTextSprite(isLandmark ? `${node.label} L` : node.label, isLandmark ? "#92400e" : "#1e3a8a");
    label.position.copy(mesh.position).add(new THREE.Vector3(0, 0.17, 0));
    root.add(label);
});

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let dragging = false;
let lastX = 0;
let lastY = 0;

function updateSize() {
    const rect = sceneHost.getBoundingClientRect();
    renderer.setSize(rect.width, rect.height, false);
    camera.aspect = rect.width / Math.max(rect.height, 1);
    camera.updateProjectionMatrix();
}

function updateThreshold() {
    const threshold = Number(thresholdInput.value);
    thresholdValue.innerText = threshold.toFixed(2);
    edgeObjects.forEach(object => {
        object.visible = object.userData.probability >= threshold;
    });
    updatePanel();
}

function updatePanel() {
    const node = nodeById.get(selectedNodeId);
    if (!node) return;
    selectedTitle.innerText = `Outgoing from ${node.label}`;
    const outgoing = DATA.edges
        .filter(edge => edge.source === selectedNodeId && edge.probability >= Number(thresholdInput.value))
        .sort((a, b) => b.probability - a.probability);
    if (!outgoing.length) {
        transitionTable.innerHTML = `
            <div class="transition-row"><span>No visible outgoing edge</span><span></span></div>
        `;
        return;
    }
    transitionTable.innerHTML = outgoing.map(edge => `
        <div class="transition-row">
            <span>${edge.source} -> ${edge.target}</span>
            <span>p=${edge.probability.toFixed(2)} · n=${edge.count}</span>
        </div>
    `).join("");
}

function updateSelection() {
    objectById.forEach((mesh, id) => {
        const node = nodeById.get(id);
        mesh.material.color.set(id === selectedNodeId ? 0xdc2626 : (node.is_landmark ? 0xf59e0b : 0x2563eb));
        mesh.scale.setScalar(id === selectedNodeId ? 1.28 : 1.0);
    });
    updatePanel();
}

function setPointer(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function pickNode(event) {
    setPointer(event);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects([...objectById.values()], false);
    if (hits.length) {
        selectedNodeId = hits[0].object.userData.id;
        updateSelection();
    }
}

renderer.domElement.addEventListener("pointerdown", event => {
    dragging = true;
    lastX = event.clientX;
    lastY = event.clientY;
});

renderer.domElement.addEventListener("pointermove", event => {
    setPointer(event);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects([...objectById.values()], false);
    if (hits.length) {
        const node = nodeById.get(hits[0].object.userData.id);
        tooltip.style.display = "block";
        tooltip.style.left = `${event.offsetX + 12}px`;
        tooltip.style.top = `${event.offsetY + 12}px`;
        tooltip.innerHTML = `
            state ${node.label}<br>score=${node.score.toFixed(3)}<br>rate=${node.selection_rate.toFixed(3)}
        `;
    } else {
        tooltip.style.display = "none";
    }

    if (!dragging) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    root.rotation.y += dx * 0.008;
    root.rotation.x += dy * 0.008;
    lastX = event.clientX;
    lastY = event.clientY;
});

window.addEventListener("pointerup", () => { dragging = false; });
renderer.domElement.addEventListener("click", pickNode);
renderer.domElement.addEventListener("wheel", event => {
    event.preventDefault();
    camera.position.z = Math.max(2.2, Math.min(8.0, camera.position.z + event.deltaY * 0.004));
}, { passive: false });
thresholdInput.addEventListener("input", updateThreshold);
window.addEventListener("resize", updateSize);

document.getElementById("landmark-table").innerHTML = DATA.nodes
    .filter(node => node.is_landmark)
    .sort((a, b) => b.selection_rate - a.selection_rate)
    .map(node => `
        <div class="landmark-row">
            <span>state ${node.label}</span>
            <span>score=${node.score.toFixed(3)} · rate=${node.selection_rate.toFixed(3)}</span>
        </div>
    `).join("");

updateSize();
updateThreshold();
updateSelection();

function animate() {
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}
animate();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
