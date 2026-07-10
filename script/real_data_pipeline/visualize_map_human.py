# -*- coding: utf-8 -*-
"""
基于人类 chunk 推断结果构建认知地图并可视化。

读取 ``preprocess_human_data.py`` 和 ``infer_chunk_human.py`` 的输出，
使用推断出的 chunk 词表对每个 trial 的动作序列重新解析（最长匹配），
以 chunk 级序列构建认知地图并渲染为交互式 HTML。

用法::

    python script/real_data_pipeline/visualize_map_human.py
    python script/real_data_pipeline/visualize_map_human.py --participants 001
    python script/real_data_pipeline/visualize_map_human.py --participants all --domains navigation,crafting
"""

import argparse
import os
import sys
from pathlib import Path

# unused import
from typing import Dict, List  # noqa: F401

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cognitivemap._env import init  # noqa: E402

init()

from cognitivemap.inference.parser_chunk import parse_with_chunks  # noqa: E402

# unused import
# unused import
# unused import
from cognitivemap.map_estimation import compute_action_js_distance  # noqa: E402,F401
from cognitivemap.map_estimation import compute_sr_distance  # noqa: E402,F401
from cognitivemap.map_estimation import compute_transition_similarity_distance  # noqa: E402,F401
from cognitivemap.map_estimation import (  # noqa: E402
    Trial,
    build_cognitive_map,
    compute_lops_distance,
    enrich_from_trials,
)
from cognitivemap.map_estimation.visualization import render_cognitive_map_html  # noqa: E402

# ============================================================================
# 被试选择
# ============================================================================

VALID_DOMAINS = ("navigation", "crafting")


def parse_participants(raw: str | None) -> List[str] | None:
    """解析被试筛选参数，返回 None 表示使用所有可用被试。"""

    if not raw:
        return None
    raw = raw.strip()
    if raw.lower() == "all":
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def parse_domains(raw: str | None) -> List[str]:
    """解析任务域筛选参数，并校验 domain 名称。"""

    if not raw or raw.strip().lower() == "all":
        return list(VALID_DOMAINS)

    domains = [d.strip().lower() for d in raw.split(",") if d.strip()]
    invalid = [d for d in domains if d not in VALID_DOMAINS]
    if invalid:
        raise ValueError(f"未知任务类型: {invalid} (available: {list(VALID_DOMAINS)})")
    return domains


def discover_participants(data_dir: Path, domain: str) -> List[str]:
    """从当前预处理数据目录中发现指定任务的所有被试。"""
    participants: List[str] = []
    for fp in sorted(data_dir.glob(f"participant_*_{domain}.joblib")):
        parts = fp.stem.split("_")
        if len(parts) >= 3:
            participants.append(parts[1])
    return sorted(set(participants))


# ============================================================================
# 数据加载
# ============================================================================


def load_participant_data(data_dir: Path, participant_id: str, domain: str) -> pd.DataFrame:
    """加载单个被试的预处理数据。"""
    fp = data_dir / f"participant_{participant_id}_{domain}.joblib"
    if not fp.exists():
        raise FileNotFoundError(f"数据文件不存在: {fp}")
    return joblib.load(fp)


def load_inference_results(all_results: dict, domain: str, participant_id: str) -> dict:
    """从推断结果中提取单个被试的 vocab。"""
    if domain in all_results and isinstance(all_results[domain], dict):
        domain_results = all_results[domain]
        if participant_id not in domain_results:
            raise KeyError(f"被试 {participant_id} 不在 {domain} 推断结果中 (available: {list(domain_results.keys())})")
        return domain_results[participant_id]

    # 兼容旧版 human_crafting_patterns.joblib: 顶层直接是 {participant_id: result}
    if participant_id in all_results:
        if domain != "crafting":
            raise KeyError(f"旧版推断结果只包含 crafting，不包含 {domain}")
        return all_results[participant_id]

    raise KeyError(f"被试 {participant_id} 不在推断结果中 (available: {list(all_results.keys())})")


def available_in_inference(all_results: dict, domain: str) -> List[str]:
    """返回推断结果中指定任务可用的被试。"""
    if domain in all_results and isinstance(all_results[domain], dict):
        return sorted(all_results[domain].keys())
    if domain == "crafting":
        return sorted(k for k, v in all_results.items() if isinstance(v, dict) and "vocab" in v)
    return []


# ============================================================================
# 使用 chunk parsing 构造 trial
# ============================================================================


def build_chunked_trials(df: pd.DataFrame, vocab: List[str]) -> List[Trial]:
    """使用推断出的 chunk 词表重新解析每个 trial，构建 chunk 级 Trial 列表。

    每个 trial 的原始 primitive 动作序列被最长匹配解析为 chunk 序列，
    状态序列相应压缩为 chunk 级（每个 chunk 对应其起始状态 + 最终状态）。
    """
    trials: List[Trial] = []

    for trial_id, group in df.sort_values("step").groupby("trial_id", sort=False):
        valid = group[group["valid"] == True]  # noqa: E712
        if valid.empty:
            continue

        primitive_actions = list(valid["action"])
        # state 列是动作执行前的状态；next_state 列是执行后的状态
        primitive_states = list(valid["state"]) + [int(valid["next_state"].iloc[-1])]

        if not vocab:
            # 没有学到任何 chunk，直接用 primitive actions
            trials.append(Trial(state_sequence=primitive_states, action_sequence=primitive_actions))
            continue

        chunks, start_indices = parse_with_chunks(primitive_actions, vocab, need_index=True)
        # 每个 chunk 的起始状态 = 原始序列中该 chunk 第一个动作对应的 state
        chunk_states = [primitive_states[idx] for idx in start_indices]
        chunk_states.append(primitive_states[-1])  # 最终状态

        trials.append(Trial(state_sequence=chunk_states, action_sequence=chunks))

    return trials


# ============================================================================
# 核心流程
# ============================================================================


def build_and_visualize(
    data_dir: Path,
    inference_results: dict,
    output_dir: Path,
    participant_id: str,
    domain: str,
    threshold: float = 0.2,
    top_k: int = 2,
) -> None:
    """为单个被试构建认知地图并生成 HTML。"""
    df = load_participant_data(data_dir, participant_id, domain)
    inference = load_inference_results(inference_results, domain, participant_id)
    vocab = inference.get("vocab", [])

    print(
        f"{domain} participant {participant_id}: {len(vocab)} vocab items, "
        f"inferred chunks: {inference.get('learned_chunks', [])}"
    )

    trials = build_chunked_trials(df, vocab)
    print(f"  {len(trials)} trials built from chunked sequences")

    # 只生成 LoPS 方法图；其它距离函数保留 import 以便后续恢复。
    methods = {
        "lops": ("LoPS-based", compute_lops_distance),
    }

    for method_key, (method_name, distance_func) in methods.items():
        distance_matrix, state_labels = distance_func(trials)
        result = build_cognitive_map(distance_matrix=distance_matrix, state_labels=state_labels, method=method_key)
        result = enrich_from_trials(result, trials)

        output_path = output_dir / domain / f"participant_{participant_id}_{domain}_cognitive_map_{method_key}.html"
        render_cognitive_map_html(
            result,
            output_path,
            title=f"Human Cognitive Map — P{participant_id} {domain} ({method_name})",
            threshold=threshold,
            top_k=top_k,
            token_labels=vocab,
        )
        print(f"  {method_key}: stress={result.stress:.4f} → {output_path}")


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """命令行入口：加载 chunk 推断结果并批量渲染认知地图 HTML。"""

    parser = argparse.ArgumentParser(description="基于人类 chunk 推断结果构建认知地图并可视化。")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/real_data_pipeline/processed"),
        help="预处理后的人类数据目录 (default: data/real_data_pipeline/processed)",
    )
    parser.add_argument(
        "--inference-path",
        type=Path,
        default=Path("data/real_data_pipeline/inference_results/chunks.joblib"),
        help="chunk 推断结果文件 (default: data/real_data_pipeline/inference_results/chunks.joblib)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figs/real_data_pipeline/cognitive_maps"),
        help="HTML 输出目录 (default: results/figs/real_data_pipeline/cognitive_maps)",
    )
    parser.add_argument("--participants", type=str, default="all", help='选择被试: "all", "001", "001,002" 等')
    parser.add_argument(
        "--domains",
        type=str,
        default="navigation,crafting",
        help='选择任务: "all", "navigation", "crafting", "navigation,crafting"',
    )
    parser.add_argument("--threshold", type=float, default=0.2, help="主 action 比例阈值")
    parser.add_argument("--top-k", type=int, default=2, help="每条边最多显示几个 action")

    args = parser.parse_args()

    if not args.inference_path.exists():
        raise FileNotFoundError(
            f"推断结果文件不存在: {args.inference_path}\n请先运行 script/real_data_pipeline/infer_chunk_human.py"
        )

    inference = joblib.load(args.inference_path)
    requested_participants = parse_participants(args.participants)
    domains = parse_domains(args.domains)

    os.makedirs(args.output_dir, exist_ok=True)

    for domain in domains:
        data_participants = discover_participants(args.data_dir, domain)
        inference_participants = available_in_inference(inference, domain)
        available = sorted(set(data_participants) & set(inference_participants))

        if requested_participants is None:
            selected = available
        else:
            missing_data = [p for p in requested_participants if p not in data_participants]
            missing_inference = [p for p in requested_participants if p not in inference_participants]
            if missing_data or missing_inference:
                raise ValueError(
                    f"{domain} 缺少被试: data={missing_data}, inference={missing_inference}; "
                    f"data available={data_participants}, inference available={inference_participants}"
                )
            selected = sorted(requested_participants)

        if not selected:
            print(f"{domain}: 未找到可用被试，跳过。")
            continue

        for pid in selected:
            build_and_visualize(
                data_dir=args.data_dir,
                inference_results=inference,
                output_dir=args.output_dir,
                participant_id=pid,
                domain=domain,
                threshold=args.threshold,
                top_k=args.top_k,
            )

    print(f"\n全部完成，HTML 文件保存在: {args.output_dir}/")


if __name__ == "__main__":
    main()
