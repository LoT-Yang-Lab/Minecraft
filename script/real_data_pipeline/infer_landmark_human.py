# -*- coding: utf-8 -*-
"""
从人类 navigation / crafting 状态序列中推断 landmark candidates。

这个脚本只做数据驱动的 landmark inference：
- 不读取 chunk inference 结果；
- 不依赖生成模型；
- 不拟合 landmark CPD。

用法::

    python script/real_data_pipeline/infer_landmark_human.py
    python script/real_data_pipeline/infer_landmark_human.py --participants 003 --domains navigation
    python script/real_data_pipeline/infer_landmark_human.py --participants all --n-jobs 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cognitivemap._env import init  # noqa: E402

init()

from cognitivemap.inference.miner_landmark import (  # noqa: E402
    DEFAULT_FEATURES,
    LandmarkMiningConfig,
    mine_landmarks,
    state_sequences_from_transition_rows,
)

VALID_DOMAINS = ("navigation", "crafting")


def parse_participants(raw: Optional[str]) -> Optional[List[str]]:
    """解析 --participants 参数。"""

    if not raw:
        return None
    raw = raw.strip()
    if raw.lower() == "all":
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def parse_domains(raw: Optional[str]) -> List[str]:
    """解析 --domains 参数。"""

    if not raw or raw.strip().lower() == "all":
        return list(VALID_DOMAINS)
    domains = [domain.strip() for domain in raw.split(",") if domain.strip()]
    invalid = [domain for domain in domains if domain not in VALID_DOMAINS]
    if invalid:
        raise ValueError(f"Unknown domains: {invalid}; expected navigation, crafting, or all")
    return domains


def parse_features(raw: Optional[str]) -> tuple[str, ...]:
    """解析 --features 参数。"""

    if not raw:
        return DEFAULT_FEATURES
    features = tuple(feature.strip() for feature in raw.split(",") if feature.strip())
    return features or DEFAULT_FEATURES


def load_human_data(
    input_dir: Path,
    domain: str,
    participants: Optional[List[str]],
) -> Dict[str, pd.DataFrame]:
    """加载指定任务的预处理人类数据。"""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    data: Dict[str, pd.DataFrame] = {}
    for fp in sorted(input_dir.glob(f"participant_*_{domain}.joblib")):
        pid = fp.stem.split("_")[1]
        if participants is not None and pid not in participants:
            continue
        data[pid] = joblib.load(fp)

    if not data:
        raise ValueError(f"在 {input_dir} 中未找到匹配的 {domain} 数据文件")
    return data


def df_to_state_sequences(df: pd.DataFrame) -> List[List[int]]:
    """从预处理 DataFrame 恢复每个 trial 的有效状态序列。"""

    records = df.sort_values(["trial_index", "step"]).to_dict("records")
    sequences = state_sequences_from_transition_rows(records)
    return [[int(state) for state in seq] for seq in sequences]


def infer_domain_landmarks(
    input_dir: Path,
    domain: str,
    participants: Optional[List[str]],
    config: LandmarkMiningConfig,
) -> Dict[str, Dict]:
    """推断单个 domain 下每个被试的 landmark candidates。"""

    data = load_human_data(input_dir, domain, participants)
    results: Dict[str, Dict] = {}
    for pid, df in data.items():
        sequences = df_to_state_sequences(df)
        print(
            f"[{domain}] participant {pid}: {len(sequences)} trials, "
            f"{sum(max(len(seq) - 1, 0) for seq in sequences)} transitions"
        )
        result = mine_landmarks(sequences, config)
        result["participant"] = pid
        result["domain"] = domain
        results[pid] = result

        stable = result["landmarks"]
        top = result["top_landmarks"]
        print(f"  selected landmarks: {stable}")
        print(f"  top-{config.max_landmarks} by full-data score: {top}")
        for item in result["candidate_ranking"][: config.max_landmarks]:
            print(
                f"    state {item['state']}: score={item['score']:.3f}, " f"selection_rate={item['selection_rate']:.3f}"
            )

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    """构造人类数据 landmark 推断脚本的命令行解析器。"""

    parser = argparse.ArgumentParser(description="从人类状态序列中推断 landmark candidates。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/real_data_pipeline/processed"),
        help="预处理后的人类数据目录 (default: data/real_data_pipeline/processed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real_data_pipeline/inference_results/landmarks.joblib"),
        help="landmark inference 输出文件 (default: data/real_data_pipeline/inference_results/landmarks.joblib)",
    )
    parser.add_argument("--participants", type=str, default="all", help='选择被试: "all", "003", "003,005"')
    parser.add_argument(
        "--domains",
        type=str,
        default="navigation,crafting",
        help='选择任务: "all", "navigation", "crafting", "navigation,crafting"',
    )
    parser.add_argument(
        "--features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help="用于 rank aggregation 的特征，逗号分隔",
    )
    parser.add_argument("--max-landmarks", type=int, default=4, help="每次 bootstrap 选择的最大 landmark 数")
    parser.add_argument("--bootstrap-iterations", type=int, default=500, help="bootstrap/subsampling 次数")
    parser.add_argument("--bootstrap-sample-ratio", type=float, default=0.8, help="每次 subsampling 的 trial 比例")
    parser.add_argument("--selection-threshold", type=float, default=0.7, help="稳定 landmark 的 selection rate 阈值")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=0,
        help="bootstrap 多进程数；0 表示使用 os.cpu_count() (default: 0)",
    )
    parser.add_argument(
        "--include-trial-endpoints-for-commonality",
        action="store_true",
        help="计算 path_commonality 时包含 trial start/goal；默认排除端点",
    )
    return parser


def main() -> None:
    """命令行入口：按 domain/被试运行 landmark 推断并保存结果。"""

    parser = build_arg_parser()
    args = parser.parse_args()

    participants = parse_participants(args.participants)
    domains = parse_domains(args.domains)
    config = LandmarkMiningConfig(
        features=parse_features(args.features),
        max_landmarks=args.max_landmarks,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_sample_ratio=args.bootstrap_sample_ratio,
        selection_threshold=args.selection_threshold,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        exclude_trial_endpoints_for_commonality=not args.include_trial_endpoints_for_commonality,
    )

    print("Landmark inference config:")
    print(f"  features={config.features}")
    print(f"  max_landmarks={config.max_landmarks}")
    print(f"  bootstrap_iterations={config.bootstrap_iterations}")
    print(f"  bootstrap_sample_ratio={config.bootstrap_sample_ratio}")
    print(f"  selection_threshold={config.selection_threshold}")
    print(f"  n_jobs={config.n_jobs or os.cpu_count() or 1}")

    all_results: Dict[str, Dict[str, Dict]] = {}
    for domain in domains:
        all_results[domain] = infer_domain_landmarks(
            input_dir=args.input_dir,
            domain=domain,
            participants=participants,
            config=config,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(all_results, args.output)
    print(f"\n全部完成，landmark 推断结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
