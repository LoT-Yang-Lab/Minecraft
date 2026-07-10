# -*- coding: utf-8 -*-
"""
生成 landmark inference recovery 流程使用的九宫格模拟数据。

本脚本只负责使用 ``cognitivemap.generative_model`` 生成 synthetic trajectory pools，
并把 true landmark、trial pool、生成参数等保存为 joblib。后续恢复验证脚本可以读取
该文件，复用同一批模拟数据做 landmark inference recovery。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from validate_landmark_recovery import (  # noqa: E402
    DEFAULT_SIMULATED_DATA_OUTPUT,
    add_generation_arguments,
    add_miner_arguments,
    apply_quick_defaults,
    build_conditions,
    build_simulated_dataset_payload,
    generate_sequence_pools,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """构造模拟数据生成脚本的命令行解析器。"""

    parser = argparse.ArgumentParser(description="Generate synthetic landmark inference recovery datasets.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SIMULATED_DATA_OUTPUT,
        help=f"generated synthetic dataset joblib path (default: {DEFAULT_SIMULATED_DATA_OUTPUT})",
    )
    add_generation_arguments(parser)
    # 当前 ValidationCondition 仍记录 miner 参数；生成阶段不使用这些参数，但保留它们可保证
    # 生成数据文件能被 validate_landmark_recovery.py 直接复用。
    add_miner_arguments(parser)
    return parser


def main() -> None:
    """脚本入口：生成轨迹池 payload 并保存为 joblib。"""

    args = build_arg_parser().parse_args()
    args.simulated_data_output = args.output
    args = apply_quick_defaults(args)
    args.output = args.simulated_data_output

    conditions = build_conditions(args)
    pools = generate_sequence_pools(conditions)
    payload = build_simulated_dataset_payload(args, conditions, pools)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output)

    print(f"Saved simulated dataset to: {args.output}")
    print(f"  conditions: {len(conditions)}")
    print(f"  sequence pools: {len(pools)}")


if __name__ == "__main__":
    main()
