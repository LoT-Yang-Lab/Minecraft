# -*- coding: utf-8 -*-
"""
人类实验数据预处理脚本
======================
从 jsPsych 原始 JSON 中提取 navigation 和 crafting 的 main 阶段数据，
将节点统一转换为整数坐标（stone_01→1，或直接使用 from_code/to_code），
按键映射为抽象动作（a→U, d→D, q→L, e→R, w→LOOP），
每个被试每个 domain 输出一个 joblib 文件。

用法::

    python script/real_data_pipeline/preprocess_human_data.py
    python script/real_data_pipeline/preprocess_human_data.py --participant 001
"""

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cognitivemap._env import init  # noqa: E402

init()

# 按键 → 抽象动作映射（与 mix_nc 数据中的 dir_code 一致）
KEY_TO_ACTION = {
    "a": "U",  # a → UP
    "d": "D",  # d → DOWN
    "q": "L",  # q → LEFT
    "e": "R",  # e → RIGHT
    "w": "LOOP",  # w → 角点顺时针跳转
}

# DataFrame 列顺序
OUTPUT_COLUMNS = [
    "trial_id",
    "raw_trial_id",
    "domain",
    "trial_index",
    "start",
    "goal",
    "outcome",
    "total_steps",
    "total_time_ms",
    "step",
    "state",
    "key",
    "action",
    "next_state",
    "valid",
    "order_completed",
    "rt_ms",
]

_STONE_RE = re.compile(r"stone_(\d+)")


def node_to_int(value: object) -> int:
    """将节点编码转换为整数。

    支持 ``stone_XX``、整数、以及数字字符串。
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    m = _STONE_RE.match(str(value))
    if m:
        return int(m.group(1))
    if str(value).isdigit():
        return int(str(value))
    raise ValueError(f"无法解析节点编码: {value}")


def extract_main_trials(data: list[dict], domain: str) -> list[dict]:
    """从原始 jsPsych 数据中提取指定 domain 的 main 阶段 trial 列表。

    每个返回元素是一个 dict，包含 trial 级字段、展开的 steps 列表，
    以及按提取顺序生成的唯一 ``trial_id``（如 ``navigation_001``）。
    """
    trials = []
    count = 0
    for entry in data:
        if "value" not in entry:
            continue
        val = entry["value"]
        if val.get("phase") != "main":
            continue
        if val.get("domain") != domain:
            continue
        count += 1
        val = dict(val)
        val["raw_trial_id"] = val.get("trial_id")
        val["trial_id"] = f"{domain}_{count:03d}"
        trials.append(val)
    return trials


def trials_to_dataframe(trials: list[dict]) -> pd.DataFrame:
    """将 trial 列表展开为每行一个 step 的 DataFrame。

    支持两类 step schema：
        - navigation/new mix: from_code/to_code
        - crafting/old mix: prev/next/order_target
    """
    rows = []
    for trial in trials:
        trial_id = trial["trial_id"]
        raw_trial_id = trial.get("raw_trial_id")
        domain = trial["domain"]
        outcome = trial["outcome"]
        total_steps = trial["total_steps"]
        total_time_ms = trial["total_time_ms"]
        trial_index = trial["trial_index"]

        steps = trial["steps"]
        if not steps:
            continue

        start = _trial_start(trial, steps)
        goal = _trial_goal(trial, steps)

        for step in steps:
            state = _step_state(step)
            key = step["key"]
            valid = step["valid"]
            rt_ms = step["rt_ms"]
            next_state = _step_next_state(step, state)

            rows.append(
                {
                    "trial_id": trial_id,
                    "raw_trial_id": raw_trial_id,
                    "domain": domain,
                    "trial_index": trial_index,
                    "start": start,
                    "goal": goal,
                    "outcome": outcome,
                    "total_steps": total_steps,
                    "total_time_ms": total_time_ms,
                    "step": step["step"],
                    "state": state,
                    "key": key,
                    "action": KEY_TO_ACTION.get(key, key),
                    "next_state": next_state,
                    "valid": valid,
                    "order_completed": step.get("order_completed"),
                    "rt_ms": rt_ms,
                }
            )

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df


def _trial_start(trial: dict, steps: list[dict]) -> int:
    """读取 trial 起点。"""
    if trial.get("start_code") is not None:
        return node_to_int(trial["start_code"])
    return _step_state(steps[0])


def _trial_goal(trial: dict, steps: list[dict]) -> int:
    """读取 trial 目标。"""
    if trial.get("goal_code") is not None:
        return node_to_int(trial["goal_code"])
    if trial.get("goal_codes"):
        return node_to_int(trial["goal_codes"][0])
    for step in steps:
        if step.get("order_target") is not None:
            return node_to_int(step["order_target"])
    raise ValueError(f"trial {trial.get('trial_id')} 缺少 goal_code/goal_codes/order_target")


def _step_state(step: dict) -> int:
    """读取 step 执行前状态。"""
    if step.get("from_code") is not None:
        return node_to_int(step["from_code"])
    if step.get("prev") is not None:
        return node_to_int(step["prev"])
    raise ValueError(f"step 缺少 from_code/prev: {step}")


def _step_next_state(step: dict, state: int) -> int:
    """读取 step 执行后状态；无效动作保持在原状态。"""
    if step.get("valid") is not True:
        return state
    if step.get("to_code") is not None:
        return node_to_int(step["to_code"])
    if step.get("next") is not None:
        return node_to_int(step["next"])
    return state


def process_one_file(input_path: Path, output_dir: Path) -> list[Path]:
    """处理单个被试文件，返回输出路径列表。"""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    participant_id = None
    for entry in data:
        if "participant_id" in entry:
            participant_id = str(entry["participant_id"]).zfill(3)
            break
    if participant_id is None:
        participant_id = input_path.stem[:20]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for domain in ("navigation", "crafting"):
        trials = extract_main_trials(data, domain)
        df = trials_to_dataframe(trials)
        output_path = output_dir / f"participant_{participant_id}_{domain}.joblib"
        joblib.dump(df, output_path)
        output_paths.append(output_path)

        unique_trials = df["trial_id"].nunique() if "trial_id" in df.columns else len(trials)
        print(f"  participant {participant_id} {domain}: {unique_trials} trials, {len(df)} steps → {output_path}")
    return output_paths


def main() -> None:
    """命令行入口：遍历原始 JSON，被试级别输出预处理后的 joblib。"""

    parser = argparse.ArgumentParser(description="预处理人类实验数据，提取 navigation 和 crafting 阶段数据。")
    parser.add_argument("--input-dir", default="data/real_data_pipeline/raw", help="原始数据目录")
    parser.add_argument("--output-dir", default="data/real_data_pipeline/processed", help="预处理结果输出目录")
    parser.add_argument("--participant", default=None, help="只处理指定被试 (如 001)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    if not files:
        print(f"在 {input_dir} 中未找到 JSON 文件")
        return

    print(f"找到 {len(files)} 个文件")
    for fp in files:
        if args.participant and args.participant not in fp.name:
            continue
        process_one_file(fp, Path(args.output_dir))


if __name__ == "__main__":
    main()
