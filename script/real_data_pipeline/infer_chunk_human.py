# -*- coding: utf-8 -*-
"""
从人类 navigation / crafting 实验数据中推断 action chunk 表征。

读取 ``preprocess_human_data.py`` 生成的 joblib 文件，将每个被试的
trial 数据拼接为动作/状态序列，调用 TemporalPatternMiner 或
DiscreteStateTemporalPatternMiner 进行 chunk 推断。

用法::

    python script/real_data_pipeline/infer_chunk_human.py
    python script/real_data_pipeline/infer_chunk_human.py --participants 001
    python script/real_data_pipeline/infer_chunk_human.py --participants 001,002 --no-separator
    python script/real_data_pipeline/infer_chunk_human.py --participants all
"""

import argparse
import sys
from pathlib import Path

# unused import
from typing import Callable, Dict, List, Optional  # noqa: F401

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cognitivemap._env import init  # noqa: E402

init()

from cognitivemap.inference.miner_chunk import TemporalPatternMiner  # noqa: E402
from cognitivemap.inference.miner_chunk_state_conditioned import DiscreteStateTemporalPatternMiner  # noqa: E402

# ============================================================================
# 动作符号转换：人类行为数据 <-> miner token 空间
# ============================================================================

HUMAN_TO_MINER_ACTION = {
    "R": "R",
    "D": "D",
    "L": "L",
    "U": "U",
    "LOOP": "P",
}

MINER_TO_HUMAN_ACTION = {v: k for k, v in HUMAN_TO_MINER_ACTION.items()}


def human_action_to_miner(action: str) -> str:
    """把人类数据中的动作名转换为 chunk miner 使用的单字符 token。"""

    if action not in HUMAN_TO_MINER_ACTION:
        raise ValueError(f"Unknown human action: {action}")
    return HUMAN_TO_MINER_ACTION[action]


def miner_action_to_human(action: str) -> str:
    """把 miner token 转回人类可读动作名，保留 trial 分隔符 ``S``。"""

    if action == "S":
        return "S"
    if action not in MINER_TO_HUMAN_ACTION:
        raise ValueError(f"Unknown miner action: {action}")
    return MINER_TO_HUMAN_ACTION[action]


def miner_chunk_to_human(chunk: str) -> str:
    """把 miner 空间的复合 chunk 转换为人类动作名序列。"""

    return "-".join(miner_action_to_human(token) for token in chunk.split("-"))


# ============================================================================
# 被试选择
# ============================================================================


def parse_participants(raw: Optional[str]) -> Optional[List[str]]:
    """解析 --participants 参数。

    ``"all"`` 或空字符串 → None（全部被试）。
    ``"001"`` → ["001"]，``"001,002"`` → ["001", "002"]。
    """
    if not raw:
        return None
    raw = raw.strip()
    if raw.lower() == "all":
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def parse_domains(raw: Optional[str]) -> List[str]:
    """解析 --domains 参数。

    ``"all"`` 或空字符串 → ["navigation", "crafting"]。
    ``"navigation"`` → ["navigation"]，``"navigation,crafting"`` → 两个 domain。
    """
    if not raw:
        return ["navigation", "crafting"]
    raw = raw.strip()
    if raw.lower() == "all":
        return ["navigation", "crafting"]

    domains = [domain.strip() for domain in raw.split(",") if domain.strip()]
    valid = {"navigation", "crafting"}
    invalid = [domain for domain in domains if domain not in valid]
    if invalid:
        raise ValueError(f"Unknown domains: {invalid}; expected navigation, crafting, or all")
    return domains


# ============================================================================
# 人类数据加载
# ============================================================================


def load_human_data(
    input_dir: Path,
    domain: str,
    participants: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """加载预处理后的人类指定 domain 数据。

    Returns:
        {participant_id: DataFrame} 映射。
    """
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


# ============================================================================
# DataFrame → miner 可用 trial
# ============================================================================


def df_to_trials(df: pd.DataFrame, use_state: bool) -> List[Dict]:
    """将一个被试的 DataFrame 转换为 miner-ready trial 列表。

    每个 trial 包含 actions 列表和 states 列表。
    states 与 actions 逐位对齐（取 step 执行前的 state）。
    """
    trials: List[Dict] = []
    for trial_id, group in df.sort_values("step").groupby("trial_id", sort=False):
        # 只保留有效动作（valid=True）——无效动作不改变状态，挖掘意义不大
        valid_steps = group[group["valid"] == True]  # noqa: E712
        if valid_steps.empty:
            continue

        actions = [human_action_to_miner(str(a)) for a in valid_steps["action"]]
        states = None
        if use_state:
            states = [int(s) for s in valid_steps["state"]]
            if len(states) != len(actions):
                raise ValueError(f"trial {trial_id}: states/actions 数量不匹配")

        trials.append({"actions": actions, "states": states})

    return trials


# ============================================================================
# trial 拼接策略
# ============================================================================


def join_trials_with_separator(trials: List[Dict]) -> tuple[List[str], Optional[List[int]]]:
    """在 trial 之间插入 ``S`` 分隔符。"""
    sequence: List[str] = []
    state_sequence: Optional[List[int]] = [] if trials and trials[0]["states"] is not None else None
    for idx, trial in enumerate(trials):
        if idx > 0:
            sequence.append("S")
            if state_sequence is not None:
                state_sequence.append(-1)
        sequence.extend(trial["actions"])
        if state_sequence is not None:
            state_sequence.extend(trial["states"])
    return sequence, state_sequence


def constr_func_separator(parent: str, child: str) -> bool:
    """禁止候选 chunk 跨越 trial 分隔符 ``S``。"""

    return "S" not in parent and "S" not in child


def join_trials_directly(trials: List[Dict]) -> tuple[List[str], Optional[List[int]]]:
    """直接拼接 trial，无分隔符。"""
    sequence: List[str] = []
    state_sequence: Optional[List[int]] = [] if trials and trials[0]["states"] is not None else None
    for trial in trials:
        sequence.extend(trial["actions"])
        if state_sequence is not None:
            state_sequence.extend(trial["states"])
    return sequence, state_sequence


def constr_func_no_separator(parent: str, child: str) -> bool:
    """无分隔符模式下的基础约束：允许所有候选再交给长度约束筛选。"""

    return True


def chunk_token_length(chunk: str) -> int:
    """返回一个原子 token 或复合 chunk 覆盖的 primitive token 数量。"""

    return len(str(chunk).split("-"))


def candidate_chunk_length(parent: str, child: str) -> int:
    """返回候选 ``parent-child`` 合成后的 token 长度。"""

    return chunk_token_length(parent) + chunk_token_length(child)


def compose_constr_func(base_constr_func, max_chunk_length: int | None):
    """组合基础约束与最大 chunk 长度约束。"""

    if max_chunk_length is not None and max_chunk_length <= 0:
        raise ValueError("max_chunk_length must be positive or None")

    def constr_func(parent: str, child: str) -> bool:
        """组合后的候选合法性检查函数。"""

        if not base_constr_func(parent, child):
            return False
        if max_chunk_length is None:
            return True
        return candidate_chunk_length(parent, child) <= max_chunk_length

    return constr_func


# ============================================================================
# 按被试运行推断
# ============================================================================


def infer_domain_chunks(
    input_dir: Path,
    domain: str,
    separator: bool = True,
    participants: Optional[List[str]] = None,
    ess: float = 10.0,
    use_state: bool = True,
    max_chunk_length: int | None = 3,
    miner_kwargs: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """加载单个 domain 的人类数据并运行 chunk 推断。

    Returns:
        {participant_id: inference_result} 映射。
    """
    data = load_human_data(input_dir, domain, participants)
    miner_kwargs = dict(miner_kwargs or {})

    if separator:
        miner_cls = TemporalPatternMiner
        join_trials_func = join_trials_with_separator
        constr_func = compose_constr_func(constr_func_separator, max_chunk_length)
        miner_kwargs.setdefault("structural_tokens", ("S",))
    else:
        miner_cls = DiscreteStateTemporalPatternMiner
        join_trials_func = join_trials_directly
        constr_func = compose_constr_func(constr_func_no_separator, max_chunk_length)

    all_results: Dict[str, Dict] = {}

    for pid, df in sorted(data.items()):
        trials = df_to_trials(df, use_state=use_state)
        sequence, state_sequence = join_trials_func(trials)
        effective_action_count = sum(len(t["actions"]) for t in trials)

        miner = miner_cls(**miner_kwargs)
        miner_result = miner.mine_patterns(
            sequence,
            ess=ess,
            constr_func=constr_func,
            state_sequence=state_sequence if use_state else None,
        )

        # 转换回人类动作符号
        parsed = [miner_chunk_to_human(c) for c in miner_result["sequence"]]
        parsed_state = list(miner_result["state_sequence"]) if miner_result["state_sequence"] is not None else None
        original = [miner_action_to_human(t) for t in sequence]
        original_state = list(state_sequence) if state_sequence is not None else None

        vocab = [miner_chunk_to_human(c) for c in miner_result["vocab"]]
        learned = sorted([c for c in vocab if "-" in c], key=lambda x: (len(x.split("-")), x))

        print(
            f"{domain} participant {pid}: {len(trials)} trials, {effective_action_count} actions, "
            f"{len(learned)} inferred chunks: {learned}",
            flush=True,
        )

        all_results[pid] = {
            "participant_id": pid,
            "domain": domain,
            "trial_count": len(trials),
            "input_token_count": len(sequence),
            "effective_action_count": effective_action_count,
            "vocab": vocab,
            "learned_chunks": learned,
            "original_action_sequence": original,
            "original_state_sequence": original_state,
            "parsed_action_sequence": parsed,
            "parsed_state_sequence": parsed_state,
            "components": {
                miner_chunk_to_human(k): [miner_chunk_to_human(c) for c in v]
                for k, v in miner_result["components"].items()
            },
            "js_history": miner_result["js_history"],
        }

    return all_results


def infer_human_chunks(
    input_dir: Path,
    domains: List[str],
    separator: bool = True,
    participants: Optional[List[str]] = None,
    ess: float = 10.0,
    use_state: bool = True,
    max_chunk_length: int | None = 3,
    miner_kwargs: Optional[Dict] = None,
) -> Dict[str, Dict[str, Dict]]:
    """按 domain 加载人类数据并运行 chunk 推断。

    Returns:
        {domain: {participant_id: inference_result}} 嵌套映射。
    """
    results: Dict[str, Dict[str, Dict]] = {}
    for domain in domains:
        results[domain] = infer_domain_chunks(
            input_dir=input_dir,
            domain=domain,
            separator=separator,
            participants=participants,
            ess=ess,
            use_state=use_state,
            max_chunk_length=max_chunk_length,
            miner_kwargs=miner_kwargs,
        )
    return results


def write_results_joblib(output_path: Path, results: Dict[str, Dict]) -> None:
    """将 chunk 推断结果写入 joblib 文件。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, output_path)
    print(f"Results saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """命令行入口：加载预处理数据、运行 chunk 推断并保存结果。"""

    parser = argparse.ArgumentParser(description="从人类 navigation / crafting 数据中推断 action chunk 表征。")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/real_data_pipeline/processed"),
        help="预处理后的人类数据目录 (default: data/real_data_pipeline/processed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real_data_pipeline/inference_results/chunks.joblib"),
        help="chunk 推断结果输出 joblib 路径",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="navigation,crafting",
        help='选择任务域: "all", "navigation", "crafting", "navigation,crafting" (default: navigation,crafting)',
    )
    parser.add_argument(
        "--participants",
        type=str,
        default="all",
        help='选择被试: "all", "001", "001,002" 等 (default: all)',
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--separator",
        action="store_true",
        dest="separator",
        default=True,
        help="使用 TemporalPatternMiner，trial 间插入分隔符 (default)",
    )
    mode_group.add_argument(
        "--no-separator",
        action="store_false",
        dest="separator",
        help="使用 DiscreteStateTemporalPatternMiner，trial 直接拼接",
    )

    parser.add_argument("--ess", type=float, default=10.0, help="BDeu 等效样本量 (default: 50.0)")
    parser.add_argument(
        "--use-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将 state_sequence 作为额外父变量",
    )
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument(
        "--max-chunk-length",
        type=int,
        default=3,
        help="BDeu 候选 chunk 的最大 primitive token 长度 (default: 3)",
    )
    parser.add_argument("--js-threshold", type=float, default=0.05)
    parser.add_argument("--convergence-window", type=int, default=5)
    parser.add_argument(
        "--min-joint-probability", type=float, default=None, help="候选最小联合频率 (default: 0.05 sep / 0.02 nosep)"
    )
    parser.add_argument(
        "--score-ratio-threshold", type=float, default=None, help="候选分数阈值 (default: 0.99 sep / 0.9 nosep)"
    )

    args = parser.parse_args()

    min_joint = args.min_joint_probability
    if min_joint is None:
        min_joint = 0.05
    score_ratio = args.score_ratio_threshold
    if score_ratio is None:
        score_ratio = 0.9

    results = infer_human_chunks(
        input_dir=args.input_dir,
        domains=parse_domains(args.domains),
        separator=args.separator,
        participants=parse_participants(args.participants),
        ess=args.ess,
        use_state=args.use_state,
        max_chunk_length=args.max_chunk_length,
        miner_kwargs={
            "max_iterations": args.max_iterations,
            "js_threshold": args.js_threshold,
            "convergence_window": args.convergence_window,
            "min_joint_probability": min_joint,
            "score_ratio_threshold": score_ratio,
        },
    )
    write_results_joblib(args.output, results)


if __name__ == "__main__":
    main()
