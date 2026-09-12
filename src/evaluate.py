#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/evaluate.py
===============

CEUA Benchmark (EU / EA) 评测脚本 —— 比较
    * benchmark gold answer （来自数据文件，绝不修改）
    * model prediction     （来自预测文件，JSONL）
并输出准确率报告与可选 JSON 结果。

设计原则
--------
* 只读取数据，绝不回写 / 修改任何原始数据文件。
* 只用 Python 标准库（json / argparse / sys / pathlib / re）。
* 不写死题目数量、模块数量、qid；未来数据扩充也能直接运行。
* qid 全局一致性检查：gold 重复、prediction 重复、prediction 中不存在
  的 qid、gold 中缺失的 prediction 等严重问题一律报错退出，绝不悄悄继续。

Gold answer 的规范化
--------------------
single_choice  -> 单个选项字母，如 "B"  -> {"B"}
two_choice     -> 两个选项组成集合，如 "CD" -> {"C","D"}，"DC" -> {"D","C"}
上述转换只发生在内存中，绝不对 JSONL 做任何回写。

Prediction 格式
---------------
每行一个 JSON object，必须含 qid 与 prediction，例如：
    {"qid": "EU_001", "prediction": "B"}
    {"qid": "EA_014", "prediction": "CD"}

prediction 的规范化统一由 src/utils.py::normalize_prediction() 负责（pipeline 与
evaluator 完全相同），因此 prediction 文件既可以直接写规范写法，也可以写
"答案：C" / "Answer: C" / "I choose D" 这类标准回答格式，都会得到同一结果。
评测使用「Exact Match」：single_choice 要求恰好 1 个有效选项；two_choice 要求
恰好 2 个不同有效选项（集合语义，CD == DC）。写错个数、重复选项、含非法字母或
拒答的 prediction 一律判为 invalid（= 0 分），绝不猜测。

用法示例
--------
    python src/evaluate.py \\
        --data data/eu.jsonl data/ea.jsonl \\
        --predictions predictions.jsonl

    python src/evaluate.py \\
        --data data/eu.jsonl \\
        --predictions predictions.jsonl

退出码：
    0 -> 正常完成评测并输出报告
    1 -> 评测失败（数据/预测存在严重一致性问题、JSON 解析错误、文件缺失）
    2 -> 命令行用法错误
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:  # 作为包导入（python -m src.evaluate / import src.evaluate）
    from . import utils as _u
except ImportError:  # 作为脚本直接运行（python src/evaluate.py）
    import utils as _u

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
VALID_OPTIONS = ("A", "B", "C", "D")
VALID_SET = frozenset(VALID_OPTIONS)

SINGLE = "single_choice"
TWO = "two_choice"
VALID_QTYPES = (SINGLE, TWO)

# 已知模块（报告中单独分列），未知模块归入 overall 但不在 EU/EA 中分列。
KNOWN_MODULES = ("EU", "EA")

# 评测依赖的字段。
REQUIRED_FIELDS = ("qid", "question_type", "answer")


# ---------------------------------------------------------------------------
# 答案解析（只处理内存中的字符串，绝不动数据文件）
# ---------------------------------------------------------------------------
def _tokenize(text):
    """把输入统一成大写，并把常见分隔符折叠为空格后按空白/逗号切分。"""
    s = text.strip().upper()
    for ch in ("，", "、", "；", ";", "/", "|"):
        s = s.replace(ch, " ")
    tokens = [tok for tok in re.split(r"[,\s]+", s) if tok]
    return tokens


def exact_canonical(text, question_type):
    """
    用于 Exact Match 的规范化 —— 直接复用 src/utils.py::normalize_prediction()，
    保证 pipeline（生成 prediction）与 evaluator（打分）使用同一套规则。

    single_choice : 必须恰好 1 个不同选项，返回 {letter}。
    two_choice    : 必须恰好 2 个不同选项，返回 {x, y}（顺序无关，CD == DC）。
    其它情况返回 None（= invalid，计 0 分）。
    """
    prediction = _u.normalize_prediction(text, question_type)
    if not prediction:
        return None
    return frozenset(prediction)


def partial_canonical(text, question_type):
    """
    用于「Partial Score（仅 two_choice）」的宽松规范化。

    two_choice 允许模型只给出 1 个或 2 个不同有效选项（例如只答 "C"），
    以便计算部分命中。规则：
      * 去掉空白 / 逗号等分隔符后，总共只能有 1 或 2 个字符；
      * 每个字符必须是 A/B/C/D；
      * 不允许出现重复选项（如 "CC"），也不允许出现 3/4 个选项（如 "BCD"）；
    不满足上述任一条件都返回空集（= partial 0），绝不误判 / 误给分。
    single_choice 不使用该函数（保持空集）。
    """
    chars = _concat_chars(text)
    if chars is None:
        return frozenset()
    if question_type == SINGLE:
        if len(chars) == 1:
            return frozenset(chars)
        return frozenset()
    if question_type == TWO:
        if len(chars) in (1, 2) and len(set(chars)) == len(chars):
            return frozenset(chars)
        return frozenset()
    return frozenset()


def _concat_chars(text):
    """
    把文本统一转大写、去掉空白与分隔符，返回全部字符的列表（保留重复）。

    任何字符不在 A/B/C/D 内、或没有任何字符时返回 None。
    供 partial_canonical 使用（保留重复，以便区分 "C" 与 "CC"）。
    """
    if not isinstance(text, str):
        return None
    tokens = _tokenize(text)
    if not tokens:
        return None
    chars = [c for tok in tokens for c in tok]
    if any(c not in VALID_SET for c in chars):
        return None
    return chars


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def read_jsonl(path):
    """读取 JSONL，返回记录列表；出现解析错误则抛出带行号的异常。"""
    rows = []
    path = Path(path)
    try:
        fh = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError("无法打开文件 %s：%s" % (path, exc))
    with fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "JSON 解析错误：文件 %s 第 %d 行不是合法 JSON（%s）。"
                    % (path, lineno, exc)
                )
            if not isinstance(obj, dict):
                raise ValueError(
                    "JSON 结构错误：文件 %s 第 %d 行不是 JSON object。"
                    % (path, lineno)
                )
            rows.append(obj)
    return rows


def load_gold(paths):
    """
    从若干数据文件合并出 gold 词典。

    返回 dict[qid] -> {"module", "question_type", "gold": frozenset}
    并对 qid 重复、字段缺失、question_type 非法、answer 无法解析等情况
    抛出明确的 ValueError。
    """
    gold = {}
    for p in paths:
        for row in read_jsonl(p):
            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                raise ValueError(
                    "文件 %s 中记录缺少字段 %s（qid=%s）。"
                    % (p, ", ".join(missing), row.get("qid"))
                )
            qid = row["qid"]
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError("文件 %s 中存在空或非字符串的 qid。" % p)
            if qid in gold:
                raise ValueError(
                    "gold qid 重复：%s（同时出现在 %s）。" % (qid, p)
                )
            qtype = row["question_type"]
            if qtype not in VALID_QTYPES:
                raise ValueError(
                    "非法 question_type=%r（qid=%s），只允许 %s。"
                    % (qtype, qid, "/".join(VALID_QTYPES))
                )
            answer = row["answer"]
            canon = exact_canonical(answer, qtype)
            if canon is None:
                raise ValueError(
                    "gold answer 无法解析（qid=%s, question_type=%s, answer=%r）。"
                    % (qid, qtype, answer)
                )

            module = row.get("module")
            if module not in KNOWN_MODULES:
                # 回退：从 qid 前缀推断已知模块
                module = qid[:2] if qid[:2] in KNOWN_MODULES else module
            gold[qid] = {
                "module": module,
                "question_type": qtype,
                "gold": canon,
                "source": str(p),
            }
    if not gold:
        raise ValueError("没有读取到任何 gold 题目（请检查 --data 路径）。")
    return gold


def load_predictions(path):
    """
    读取 predictions JSONL。

    返回 list of dict。对重复 qid、缺少 qid、缺少 prediction 等问题
    直接抛出明确异常。
    """
    records = []
    for row in read_jsonl(path):
        if "qid" not in row or not isinstance(row.get("qid"), str) \
                or not row["qid"].strip():
            raise ValueError(
                "预测文件中存在缺少 / 非法 qid 字段的记录：%r。" % row
            )
        if "prediction" not in row:
            raise ValueError(
                "预测文件中 qid=%s 的记录缺少 prediction 字段。" % row["qid"]
            )
        records.append({"qid": row["qid"], "prediction": row["prediction"]})
    seen = set()
    for r in records:
        if r["qid"] in seen:
            raise ValueError("prediction qid 重复：%s。" % r["qid"])
        seen.add(r["qid"])
    return records


# ---------------------------------------------------------------------------
# 一致性检查（gold 与 prediction 对照）
# ---------------------------------------------------------------------------
def check_consistency(gold, predictions):
    """
    检查：
      * prediction 中出现 gold 里不存在的 qid
      * gold 中存在 prediction 缺失的 qid
    任一问题抛出明确异常。
    """
    pred_qids = {p["qid"] for p in predictions}
    unknown = pred_qids - set(gold)
    if unknown:
        raise ValueError(
            "prediction 中存在不存在的 qid（gold 中找不到）：%s。"
            % ", ".join(sorted(unknown))
        )
    missing = set(gold) - pred_qids
    if missing:
        raise ValueError(
            "gold 中存在 prediction 缺失的 qid（未作答）：%s。"
            % ", ".join(sorted(missing))
        )


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------
def _rate(numerator, denominator):
    """返回比率；分母为 0 时返回 None。"""
    if denominator == 0:
        return None
    return numerator / denominator


def score_all(gold, predictions):
    """
    为每条 gold 题目打分，返回整体聚合指标字典。

    主指标一律是 Exact Match Accuracy。
    two_choice 额外计算 partial（部分命中），只作为附加指标。
    """
    pred_map = {p["qid"]: p["prediction"] for p in predictions}
    per_q = {}
    for qid, g in gold.items():
        pred = pred_map[qid]
        canon = exact_canonical(pred, g["question_type"])
        exact = canon is not None and canon == g["gold"]

        if g["question_type"] == TWO:
            pset = partial_canonical(pred, TWO)
            partial = len(pset & g["gold"]) / len(g["gold"])
        else:
            partial = None

        per_q[qid] = {
            "qid": qid,
            "module": g["module"],
            "question_type": g["question_type"],
            "exact_correct": exact,
            "partial": partial,
        }
    return per_q


def aggregate(per_q):
    """把逐题打分聚合成指标字典。"""
    def acc(predicate):
        subset = [v for v in per_q.values() if predicate(v)]
        if not subset:
            return None
        return _rate(
            sum(1 for v in subset if v["exact_correct"]), len(subset)
        )

    two_subset = [v for v in per_q.values() if v["question_type"] == TWO]
    two_partial = (
        _rate(sum(v["partial"] for v in two_subset), len(two_subset))
        if two_subset
        else None
    )

    return {
        "total_questions": len(per_q),
        "correct": sum(1 for v in per_q.values() if v["exact_correct"]),
        "accuracy": acc(lambda v: True),
        "eu_accuracy": acc(lambda v: v["module"] == "EU"),
        "ea_accuracy": acc(lambda v: v["module"] == "EA"),
        "single_choice_accuracy": acc(
            lambda v: v["question_type"] == SINGLE
        ),
        "two_choice_accuracy": acc(lambda v: v["question_type"] == TWO),
        "ea_single_choice_accuracy": acc(
            lambda v: v["module"] == "EA" and v["question_type"] == SINGLE
        ),
        "ea_two_choice_accuracy": acc(
            lambda v: v["module"] == "EA" and v["question_type"] == TWO
        ),
        "two_choice_partial_score": two_partial,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _pct(value):
    """把比率(0..1)格式化为百分数字符串；None 时返回 'N/A'。"""
    if value is None:
        return "N/A"
    return "%.2f%%" % (value * 100.0)


def print_report(metrics, data_files):
    line = "=" * 40
    print(line)
    print("Benchmark Evaluation Results")
    print(line)
    print()
    print("Data files:")
    for f in data_files:
        print("  %s" % f)
    print()
    print("Total Questions: %d" % metrics["total_questions"])
    print("Correct (Exact Match): %d" % metrics["correct"])
    print()
    print("Overall Exact Match Accuracy: %s"
          % _pct(metrics["accuracy"]))
    print()
    print("EU Accuracy: %s" % _pct(metrics["eu_accuracy"]))
    print("EA Accuracy: %s" % _pct(metrics["ea_accuracy"]))
    print()
    print("Single-choice Accuracy: %s" % _pct(metrics["single_choice_accuracy"]))
    print("Two-choice Accuracy: %s" % _pct(metrics["two_choice_accuracy"]))
    print()
    print("EA Single-choice Accuracy: %s"
          % _pct(metrics["ea_single_choice_accuracy"]))
    print("EA Two-choice Accuracy: %s"
          % _pct(metrics["ea_two_choice_accuracy"]))
    print()
    print("Two-choice Partial Score: %s"
          % _pct(metrics["two_choice_partial_score"]))
    print(line)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="CEUA Benchmark (EU / EA) 评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python src/evaluate.py --data data/eu.jsonl data/ea.jsonl \\\n"
            "      --predictions predictions.jsonl\n"
            "  python src/evaluate.py --data data/eu.jsonl \\\n"
            "      --predictions predictions.jsonl\n"
        ),
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        metavar="FILE",
        help="一个或多个 gold 数据 JSONL 文件（可同时传 EU/EA）。",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        metavar="FILE",
        help="模型预测 JSONL 文件（每行含 qid 与 prediction）。",
    )
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # 兼容 Windows / VS Code 控制台的中文输出
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        gold = load_gold(args.data)
        predictions = load_predictions(args.predictions)
        check_consistency(gold, predictions)
    except ValueError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        print("评测中止（未计算指标）。请修正数据 / 预测文件后重试。",
              file=sys.stderr)
        return 1

    per_q = score_all(gold, predictions)
    metrics = aggregate(per_q)
    print_report(metrics, args.data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
