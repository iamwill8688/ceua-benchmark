#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/validate_dataset.py
=========================

自动校验 EA / EU 两个 CEUA Benchmark 数据集的 Schema 与数据完整性。

只做「检查」，绝不修改数据 —— 不修改题目 / subject / scenario /
question / choices / answer，不自动修复错误，不自动补数据。

仅使用 Python 标准库（json / re / sys / pathlib），不依赖
pandas / numpy 等任何第三方库。Windows + VS Code 可直接运行。

用法（在项目根目录，或在任意位置执行均可）：
    python src/validate_dataset.py
    python src/validate_dataset.py <eu_file.jsonl> <ea_file.jsonl>

默认数据文件：
    data/eu.jsonl
    data/ea.jsonl

退出码：
    0  -> 全部检查通过
    1  -> 存在错误（VALIDATION FAILED）
    2  -> 命令行用法错误

检查内容：
    - JSONL 格式：每行合法 JSON、空行、BOM、额外 Markdown/内容、
      一行一条记录、完全重复的记录行
    - Schema：必需字段、字段类型、字段顺序、额外字段
    - 值：qid 格式/重复/连续性，language=zh，module=EU|EA，
      question_type ∈ {single_choice, two_choice}，
      choices 含 A/B/C/D 且为非空字符串，
      answer：single_choice 为单个 A/B/C/D；
              two_choice 为两个不同的 A/B/C/D（如 "CD"/"DC"/"CA"），
      所有必需字段非 null / 非空串 / 非纯空格
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EU = "EU"
EA = "EA"

# 每个模块的顶层字段，且顺序就是记录的期望顺序。
SCHEMA = {
    EU: ["qid", "language", "module", "question_type",
         "question", "subject", "choices", "answer"],
    EA: ["qid", "language", "module", "question_type",
         "scenario", "subject", "question", "choices", "answer"],
}

CHOICE_KEYS = ("A", "B", "C", "D")

# 合法的 question_type 值。
VALID_QTYPES = ("single_choice", "two_choice")

# 期望的 qid 形式：EU_001 / EA_001 ……（三位数字）
QID_RE = {m: re.compile(r"^%s_(\d{3})$" % m) for m in (EU, EA)}
# 用于在“解析失败的行”里尽量读出 qid（best-effort，仅用于报告）
QID_IN_LINE_RE = re.compile(r'"qid"\s*:\s*"([^"]*)"')

DATASETS = [
    {"module": EU, "file": ROOT / "data" / "eu.jsonl"},
    {"module": EA, "file": ROOT / "data" / "ea.jsonl"},
]

# 报告中的固定统计项（顺序按模板）。
STAT_TEMPLATE = [
    ("Records",           "records"),
    ("Valid records",     "valid"),
    ("Invalid records",   "invalid"),
    ("Duplicate qid",     "duplicate_qid"),
    ("Missing qid",       "missing_qid"),
    ("JSON parse errors", "json_parse"),
    ("Schema errors",     "schema"),
    ("Empty fields",      "empty_value"),
    ("Unexpected fields", "unexpected"),
]

# 仅在计数 > 0 时追加的补充统计项。
EXTRA_STATS = [
    ("Empty lines",       "empty_line"),
    ("BOM",               "bom"),
    ("Duplicate records", "duplicate_record"),
    ("Non-JSON content",  "extra_content"),
]

# 每条问题的「错误类型」显示名。
CATEGORY_LABEL = {
    "empty_line":       "Empty Line",
    "bom":              "BOM",
    "json_parse":       "JSON Parse Error",
    "extra_content":    "Non-JSON Content",
    "duplicate_record": "Duplicate Record",
    "duplicate_qid":    "Duplicate qid",
    "qid_gap":          "Missing qid",
    "schema":           "Schema Error",
    "empty_value":      "Empty Field",
    "unexpected":       "Unexpected Field",
    "file_missing":     "File Not Found",
    "encoding":         "Encoding Error",
}

def make_issue(file_path, line, qid, cat, field, problem):
    """构造一条问题记录（file / line / qid / 类型 / 字段 / 原因）。"""
    return {
        "file": str(file_path),
        "line": line,
        "qid": qid,
        "cat": cat,
        "field": field,
        "problem": problem,
    }


def extract_qid_from_line(raw_line):
    """在无法完整解析的行中尽力读取 qid，仅用于报告。"""
    m = QID_IN_LINE_RE.search(raw_line)
    return m.group(1) if m else None


def empty_stats():
    return {
        "records": 0, "valid": 0, "invalid": 0,
        "duplicate_qid": 0, "missing_qid": 0, "json_parse": 0,
        "schema": 0, "empty_value": 0, "unexpected": 0,
        "empty_line": 0, "bom": 0, "duplicate_record": 0,
        "extra_content": 0,
    }


def check_file(cfg):
    """校验一个 JSONL 数据集。返回 (stats, issues)。"""
    module = cfg["module"]
    file_path = cfg["file"]
    expected_fields = list(SCHEMA[module])
    expected_set = set(expected_fields)
    qid_re = QID_RE[module]
    stats = empty_stats()
    issues = []
    bad_lines = set()          # 存在记录级错误的行（用于 valid / invalid）
    qid_refs = []              # (行号, qid, 数字部分)，qid 符合 <MODULE>_NNN
    first_line_of_qid = {}     # qid -> 第一次出现的行号
    seen_text = {}             # 去除两端空白后的整行 -> 第一次出现的行号

    # 问题类别 -> 统计 key
    CAT_STAT = {
        "empty_line": "empty_line", "bom": "bom",
        "json_parse": "json_parse", "extra_content": "extra_content",
        "duplicate_record": "duplicate_record",
        "duplicate_qid": "duplicate_qid", "qid_gap": "missing_qid",
        "schema": "schema", "empty_value": "empty_value",
        "unexpected": "unexpected",
    }

    def add(cat, problem, line=None, qid="-", field=None, marks_line=False):
        """追加一条问题并更新统计计数。"""
        issues.append(make_issue(file_path, line, qid, cat, field, problem))
        key = CAT_STAT.get(cat)
        if key is not None:
            stats[key] += 1
        if marks_line and line is not None:
            bad_lines.add(line)

    if not file_path.is_file():
        add("file_missing", "File not found: %s" % file_path,
            line=None, qid="-")
        return stats, issues

    try:
        raw = file_path.read_bytes()
    except OSError as exc:
        add("file_missing", "Cannot read file: %s" % exc, line=None, qid="-")
        return stats, issues

    if raw.startswith(b"\xef\xbb\xbf"):
        add("bom", "File starts with a UTF-8 BOM (\\ufeff); "
                   "it should be plain UTF-8 text.", line=1, qid="-")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        add("encoding", "File is not valid UTF-8 text: %s" % exc,
            line=1, qid="-")
        return stats, issues

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()

        # --- 空行检查 --------------------------------------------------
        if stripped == "":
            add("empty_line", "Blank line.", line=lineno, qid="-")
            continue

        stats["records"] += 1
        qid_hint = extract_qid_from_line(raw_line)

        # --- 完全重复的记录行 ------------------------------------------
        first_seen = seen_text.get(stripped)
        if first_seen is not None:
            add("duplicate_record",
                "Line is an exact duplicate of line %d." % first_seen,
                line=lineno, qid=qid_hint, marks_line=True)
        else:
            seen_text[stripped] = lineno

        # --- JSONL 每一行必须从 “{” 开始（一条记录） -------------------
        if not (stripped.startswith("{") or stripped.startswith("[")):
            add("extra_content",
                "Line is not a JSON object (looks like extra content / "
                "Markdown?). Snippet: %r" % raw_line[:60],
                line=lineno, qid=qid_hint, marks_line=True)
            continue

        # --- 每行解析出一个 JSON 值 ------------------------------------
        try:
            obj, end = json.JSONDecoder().raw_decode(stripped)
        except ValueError as exc:
            add("json_parse", "JSON decode error: %s" % exc,
                line=lineno, qid=qid_hint, marks_line=True)
            continue

        rest = stripped[end:].strip()
        if rest:
            if rest.startswith("{") or rest.startswith("["):
                problem = ("More than one JSON value on the line; one line "
                           "must hold exactly one record.")
            else:
                problem = ("Trailing non-JSON content after the record: %r"
                           % rest[:60])
            qid_here = qid_hint
            if isinstance(obj, dict) and qid_here is None:
                qid_here = obj.get("qid")
            add("json_parse", problem, line=lineno, qid=qid_here,
                marks_line=True)
            continue

        if not isinstance(obj, dict):
            add("schema",
                "Record is not a JSON object (got %s)."
                % type(obj).__name__,
                line=lineno, qid=qid_hint, marks_line=True)
            continue

        # ================ 逐字段校验一条完整记录 ========================
        record = obj
        qid_value = record.get("qid")
        if not isinstance(qid_value, str):
            qid_value = None

        def radd(cat, problem, field=None, marks_line=True):
            add(cat, problem, line=lineno, qid=qid_value or "-",
                field=field, marks_line=marks_line)

        # 1) 必需字段是否存在
        for f in expected_fields:
            if f not in record:
                radd("schema", 'Missing required field "%s".' % f, field=f)

        # 2) 是否存在 schema 之外的额外字段
        for f in record:
            if f not in expected_set:
                radd("unexpected", 'Unexpected field "%s".' % f, field=f)

        # 3) 字段顺序是否与模板一致
        #    比较“实际出现字段”的顺序与“模板中对应字段”的顺序，
        #    这样缺少/多出的字段只单独报告，不会再次触发顺序报错。
        present_order = [f for f in record if f in expected_set]
        expected_order = [f for f in expected_fields if f in record]
        if present_order != expected_order:
            radd("schema",
                 "Field order mismatch. Expected %s; found %s."
                 % (expected_order, present_order),
                 field="<field order>")

        # 4) 类型 / 空值 / 具体取值检查
        for f in expected_fields:
            if f not in record:
                continue
            value = record[f]

            if f == "choices":
                if value is None:
                    radd("empty_value", 'Field "choices" is null.', field=f)
                    continue
                if not isinstance(value, dict):
                    radd("schema",
                         'Field "choices" must be a JSON object, got %s.'
                         % type(value).__name__, field=f)
                    continue
                _check_choices(radd, value)
                continue

            # 标量字符串字段
            if value is None:
                radd("empty_value", 'Field "%s" is null.' % f, field=f)
                continue
            if not isinstance(value, str):
                radd("schema",
                     'Field "%s" must be a string, got %s.'
                     % (f, type(value).__name__), field=f)
                continue
            if value.strip() == "":
                radd("empty_value",
                     'Field "%s" is empty or only whitespace.' % f, field=f)
                continue

            if f == "language" and value != "zh":
                radd("schema",
                     'Invalid language value "%s"; expected "zh".' % value,
                     field=f)
            elif f == "module" and value != module:
                radd("schema",
                     'Invalid module value "%s"; expected "%s".'
                     % (value, module), field=f)
            elif f == "question_type" and value not in VALID_QTYPES:
                radd("schema",
                     'Invalid question_type "%s"; expected one of %s.'
                     % (value, "/".join(VALID_QTYPES)), field=f)
            elif f == "answer" and not _is_valid_answer(
                    value, record.get("question_type")):
                radd("schema",
                     'Invalid answer "%s" for question_type "%s".'
                     % (value, record.get("question_type")), field=f)
            elif f == "qid":
                if qid_re.match(value) is None:
                    radd("schema",
                         'QID "%s" does not match the pattern %s_001 '
                         '(module prefix + three digits).' % (value, module),
                         field=f)

        # 5) 收集格式正确的 qid，供「重复 qid」和「连续性」检查
        if isinstance(qid_value, str):
            m = qid_re.match(qid_value)
            if m is not None:
                qid_refs.append((lineno, qid_value, int(m.group(1))))

    # ================ 全文件级检查：重复 qid ============================
    for lineno, qid, _num in qid_refs:
        first_line = first_line_of_qid.get(qid)
        if first_line is None:
            first_line_of_qid[qid] = lineno
        else:
            add("duplicate_qid",
                'Duplicate qid "%s" (first seen on line %d).'
                % (qid, first_line),
                line=lineno, qid=qid, field="qid", marks_line=True)

    # ================ 全文件级检查：qid 连续性 ==========================
    # 只对“第一次出现”的 qid 判断顺序，避免与「重复 qid」重复报告。
    unique_refs = []
    seen_qid = set()
    for lineno, qid, num in qid_refs:
        if qid not in seen_qid:
            seen_qid.add(qid)
            unique_refs.append((lineno, qid, num))

    expected_num = 1  # 编号期望从 ..._001 开始
    for lineno, qid, num in unique_refs:
        if num > expected_num:
            for missing in range(expected_num, num):
                add("qid_gap",
                    "Missing qid %s_%03d in the sequence "
                    "(found %s on line %d)."
                    % (module, missing, qid, lineno),
                    line=lineno, qid=qid, field="qid")
            expected_num = num + 1
        elif num == expected_num:
            expected_num += 1
        else:
            add("qid_gap",
                'QID "%s" (line %d) is out of order; expected %s_%03d next.'
                % (qid, lineno, module, expected_num),
                line=lineno, qid=qid, field="qid")

    if stats["records"] == 0:
        add("schema", "Dataset file is empty (no records found).",
            line=None, qid="-")

    stats["invalid"] = len(bad_lines)
    stats["valid"] = stats["records"] - stats["invalid"]
    return stats, issues


def _is_valid_answer(answer, qtype):
    """
    校验 answer 是否符合其 question_type：
      * single_choice -> 必须正好是 A/B/C/D 中单个字母
      * two_choice    -> 必须正好是 A/B/C/D 中两个不同字母（如 "CD"/"DC"/"CA"）
    其它 / 无法解析 -> False。
    """
    if not isinstance(answer, str):
        return False
    if qtype == "single_choice":
        return answer in CHOICE_KEYS
    if qtype == "two_choice":
        if len(answer) != 2:
            return False
        if not set(answer).issubset(set(CHOICE_KEYS)):
            return False
        return len(set(answer)) == 2  # 两个不同选项，拒绝 "AA" 之类重复
    return False


def _check_choices(radd, choices):
    """校验 choices 对象：必须含 A/B/C/D，且每个值为非空字符串。"""
    for key in CHOICE_KEYS:
        if key not in choices:
            radd("schema", 'Missing key "%s" in choices.' % key,
                 field="choices")
    for key, value in choices.items():
        if key not in CHOICE_KEYS:
            radd("schema",
                 'Unexpected key "%s" in choices (only A/B/C/D allowed).'
                 % key, field="choices")
            continue
        if value is None:
            radd("empty_value", 'Choice "%s" is null.' % key,
                 field="choices.%s" % key)
        elif not isinstance(value, str):
            radd("schema",
                 'Choice "%s" must be a non-empty string, got %s.'
                 % (key, type(value).__name__),
                 field="choices.%s" % key)
        elif value.strip() == "":
            radd("empty_value", 'Choice "%s" is empty or only whitespace.' % key,
                 field="choices.%s" % key)


def format_issue(issue):
    """把一条问题按报告的格式打印。"""
    out = ["[ERROR]"]
    out.append("File: %s" % issue["file"])
    out.append("Line: %s" % (issue["line"] if issue["line"] is not None else "-"))
    out.append("QID: %s" % (issue["qid"] if issue["qid"] else "-"))
    out.append("Type: %s" % CATEGORY_LABEL.get(issue["cat"], issue["cat"]))
    if issue["field"] is not None:
        out.append("Field: %s" % issue["field"])
    out.append("Problem: %s" % issue["problem"])
    return "\n".join(out)


def print_report(results):
    """输出统计报告，返回是否全部通过。"""
    print("=" * 32)
    print("Dataset Validation Report")
    print("=" * 32)
    print()
    print("Files checked:")
    for cfg in results:
        missing = "" if cfg["file"].is_file() else "   (file not found)"
        print("  %-4s %s%s" % (cfg["module"], cfg["file"], missing))
    print()
    for cfg in results:
        stats = cfg["stats"]
        print(cfg["module"])
        print("-" * 32)
        for label, key in STAT_TEMPLATE:
            print("%-22s %6d" % (label + ":", stats[key]))
        for label, key in EXTRA_STATS:
            if stats[key] > 0:
                print("%-22s %6d" % (label + ":", stats[key]))
        print()
    return all(len(cfg["issues"]) == 0 for cfg in results)


def print_issues(results):
    """打印所有问题的详细信息。"""
    for cfg in results:
        for issue in cfg["issues"]:
            print(format_issue(issue))
            print()


def main(argv):
    # Windows 控制台 / VS Code 下避免中文或特殊字符编码报错
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = argv[1:]
    if len(args) == 0:
        datasets = [dict(cfg) for cfg in DATASETS]
    elif len(args) == 2:
        datasets = [
            {"module": EU, "file": Path(args[0])},
            {"module": EA, "file": Path(args[1])},
        ]
    else:
        print(__doc__)
        print("Usage: python src/validate_dataset.py "
              "[<eu_file.jsonl> <ea_file.jsonl>]")
        return 2

    results = []
    for cfg in datasets:
        stats, issues = check_file(cfg)
        results.append({"module": cfg["module"], "file": cfg["file"],
                        "stats": stats, "issues": issues})

    all_ok = print_report(results)

    print()
    print("=" * 32)
    if all_ok:
        print("ALL CHECKS PASSED")
        print("=" * 32)
        return 0

    print("VALIDATION FAILED")
    print("=" * 32)
    print()
    print("Problem details:")
    print("-" * 32)
    print_issues(results)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))




