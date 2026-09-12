# -*- coding: utf-8 -*-
"""
src/data.py
===========

数据加载与 Prompt 构造。

硬性要求：
  * 只读数据文件，绝不修改 / 覆盖 / 回写任何数据。
  * 绝不把 gold answer 发送给模型（构造 prompt 时只取 question /
    subject / scenario / choices / question_type 等非答案字段）。
"""

import json
from pathlib import Path

# 在 README 中说明确定性解码（temperature=0 等）。
SYSTEM_PROMPT = (
    "你正在参加一个中文情感智能测试（Emotional Understanding / Application）。\n"
    "请认真分析题目与选项，选择最符合题意的答案。\n"
    "你必须只输出最终答案选项字母，不要输出任何解释文字。\n"
    "单选题只输出一个选项字母：A、B、C 或 D。\n"
    "双选题输出两个不同的选项字母（无逗号、无空格），例如：AB、AC、BD、CD。\n"
    "不要输出其他文字。"
)

QTYPE_LABEL = {
    "single_choice": "本题为单选题，请只选择一个选项。",
    "two_choice": "本题为双选题，请选择两个选项。",
}


def read_jsonl(path):
    """读取一个 JSONL 文件，返回记录列表（原始 dict 的浅拷贝副本）。"""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as fh:
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
    if not rows:
        raise ValueError("数据文件为空：%s" % path)
    return rows


def module_to_files(module, root):
    """
    由模块名得到数据文件列表（默认 data/ 下）：
      EU -> data/eu.jsonl；EA -> data/ea.jsonl；EU+EA/all -> 两者。
    root 为仓库根目录。
    """
    data_dir = Path(root) / "data"
    mapping = {
        "EU": [data_dir / "eu.jsonl"],
        "EA": [data_dir / "ea.jsonl"],
        "EU+EA": [data_dir / "eu.jsonl", data_dir / "ea.jsonl"],
    }
    return mapping.get(module)


def resolve_data_files(module, data_args, root):
    """
    决定最终参与运行的数据文件。
    优先级：显式 --data 参数 > 默认 data/ 文件。
    module 取 EU / EA 时默认只跑对应单文件；EU+EA 时两者都跑。
    """
    if data_args:
        return [Path(p) for p in data_args]
    return module_to_files(module, root)


def _format_choices(choices):
    lines = []
    if not isinstance(choices, dict):
        return lines
    for key in ("A", "B", "C", "D"):
        if key in choices:
            lines.append("%s. %s" % (key, choices[key]))
    return lines


def build_user_message(record):
    """
    由一条记录构造发给模型的用户消息（纯文本），绝不包含 answer 字段。
    """
    parts = []
    if record.get("qid"):
        parts.append("题目编号：%s" % record["qid"])
    scenario = record.get("scenario")
    if scenario:
        parts.append("情境：%s" % scenario)
    subject = record.get("subject")
    if subject:
        parts.append("人物：%s" % subject)
    question = record.get("question")
    if question:
        parts.append("问题：%s" % question)
    choices = record.get("choices")
    if isinstance(choices, dict):
        parts.append("选项：")
        parts.extend("  " + line for line in _format_choices(choices))

    qtype = record.get("question_type", "single_choice")
    parts.append(QTYPE_LABEL.get(qtype, QTYPE_LABEL["single_choice"]))
    return "\n".join(parts)
