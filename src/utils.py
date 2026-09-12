# -*- coding: utf-8 -*-
"""
src/utils.py
============

工具函数：答案规范化、日志、路径安全化。

绝不在此处修改任何数据文件；只处理内存中的模型输出字符串。

答案规范化（single source of truth）
-----------------------------------
normalize_prediction() 是全项目**唯一**的 prediction 规范化实现，
src/main.py（生成 prediction）与 src/evaluate.py（打分）都调用它，
保证「pipeline 产出的 prediction」与「evaluator 认定的 prediction」完全一致。

接受的写法（大小写不敏感、全角字母自动转半角）：
  * 纯选项          : "B" / "b" / "CD" / "DC" / "C,D" / "C D" / "C、D" / "C和D" / "C and D"
  * 带轻度标点      : "B." / "(B)" / "「B」"
  * 显式标记 + 选项 : "答案：C" / "答案是B" / "我的答案是C和D" / "选项 CD"
                      "Answer: C" / "The answer is C" / "I choose D" / "Correct answer is B"
  * 选项在前、解释在后: "B. 因为…" / "CD\n理由：…"

明确拒绝（返回 ""，即 invalid）：
  * 拒答 / 无法回答      : "I cannot answer this question." / "抱歉，我无法回答。"
  * 选项个数不符        : single_choice 给 2 个选项；two_choice 给 1 个或 3+ 个选项
  * 重复选项            : "CC"
  * 含非法字母的串      : "BCD" / "ABCD"
  * 空 / 纯空白 / 非字符串
关键约束：**绝不从普通英文单词中抓取 A/B/C/D**（例如 "Answer"、"Choose"、
"Because"、"Cannot" 里的字母不会被当作选项），也绝不对无法可靠解析的回答做猜测。
"""

import logging
import re
import sys
from pathlib import Path

OPTION_CHARS = "ABCD"
OPTION_SET = frozenset(OPTION_CHARS)

SINGLE = "single_choice"
TWO = "two_choice"

# ---------------------------------------------------------------------------
# 答案规范化
# ---------------------------------------------------------------------------
# 全角 -> 半角（含全角字母 Ａ-Ｄ、全角标点、全角空格）
_FULLWIDTH_TABLE = {i + 0xFEE0: i for i in range(0x21, 0x7F)}
_FULLWIDTH_TABLE[0x3000] = 0x20

# 出现在文本首尾时应当忽略的字符（引号 / 括号 / 句末标点等）
_TRIM_CHARS = " \t\r\n\"'“”‘’()（）[]【】{}<>《》.。,，;；:：、/|"

# 选项之间的分隔符
_SEP = r"(?:\s*(?:[,，、/|;；]|\s|和|与|AND)\s*)"

_SINGLE_RE = re.compile(r"^([ABCD])$")
_TWO_ADJ_RE = re.compile(r"^([ABCD])([ABCD])$")
_TWO_SEP_RE = re.compile(r"^([ABCD])" + _SEP + r"([ABCD])$")
# 选项出现在报文开头（后面跟解释），要求字母边界，避免匹配 "BCAUSE" 这类单词
_LEAD_TWO_RE = re.compile(r"^[ABCD]" + _SEP + r"?[ABCD](?![A-Za-z])")
_LEAD_SINGLE_RE = re.compile(r"^[ABCD](?![A-Za-z])")

# 显式答案标记（长标记在前，避免 "选项" 被 "选" 抢先匹配）。
# 只取**最靠右**的一个标记，且要求标记后面紧跟一个合法选项 token。
_MARKER_RE = re.compile(
    r"(?:正确答案|正确选项|标准答案|我的答案|答案"
    r"|选项|选择|应选|我认为|我觉得|应为|是选|选"
    r"|CORRECT\s+ANSWER|ANSWER\s+IS|ANSWER|CHOOSING|CHOOSE|CHOICE|OPTION|PICK)",
    re.IGNORECASE,
)

# 标记之后允许出现的填充词 / 标点（可重复剥离）。
# 注意：绝不把单个选项字母 A/B/C/D 当作填充词，否则会吃掉正确答案。
_FILLER_PUNCT = r"[\s:：=,，.。;；!！?？\-—*\"'“”‘’()（）\[\]【】{}<>《》]"
_FILLER_RE = re.compile(
    r"^(?:" + _FILLER_PUNCT + r"+"
    r"|是|为|应该|应当|应|选|答|答复|回答|正确|我的|我|的"
    r"|IS|ARE|WAS|WERE|THE|OPTION|OPTIONS|CHOICE|CHOICES|LETTER|ANSWER"
    r"|CORRECT|CHOOSE|CHOOSING|PICK|TO|OF)"
    + _FILLER_PUNCT + r"*",
    re.IGNORECASE,
)


def _clean_text(text):
    """全角转半角 + 大写 + 去掉首尾引号 / 括号 / 句末标点。"""
    return text.translate(_FULLWIDTH_TABLE).upper().strip(_TRIM_CHARS)


def _match_token(text, question_type):
    """
    把一段（已清洗的）文本整体匹配成选项 token，返回规范 prediction；
    不匹配或选项个数不符合题型时返回 ""。

    注意：选项字母只从正则捕获组读取，绝不对整段文本做字母扫描
    （否则 "C AND D" 里的 "AND" 会被误算成选项 A/D）。
    """
    text = text.strip()
    if not text:
        return ""
    m = _SINGLE_RE.match(text)
    if m:
        return m.group(1) if question_type == SINGLE else ""
    for rx in (_TWO_ADJ_RE, _TWO_SEP_RE):
        m = rx.match(text)
        if m:
            first, second = m.group(1), m.group(2)
            if first != second:
                return first + second if question_type == TWO else ""
            return ""
    return ""


def _strip_filler(text):
    """反复剥离标记之后的填充词 / 标点（"是"、":"、"IS"、"THE OPTION" 等）。"""
    s = text
    while True:
        new = _FILLER_RE.sub("", s, count=1)
        if new == s:
            break
        s = new
    return s.strip(_TRIM_CHARS)


def _raw_leading(text):
    """
    取文本开头的「选项表达式」（原始字符串，如 "C" / "CD" / "C AND D"）。

    只接受位于报文开头、且带字母边界的表达式，返回原始子串；
    没有则返回 ""。注意这里只做「截取」，合法性交给 _match_token 判断。
    """
    text = text.strip()
    if not text:
        return ""
    m = _LEAD_TWO_RE.match(text)
    if m:
        return m.group(0).strip()
    m = _LEAD_SINGLE_RE.match(text)
    if m:
        return m.group(0)
    return ""


def normalize_prediction(raw_text, question_type):
    """
    把模型原始回答规范化为标准 prediction（single source of truth）。

    返回字符串：
      * single_choice -> "A" / "B" / "C" / "D"
      * two_choice    -> 两个不同选项字母，顺序保留（如 "CD" / "DC"），
                         交给 evaluator 按集合判等（CD == DC）
      * 无法可靠解析  -> ""（invalid，计 0 分；绝不猜测）

    判定顺序：
      1. 若文本中出现显式答案标记（"答案" / "Answer" / "choose" …），
         从最靠右的标记开始，取标记后紧跟的选项表达式；
         找到就按题型判定，**不再回退**（避免"选项 A 不对，正确答案是 C"这类
         句子被句首的 "A" 抢答）；
      2. 若标记后都取不到选项表达式，或整段没有标记，
         则只接受位于报文开头的独立选项表达式（"B. 因为…" 这种答案在前的格式）；
      3. 其余一律 invalid。

    关键约束：绝不从普通英文单词中抓取 A/B/C/D（"Answer"、"Choose"、
    "Because"、"Cannot" 里的字母不会被当作选项），也绝不对无法可靠解析的回答做猜测。
    """
    if not isinstance(raw_text, str):
        return ""
    s = _clean_text(raw_text)
    if not s:
        return ""

    markers = list(_MARKER_RE.finditer(s))
    if markers:
        for m in reversed(markers):
            raw = _raw_leading(_strip_filler(s[m.end():]))
            if raw:
                return _match_token(raw, question_type)

    raw = _raw_leading(s)
    return _match_token(raw, question_type) if raw else ""


def extract_prediction(raw_text, question_type):
    """
    从模型原始回答中提取标准 prediction（normalize_prediction 的薄封装）。

    返回 (prediction, parse_status)：
      * prediction   -> 规范 prediction 字符串；无法可靠解析时为空串 ""
      * parse_status -> "valid" | "invalid"
    与 src/evaluate.py 使用完全相同的规范化规则，绝不猜测、绝不生成 gold。
    """
    prediction = normalize_prediction(raw_text, question_type)
    if prediction:
        return prediction, "valid"
    return "", "invalid"


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def safe_name(text, fallback="model"):
    """把模型名等转成安全的文件系统名字（用于目录命名）。"""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-._")
    return name if name else fallback


def ensure_dir(path):
    """确保目录存在并返回 Path。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def get_logger(name="ceua-benchmark", level=logging.INFO):
    """控制台日志。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
