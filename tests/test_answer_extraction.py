# -*- coding: utf-8 -*-
"""
tests/test_answer_extraction.py
===============================

答案规范化（src/utils.py::normalize_prediction）的回归测试。

核心保证：
  * 绝不从普通英文单词（Answer / Choose / Because / Cannot …）中抓取 A/B/C/D；
  * 绝不从拒答文本中猜答案；
  * 无法可靠解析 -> invalid（""），不猜测；
  * 支持合理的标准回答格式。

运行：python -m unittest discover -s tests -t .
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import utils as u  # noqa: E402

SINGLE = "single_choice"
TWO = "two_choice"

# (raw model output, expected single_choice prediction, expected two_choice prediction)
# "" 表示应当判为 invalid（计 0 分）。
CASES = [
    # --- 纯选项 ---
    ("B", "B", ""),
    ("b", "B", ""),
    (" B ", "B", ""),
    ("D", "D", ""),
    ("CD", "", "CD"),
    ("DC", "", "DC"),
    ("bd", "", "BD"),
    # --- 带轻度标点 / 括号 / 引号 / 全角 ---
    ("B.", "B", ""),
    ("B)", "B", ""),
    ("(B)", "B", ""),
    ('"B"', "B", ""),
    ("\u201cB\u201d", "B", ""),      # 中文引号 “B”
    ("\uff22", "B", ""),              # 全角 Ｂ
    ("\uff23\uff24", "", "CD"),       # 全角 ＣＤ
    # --- two_choice 分隔符 ---
    ("C,D", "", "CD"),
    ("C, D", "", "CD"),
    ("C D", "", "CD"),
    ("C\u3001D", "", "CD"),
    ("C\u548cD", "", "CD"),
    ("C and D", "", "CD"),
    ("D/C", "", "DC"),
    # --- 中文标记 ---
    ("\u7b54\u6848\uff1aC", "C", ""),                        # 答案：C
    ("\u7b54\u6848\u662fB", "B", ""),                        # 答案是B
    ("\u6211\u7684\u7b54\u6848\u662fC\u548cD", "", "CD"),    # 我的答案是C和D
    ("\u9009\u9879 CD", "", "CD"),                            # 选项 CD
    ("\u6211\u8ba4\u4e3a\u6b63\u786e\u7684\u662f A", "A", ""),  # 我认为正确的是 A
    ("\u9009\u9879 A \u4e0d\u5bf9\uff0c\u6b63\u786e\u7b54\u6848\u662f C", "C", ""),
    # --- 英文标记 ---
    ("Answer: C", "C", ""),
    ("The answer is C", "C", ""),
    ("I choose D", "D", ""),
    ("Correct answer is B", "B", ""),
    ("Because of this, I choose D", "D", ""),
    ("C is the correct answer.", "C", ""),
    ("D is the answer because it helps", "D", ""),
    # --- 选项在前、解释在后 ---
    ("B. \u56e0\u4e3a\u538b\u529b\u5f88\u5927", "B", ""),
    ("CD\n\u7406\u7531\uff1a\u90fd\u5408\u9002", "", "CD"),
    ("A and B are both wrong; the answer is C.", "C", ""),
    # --- 必须判 invalid（不得猜测）---
    ("I cannot answer this question.", "", ""),
    ("\u62b1\u6b49\uff0c\u6211\u65e0\u6cd5\u56de\u7b54\u3002", "", ""),
    ("", "", ""),
    ("   ", "", ""),
    ("\n\n", "", ""),
    ("ABCD", "", ""),
    ("BCD", "", ""),
    ("ABC", "", ""),
    ("CC", "", ""),
    ("BCD and BCA", "", ""),
    ("BCAUSE OF THIS", "", ""),
    ("\u6211\u89c9\u5f97\u90fd\u6709\u53ef\u80fd\uff0c\u4e0d\u786e\u5b9a\u3002", "", ""),
]


class TestNormalizePrediction(unittest.TestCase):
    """逐条校验 normalize_prediction 的行为。"""

    def test_regression_table(self):
        for raw, expect_single, expect_two in CASES:
            with self.subTest(raw=raw, qtype=SINGLE):
                self.assertEqual(
                    u.normalize_prediction(raw, SINGLE), expect_single,
                    "single_choice 规范化结果不符：%r" % raw)
            with self.subTest(raw=raw, qtype=TWO):
                self.assertEqual(
                    u.normalize_prediction(raw, TWO), expect_two,
                    "two_choice 规范化结果不符：%r" % raw)

    def test_no_letter_harvesting_from_english_words(self):
        """英文单词里的 A/B/C/D 绝不能被当作选项。"""
        for raw in ["Answer:", "Because", "Cannot", "Choose all of them",
                    "The correct answer", "Option", "Based on the text"]:
            self.assertEqual(u.normalize_prediction(raw, SINGLE), "",
                             "从单词 %r 中错误地提取了选项" % raw)
            self.assertEqual(u.normalize_prediction(raw, TWO), "",
                             "从单词 %r 中错误地提取了选项" % raw)

    def test_refusal_is_invalid(self):
        for raw in ["I cannot answer this question.",
                    "\u62b1\u6b49\uff0c\u6211\u65e0\u6cd5\u56de\u7b54\u3002",
                    "Sorry, I don't know.",
                    "This question is unclear."]:
            self.assertEqual(u.normalize_prediction(raw, SINGLE), "")
            self.assertEqual(u.normalize_prediction(raw, TWO), "")

    def test_wrong_arity_is_invalid(self):
        self.assertEqual(u.normalize_prediction("C and D", SINGLE), "")
        self.assertEqual(u.normalize_prediction("CD", SINGLE), "")
        self.assertEqual(u.normalize_prediction("C", TWO), "")
        self.assertEqual(u.normalize_prediction("B.", TWO), "")
        self.assertEqual(u.normalize_prediction("ABC", TWO), "")
        self.assertEqual(u.normalize_prediction("ABCD", TWO), "")

    def test_non_string_is_invalid(self):
        for raw in [None, 123, ["B"], {"answer": "B"}]:
            self.assertEqual(u.normalize_prediction(raw, SINGLE), "")
            self.assertEqual(u.normalize_prediction(raw, TWO), "")

    def test_idempotent(self):
        for pred, qt in [("B", SINGLE), ("CD", TWO), ("DC", TWO)]:
            self.assertEqual(u.normalize_prediction(pred, qt), pred)


class TestExtractPrediction(unittest.TestCase):
    """extract_prediction 是 normalize_prediction 的薄封装：返回 (pred, status)。"""

    def test_status_matches_normalization(self):
        for raw, expect_single, _ in CASES:
            pred, status = u.extract_prediction(raw, SINGLE)
            self.assertEqual(pred, expect_single)
            self.assertEqual(status, "valid" if expect_single else "invalid")


class TestEvaluatorAgreesWithPipeline(unittest.TestCase):
    """evaluator 与 pipeline 必须使用完全一致的规范化规则。"""

    def test_exact_canonical_matches_utils(self):
        for raw, _, _ in CASES:
            for qt in (SINGLE, TWO):
                expected = u.normalize_prediction(raw, qt)
                got = ev.exact_canonical(raw, qt)
                self.assertEqual(
                    got, frozenset(expected) if expected else None,
                    "evaluator 与 utils 不一致：%r (%s)" % (raw, qt))

    def test_gold_answers_in_dataset_still_parse(self):
        """真实数据集里的 gold answer 必须仍能被解析（只读，不修改数据）。"""
        data = [ROOT / "data" / "eu.jsonl", ROOT / "data" / "ea.jsonl"]
        gold = ev.load_gold([str(p) for p in data])
        self.assertEqual(len(gold), 412)
        for qid, item in gold.items():
            self.assertTrue(item["gold"], "gold 解析失败：%s" % qid)


if __name__ == "__main__":
    unittest.main()

