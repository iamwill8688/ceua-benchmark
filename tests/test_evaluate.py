# -*- coding: utf-8 -*-
"""
tests/test_evaluate.py
======================

评测逻辑回归测试：单/双选题打分、集合语义、partial score、
以及 gold / prediction 的一致性检查（重复 qid、缺失 qid、未知 qid 等）。

运行：python -m unittest discover -s tests -t .
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402

SINGLE = ev.SINGLE
TWO = ev.TWO


def _gold_item(qid, module, qtype, answer):
    return {"qid": qid, "language": "zh", "module": module,
            "question_type": qtype, "question": "q", "subject": "s",
            "choices": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer": answer}


class _TempJsonl(unittest.TestCase):
    """提供临时 JSONL 文件（写入系统临时目录，绝不触碰仓库数据）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name, records):
        path = self.tmpdir / name
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return str(path)


class TestGoldAndPredictionLoading(_TempJsonl):

    def test_load_gold_ok(self):
        path = self.write("eu.jsonl", [_gold_item("EU_001", "EU", SINGLE, "B"),
                                       _gold_item("EU_002", "EU", SINGLE, "C")])
        gold = ev.load_gold([path])
        self.assertEqual(gold["EU_001"]["gold"], frozenset({"B"}))
        self.assertEqual(gold["EU_001"]["question_type"], SINGLE)

    def test_load_gold_rejects_invalid_answer(self):
        path = self.write("eu.jsonl", [_gold_item("EU_001", "EU", SINGLE, "XY")])
        with self.assertRaises(ValueError):
            ev.load_gold([path])

    def test_load_gold_rejects_duplicate_qid(self):
        path = self.write("eu.jsonl", [_gold_item("EU_001", "EU", SINGLE, "B"),
                                       _gold_item("EU_001", "EU", SINGLE, "C")])
        with self.assertRaises(ValueError):
            ev.load_gold([path])

    def test_load_predictions_ok(self):
        path = self.write("p.jsonl", [{"qid": "EU_001", "prediction": "B"}])
        self.assertEqual(ev.load_predictions(path),
                         [{"qid": "EU_001", "prediction": "B"}])

    def test_duplicate_prediction_raises(self):
        path = self.write("p.jsonl", [{"qid": "EU_001", "prediction": "B"},
                                      {"qid": "EU_001", "prediction": "C"}])
        with self.assertRaises(ValueError):
            ev.load_predictions(path)

    def test_missing_qid_raises(self):
        path = self.write("p.jsonl", [{"prediction": "B"}])
        with self.assertRaises(ValueError):
            ev.load_predictions(path)

    def test_missing_prediction_field_raises(self):
        path = self.write("p.jsonl", [{"qid": "EU_001"}])
        with self.assertRaises(ValueError):
            ev.load_predictions(path)


class TestConsistency(_TempJsonl):

    def setUp(self):
        super().setUp()
        self.gold = {"EU_001": {"module": "EU", "question_type": SINGLE,
                                "gold": frozenset({"B"}), "source": "x"}}

    def test_unknown_qid_raises(self):
        with self.assertRaises(ValueError):
            ev.check_consistency(
                self.gold, [{"qid": "EU_001", "prediction": "B"},
                            {"qid": "EU_999", "prediction": "A"}])

    def test_missing_prediction_raises(self):
        with self.assertRaises(ValueError):
            ev.check_consistency(self.gold, [])

    def test_exact_match_ok(self):
        ev.check_consistency(self.gold, [{"qid": "EU_001", "prediction": "B"}])


class TestScoring(unittest.TestCase):

    def setUp(self):
        self.gold = {
            "EU_001": {"module": "EU", "question_type": SINGLE,
                       "gold": frozenset({"B"}), "source": "t"},
            "EA_001": {"module": "EA", "question_type": SINGLE,
                       "gold": frozenset({"C"}), "source": "t"},
            "EA_014": {"module": "EA", "question_type": TWO,
                       "gold": frozenset({"C", "D"}), "source": "t"},
        }

    def _score(self, predictions):
        return ev.score_all(self.gold, predictions)

    def test_single_choice_exact_match(self):
        per_q = self._score([{"qid": "EU_001", "prediction": "B"},
                             {"qid": "EA_001", "prediction": "A"},
                             {"qid": "EA_014", "prediction": "CD"}])
        self.assertTrue(per_q["EU_001"]["exact_correct"])
        self.assertFalse(per_q["EA_001"]["exact_correct"])
        self.assertTrue(per_q["EA_014"]["exact_correct"])

    def test_two_choice_set_semantics(self):
        """CD == DC（集合语义），顺序不影响判定。"""
        for pred in ("CD", "DC", "C,D", "D C"):
            per_q = self._score([{"qid": "EU_001", "prediction": "B"},
                                 {"qid": "EA_001", "prediction": "C"},
                                 {"qid": "EA_014", "prediction": pred}])
            self.assertTrue(per_q["EA_014"]["exact_correct"],
                            "two_choice 集合语义错误：%r" % pred)

    def test_two_choice_wrong_arity_not_correct(self):
        for pred in ("C", "", "ABC", "ABCD", "CC"):
            per_q = self._score([{"qid": "EU_001", "prediction": "B"},
                                 {"qid": "EA_001", "prediction": "C"},
                                 {"qid": "EA_014", "prediction": pred}])
            self.assertFalse(per_q["EA_014"]["exact_correct"],
                             "错误个数的 two_choice 被判对：%r" % pred)

    def test_partial_score(self):
        """partial = |pred ∩ gold| / |gold|，仅 two_choice。"""
        per_q = self._score([{"qid": "EU_001", "prediction": "B"},
                             {"qid": "EA_001", "prediction": "C"},
                             {"qid": "EA_014", "prediction": "C"}])
        self.assertEqual(per_q["EA_014"]["partial"], 0.5)
        self.assertIsNone(per_q["EU_001"]["partial"])

    def test_invalid_prediction_is_counted_as_zero_not_skipped(self):
        per_q = self._score([{"qid": "EU_001", "prediction": ""},
                             {"qid": "EA_001", "prediction": "C"},
                             {"qid": "EA_014", "prediction": "CD"}])
        metrics = ev.aggregate(per_q)
        self.assertEqual(metrics["total_questions"], 3)   # 未被跳过
        self.assertEqual(metrics["correct"], 2)

    def test_aggregate_metrics(self):
        per_q = self._score([{"qid": "EU_001", "prediction": "B"},
                             {"qid": "EA_001", "prediction": "A"},
                             {"qid": "EA_014", "prediction": "CD"}])
        m = ev.aggregate(per_q)
        self.assertEqual(m["total_questions"], 3)
        self.assertEqual(m["correct"], 2)
        self.assertAlmostEqual(m["eu_accuracy"], 1.0)
        self.assertAlmostEqual(m["ea_accuracy"], 0.5)
        self.assertAlmostEqual(m["single_choice_accuracy"], 0.5)
        self.assertAlmostEqual(m["two_choice_accuracy"], 1.0)
        self.assertAlmostEqual(m["ea_single_choice_accuracy"], 0.0)
        self.assertAlmostEqual(m["ea_two_choice_accuracy"], 1.0)
        self.assertAlmostEqual(m["two_choice_partial_score"], 1.0)
        self.assertAlmostEqual(m["accuracy"], 2 / 3)

    def test_empty_predictions_score_zero(self):
        per_q = self._score([{"qid": q, "prediction": ""} for q in self.gold])
        self.assertEqual(ev.aggregate(per_q)["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()

