# -*- coding: utf-8 -*-
"""
src/main.py
===========

一键运行入口：读取 Benchmark 数据 -> 调用模型作答 -> 保存原始回答与
标准 prediction -> 调用 evaluate.py 打分 -> 输出分数并保存 results/。

用法（在仓库根目录）：
    export DEEPSEEK_API_KEY="YOUR_API_KEY"
    python -m src.main --provider deepseek --model <model-name>

    export OPENAI_API_KEY="YOUR_API_KEY"
    python -m src.main --provider openai --model <model-name>

    # 测试用假模型（不需 API Key，仅用于本地 dry-run）：
    python -m src.main --provider mock --model mock-chat --limit 10

    # OpenRouter（模型 ID 原样传入，含 provider 前缀）：
    export OPENROUTER_API_KEY="YOUR_API_KEY"
    python -m src.main --provider openrouter --model <provider/model> --workers 6

    # 硅基流动（模型 ID 原样传入；国际站可加 --base-url）：
    export SILICONFLOW_API_KEY="YOUR_API_KEY"
    python -m src.main --provider siliconflow --model <org/model> --workers 4

    python -m src.main -h

硬性约束：
  * 只读 data/ 数据，绝不修改 / 覆盖 / 回写 gold answer。
  * API Key 只从环境变量读取，绝不写入代码 / 配置 / 结果 / 日志。
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:  # 优先以包方式导入（python -m src.main）
    from . import data as d
    from . import evaluate as ev
    from . import model as mdl
    from . import utils as u
except ImportError:  # 兼容直接以脚本运行（python src/main.py）
    import data as d
    import evaluate as ev
    import model as mdl
    import utils as u

ROOT = Path(__file__).resolve().parents[1]

ALL_MODULE = "EU+EA"
MODULE_CHOICES = ["EU", "EA", ALL_MODULE, "all"]


def _resolve_module(module):
    return ALL_MODULE if module in ("all", ALL_MODULE) else module

def load_questions(data_files, module):
    """
    读取数据文件，拼出待测题目列表（保留原始记录，但 prompt 不会含 answer）。
    做 qid 去重检查，并按 module 过滤。
    """
    rows = []
    seen = {}
    for f in data_files:
        for rec in d.read_jsonl(f):
            qid = rec.get("qid")
            if qid in seen:
                raise ValueError("gold qid 重复：%s" % qid)
            seen[qid] = rec
            rows.append(rec)

    if module == "EU":
        rows = [r for r in rows if r.get("module") == "EU"]
    elif module == "EA":
        rows = [r for r in rows if r.get("module") == "EA"]
    if not rows:
        raise ValueError("没有读取到任何题目（module=%s）。" % module)
    return rows


def _read_done(raw_path):
    """读取已完成的 qid（来自 raw_responses.jsonl），用于断点续跑。"""
    done = {}
    if not Path(raw_path).is_file():
        return done
    with Path(raw_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = obj.get("qid")
            if qid:
                done[qid] = True
    return done


def _work_one(model, system, record):
    """在（可能并发的）worker 线程中完成单题：返回统一结果字典（含响应诊断信息）。"""
    user = d.build_user_message(record)
    try:
        raw, meta = model.respond_full(system, user)
    except Exception as exc:  # 单题任何失败都不应拖垮整批
        return {"qid": record["qid"], "ok": False, "error": str(exc)}
    prediction, status = u.extract_prediction(raw, record["question_type"])
    return {"qid": record["qid"], "ok": True, "raw": raw,
            "prediction": prediction, "status": status, "meta": meta}


def run_questions(model, questions, predictions_path, raw_path, resume,
                  workers=1, logger=None):
    """
    作答所有待测题目（支持并发），并把结果追加写入两个文件。

    并发实现：
      * 用 ThreadPoolExecutor(workers) 并行调用模型（workers=1 时退回顺序）。
      * 只由主线程负责写文件，天然线程安全；每题写完即 flush，
        即使中途中断也能续跑不丢。
    返回 (processed_new, resumed_count, failures)。
      failures = [(qid, reason)]：多次重试仍失败，不写入文件，下次可续跑。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    predictions_path = Path(predictions_path)
    raw_path = Path(raw_path)

    if resume:
        done = _read_done(raw_path)
    else:
        done = {}
        # 非续跑：清空旧结果，从零开始
        predictions_path.write_text("", encoding="utf-8")
        raw_path.write_text("", encoding="utf-8")

    pending = [q for q in questions if q["qid"] not in done]
    resumed_count = len(questions) - len(pending)

    if not pending:
        return [], resumed_count, []

    workers = max(1, int(workers))
    system = d.SYSTEM_PROMPT
    total = len(pending)
    step = max(1, total // 20)  # 进度提示的粒度
    done_new = 0
    processed = []
    failures = []

    def emit(result):
        """主线程内处理一个结果并写文件（保证线程安全 + 增量保存）。"""
        nonlocal done_new
        done_new += 1
        if not result["ok"]:
            failures.append((result["qid"], result["error"]))
        else:
            meta = result.get("meta") or {}
            # 截断过长 reasoning，避免文件膨胀（如需完整分析可再放开）
            reasoning = meta.get("reasoning_content")
            if isinstance(reasoning, str) and len(reasoning) > 4000:
                reasoning = reasoning[:4000] + "...[truncated]"
            rec = {
                "qid": result["qid"],
                "raw_response": result["raw"],
                "prediction": result["prediction"],
                "parse_status": result["status"],
                "api_model": meta.get("api_model"),
                "finish_reason": meta.get("finish_reason"),
                "native_finish_reason": meta.get("native_finish_reason"),
                "usage": meta.get("usage"),
                "reasoning_tokens": meta.get("reasoning_tokens"),
                "reasoning_content": reasoning,
            }
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rf.flush()
            pf.write(json.dumps({
                "qid": result["qid"], "prediction": result["prediction"],
            }, ensure_ascii=False) + "\n")
            pf.flush()
            processed.append({"qid": result["qid"],
                              "prediction": result["prediction"],
                              "parse_status": result["status"]})
        if logger is not None and (done_new % step == 0 or done_new == total):
            logger.info("Progress: %d / %d", done_new, total)

    # 两个文件都追加写，并每次 flush，保证中途中断也能续跑且不丢记录。
    with predictions_path.open("a", encoding="utf-8") as pf, \
            raw_path.open("a", encoding="utf-8") as rf:
        if workers <= 1 or total <= 1:
            # 顺序（默认，保证行为与测试一致）
            for q in pending:
                emit(_work_one(model, system, q))
        else:
            # 并发
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_work_one, model, system, q)
                           for q in pending]
                for fut in as_completed(futures):
                    emit(fut.result())

    return processed, resumed_count, failures

def evaluate_and_report(data_files, predictions_path):
    """
    调用官方 evaluator（src/evaluate.py）对 predictions 打分并在终端打印报告。

    数据文件保持只读；本函数不会写出任何额外的结果 / 配置 / 状态文件，
    指标只打印到终端（predictions 与 raw responses 已由 run_questions 保存）。
    """
    gold = ev.load_gold([str(p) for p in data_files])
    preds = ev.load_predictions(str(predictions_path))
    ev.check_consistency(gold, preds)
    metrics = ev.aggregate(ev.score_all(gold, preds))
    ev.print_report(metrics, data_files)
    return metrics

def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="CEUA Benchmark (EU / EA) 一键运行：调用模型 -> 保存回答 -> 评测打分。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例（API Key 只从环境变量读取，下面的值只是占位符）：\n"
            "  export DEEPSEEK_API_KEY=\"YOUR_API_KEY\"\n"
            "  python -m src.main --provider deepseek --model <model-name>\n\n"
            "  export OPENAI_API_KEY=\"YOUR_API_KEY\"\n"
            "  python -m src.main --provider openai --model <model-name>\n\n"
            "  export OPENROUTER_API_KEY=\"YOUR_API_KEY\"\n"
            "  python -m src.main --provider openrouter --model <provider/model>\n\n"
            "  export SILICONFLOW_API_KEY=\"YOUR_API_KEY\"\n"
            "  python -m src.main --provider siliconflow --model <org/model>\n\n"
            "  python -m src.main --provider mock --model mock-chat --limit 10\n"
        ),
    )
    parser.add_argument("--provider", required=True, type=str,
                        choices=["openai", "deepseek", "openrouter",
                                 "siliconflow", "mock"],
                        help="模型提供方：openai / deepseek / openrouter / siliconflow / mock(仅测试)。")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称；缺省用 provider 默认模型。")
    parser.add_argument("--base-url", type=str, default=None,
                        help="覆盖 provider 默认端点（OpenAI 兼容 base_url），例如硅基流动国际站 https://api.siliconflow.com/v1。")
    parser.add_argument("--data", nargs="+", default=None, metavar="FILE",
                        help="覆盖默认数据文件（默认 data/eu.jsonl data/ea.jsonl）。")
    parser.add_argument("--module", type=str, default=ALL_MODULE,
                        choices=MODULE_CHOICES,
                        help="运行模块：EU / EA / EU+EA（默认全量）。")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="结果输出目录；默认 results/<model>。")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度（默认 0，利于可复现）。thinking 模式下可能不生效。")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="每次回答的最大 token 数（默认 2048）。reasoning 模型的 reasoning+content 都计入此预算；过小会导致 content 为空（可看 raw 里的 finish_reason=length / reasoning_tokens）。")
    parser.add_argument("--thinking", type=str, default="default",
                        choices=["enabled", "disabled", "default"],
                        help="DeepSeek V4 thinking 模式：enabled/disabled。default=DeepSeek 用 disabled、其它 provider 不发送。openai/openrouter 不注入此参数。")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="DeepSeek V4 可选推理强度（仅 thinking=enabled 时有意义）。")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="关闭模型思考模式（目前仅 siliconflow 生效，发送 enable_thinking=false；其它 provider 忽略）。")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="单题失败最大重试次数（默认 3）。")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="单次 HTTP 超时秒数（默认 60）。")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发请求数（默认 1=顺序）。例如 6 可显著加速，但请留意限流。")
    parser.add_argument("--limit", type=int, default=None,
                        help="只运行前 N 题（开发调试用）。")
    parser.add_argument("--mock-fail-first", type=int, default=0,
                        help="仅 mock 生效：让前 N 次“尝试”抛瞬时错误，用于测试 retry/失败。")
    parser.add_argument("--no-resume", action="store_true",
                        help="关闭断点续跑，重新从头运行并覆盖本目录旧结果。")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    logger = u.get_logger()

    module = _resolve_module(args.module)
    data_files = d.resolve_data_files(module, args.data, ROOT)
    missing = [str(f) for f in data_files if not Path(f).is_file()]
    if missing:
        print("ERROR: 找不到数据文件：%s" % ", ".join(missing), file=sys.stderr)
        print("请确认在仓库根目录运行，或用 --data 指定文件。", file=sys.stderr)
        return 1

    # API Key 检查（mock 不需要）
    provider = args.provider.lower()
    if provider != "mock" and not mdl.resolve_api_key(provider):
        env = mdl.provider_env_var(provider)
        print("Error: %s is not set." % env, file=sys.stderr)
        print("Please set it with:", file=sys.stderr)
        print("\nexport %s=\"YOUR_API_KEY\"\n" % env, file=sys.stderr)
        return 1

    # thinking 默认行为：DeepSeek 用 disabled（保证能拿到最终 content）；
    # 其它 provider 不注入 thinking，避免对不支持的模型硬编码行为。
    if args.thinking == "default":
        thinking = "disabled" if provider == "deepseek" else None
    else:
        thinking = args.thinking
    reasoning_effort = args.reasoning_effort
    if provider not in ("deepseek",) and (
            args.thinking != "default" or args.reasoning_effort):
        logger.warning(
            "--thinking / --reasoning-effort 目前仅在 deepseek provider 层生效，"
            "已对 provider=%s 忽略（不注入请求）。", provider)
        thinking = None
        reasoning_effort = None

    # SiliconFlow/Qwen3 的 enable_thinking：仅 --disable-thinking 时注入 false
    enable_thinking = None
    if args.disable_thinking:
        if provider == "siliconflow":
            enable_thinking = False
        else:
            logger.warning(
                "--disable-thinking 目前仅在 siliconflow provider 生效，"
                "已对 provider=%s 忽略。", provider)

    model = mdl.build_model(provider=provider, model=args.model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                            max_retries=args.max_retries,
                            thinking=thinking,
                            reasoning_effort=reasoning_effort,
                            base_url=args.base_url,
                            enable_thinking=enable_thinking,
                            mock_fail_first=args.mock_fail_first)

    try:
        questions = load_questions(data_files, module)
    except ValueError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1

    if args.limit is not None:
        if args.limit <= 0:
            print("ERROR: --limit 必须为正整数。", file=sys.stderr)
            return 1
        questions = questions[:args.limit]

    model_name = args.model or model.model
    output_dir = args.output_dir or (ROOT / "results" / u.safe_name(model_name))
    u.ensure_dir(output_dir)
    predictions_path = Path(output_dir) / "predictions.jsonl"
    raw_path = Path(output_dir) / "raw_responses.jsonl"

    logger.info("Model=%s Provider=%s Endpoint=%s Questions=%d Output=%s",
                model_name, provider, getattr(model, "base_url", None),
                len(questions), output_dir)

    processed, resumed_count, failures = run_questions(
        model, questions, predictions_path, raw_path,
        resume=not args.no_resume,
        workers=args.workers,
        logger=logger,
    )

    total_completed = resumed_count + len(processed)
    logger.info("Already completed (resumed): %d", resumed_count)
    logger.info("Processed this run: %d", len(processed))
    logger.info("Failed (will be retried on next run): %d", len(failures))
    for qid, reason in failures:
        logger.error("  failed %s: %s", qid, reason)

    full_run = args.limit is None and len(questions) == total_completed \
        and not failures
    if full_run:
        try:
            evaluate_and_report(data_files, predictions_path)
        except ValueError as exc:
            print("ERROR during evaluation: %s" % exc, file=sys.stderr)
            print("预测文件可能不完整，请检查并重跑。", file=sys.stderr)
            return 1
    else:
        print("Completed: %d / %d" % (total_completed, len(questions)))
        print("Failed: %d" % len(failures))
        if args.limit is not None:
            print("Note: --limit 仅用于开发调试，已跳过自动评测。")
        elif failures:
            print("Note: 因存在失败题目，已跳过自动评测；"
                  "修复后可再次运行以自动续跑并打分。")
        print("预测文件：%s" % predictions_path)
        print("原始回答：%s" % raw_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
