from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.logging_config import logging_context, setup_logging
from app.models.schemas import TripRequest
from app.workflows.trip_planner_graph import get_trip_planner_workflow, reset_workflow

RETRIEVAL_NODES = ("search_attractions", "check_weather", "find_hotels")
RAIN_TERMS = (
    "雨",
    "雷阵雨",
    "暴雨",
    "雨夹雪",
    "雪",
    "storm",
    "shower",
    "rain",
)
OUTDOOR_TERMS = (
    "长城",
    "爬山",
    "登山",
    "徒步",
    "古道",
    "山",
    "峰",
    "岭",
    "森林公园",
    "湿地",
    "观景台",
    "露营",
    "漂流",
    "海滨",
    "沙滩",
    "海岛",
    "景区",
    "hiking",
)


@dataclass
class CaseResult:
    case_id: str
    status: str
    error: str
    constraint_passed: bool
    violations: List[str]
    workflow_ms: Optional[int]
    plan_itinerary_ms: Optional[int]
    search_attractions_ms: Optional[int]
    parse_attractions_ms: Optional[int]
    parse_hotels_ms: Optional[int]
    retrieval_total: int
    mcp_hits: int
    fallback_hits: int
    weather_source: str
    run_id: str


NODE_DONE_RE = re.compile(
    r"node_done node=(?P<node>[a-z_]+)(?: source=(?P<source>[a-z_]+))?.*?elapsed_ms=(?P<elapsed>\d+)",
    re.IGNORECASE,
)
PARSE_RE = re.compile(
    r"parse_done type=(?P<kind>attractions|hotels) .*?detail_calls=(?P<detail>\d+) .*?elapsed_ms=(?P<elapsed>\d+)",
    re.IGNORECASE,
)
WORKFLOW_RE = re.compile(r"workflow_done .*?elapsed_ms=(\d+)", re.IGNORECASE)


def _read_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Case file not found: {path}")

    for idx, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL line {idx}: {exc}") from exc
    return cases


def _normalize_constraints(raw: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "min_attractions_per_day": 2,
        "max_attractions_per_day": 3,
        "required_meal_types": ["breakfast", "lunch", "dinner"],
        "avoid_outdoor_on_rain": True,
    }
    merged = dict(defaults)
    merged.update(raw or {})

    merged["min_attractions_per_day"] = int(merged.get("min_attractions_per_day", 2))
    merged["max_attractions_per_day"] = int(merged.get("max_attractions_per_day", 3))
    merged["required_meal_types"] = [str(x).lower() for x in (merged.get("required_meal_types") or [])]
    merged["avoid_outdoor_on_rain"] = bool(merged.get("avoid_outdoor_on_rain", True))
    return merged


def _is_rainy(day_weather: str, night_weather: str) -> bool:
    text = f"{day_weather} {night_weather}".lower()
    return any(term in text for term in RAIN_TERMS)


def _is_outdoor_attraction(attr: Any) -> bool:
    name = getattr(attr, "name", "") or ""
    category = getattr(attr, "category", "") or ""
    desc = getattr(attr, "description", "") or ""
    text = f"{name} {category} {desc}".lower()
    return any(term in text for term in OUTDOOR_TERMS)


def _evaluate_constraints(plan: Any, constraints: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    if plan is None:
        return ["trip_plan is null"]

    min_cnt = constraints["min_attractions_per_day"]
    max_cnt = constraints["max_attractions_per_day"]
    required_meals = constraints["required_meal_types"]
    avoid_outdoor_on_rain = constraints["avoid_outdoor_on_rain"]

    weather_by_date: Dict[str, Any] = {
        getattr(w, "date", ""): w for w in (getattr(plan, "weather_info", []) or []) if getattr(w, "date", "")
    }

    for day in getattr(plan, "days", []) or []:
        day_date = getattr(day, "date", "")
        day_idx = getattr(day, "day_index", "?")
        attractions = getattr(day, "attractions", []) or []
        count = len(attractions)
        if count < min_cnt or count > max_cnt:
            violations.append(
                f"day {day_idx} ({day_date}) attractions={count}, expected in [{min_cnt}, {max_cnt}]"
            )

        meal_types = {
            (getattr(meal, "type", "") or "").lower() for meal in (getattr(day, "meals", []) or [])
        }
        for meal_type in required_meals:
            if meal_type not in meal_types:
                violations.append(f"day {day_idx} ({day_date}) missing meal type: {meal_type}")

        weather = weather_by_date.get(day_date)
        if avoid_outdoor_on_rain and weather is not None:
            if _is_rainy(getattr(weather, "day_weather", ""), getattr(weather, "night_weather", "")):
                bad = [getattr(a, "name", "") for a in attractions if _is_outdoor_attraction(a)]
                if bad:
                    violations.append(
                        f"day {day_idx} ({day_date}) rainy but has outdoor attractions: {', '.join(bad[:3])}"
                    )

    return violations


def _collect_run_metrics(log_path: Path, run_id: str) -> Dict[str, Any]:
    if not log_path.exists():
        return {}

    lines = [
        line
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if f"run={run_id}" in line
    ]

    node_elapsed: Dict[str, int] = {}
    node_source: Dict[str, str] = {}
    parse_elapsed: Dict[str, int] = {}

    workflow_ms: Optional[int] = None
    for line in lines:
        node_match = NODE_DONE_RE.search(line)
        if node_match:
            node = node_match.group("node")
            node_elapsed[node] = int(node_match.group("elapsed"))
            source = node_match.group("source")
            if source:
                node_source[node] = source

        parse_match = PARSE_RE.search(line)
        if parse_match:
            parse_elapsed[parse_match.group("kind")] = int(parse_match.group("elapsed"))

        if workflow_ms is None:
            wf_match = WORKFLOW_RE.search(line)
            if wf_match:
                workflow_ms = int(wf_match.group(1))

    retrieval_total = 0
    mcp_hits = 0
    fallback_hits = 0
    for node in RETRIEVAL_NODES:
        source = node_source.get(node)
        if not source:
            continue
        retrieval_total += 1
        if source == "mcp":
            mcp_hits += 1
        if source == "llm_fallback":
            fallback_hits += 1

    return {
        "workflow_ms": workflow_ms,
        "node_elapsed": node_elapsed,
        "node_source": node_source,
        "parse_elapsed": parse_elapsed,
        "retrieval_total": retrieval_total,
        "mcp_hits": mcp_hits,
        "fallback_hits": fallback_hits,
    }


def _safe_mean(values: List[int]) -> Optional[float]:
    if not values:
        return None
    return round(float(mean(values)), 2)


def _safe_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _summary_cn(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "总用例数": summary.get("total_cases"),
        "成功用例数": summary.get("success_cases"),
        "失败用例数": summary.get("failed_cases"),
        "约束通过用例数": summary.get("constraint_passed_cases"),
        "成功率": _format_percent(float(summary.get("success_rate", 0.0))),
        "失败率": _format_percent(float(summary.get("failure_rate", 0.0))),
        "约束满足率": _format_percent(float(summary.get("constraint_satisfaction_rate", 0.0))),
        "MCP命中率": _format_percent(float(summary.get("mcp_hit_rate", 0.0))),
        "回退率": _format_percent(float(summary.get("fallback_rate", 0.0))),
        "平均总耗时(ms)": summary.get("avg_workflow_ms"),
        "平均行程生成耗时(ms)": summary.get("avg_plan_itinerary_ms"),
        "平均景点检索耗时(ms)": summary.get("avg_search_attractions_ms"),
        "平均景点解析耗时(ms)": summary.get("avg_parse_attractions_ms"),
    }


def _make_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines: List[str] = []
    lines.append("# 评测报告")
    lines.append("")
    lines.append(f"- 生成时间: {report['generated_at']}")
    lines.append(f"- 用例路径: {report['cases_path']}")
    lines.append(f"- 总用例数: {summary['total_cases']}")
    lines.append(f"- 成功率: {_format_percent(summary['success_rate'])}")
    lines.append(f"- 约束满足率: {_format_percent(summary['constraint_satisfaction_rate'])}")
    lines.append(f"- MCP命中率: {_format_percent(summary['mcp_hit_rate'])}")
    lines.append(f"- 回退率: {_format_percent(summary['fallback_rate'])}")
    lines.append(f"- 失败率: {_format_percent(summary['failure_rate'])}")
    lines.append(f"- 平均总耗时(ms): {summary['avg_workflow_ms']}")
    lines.append(f"- 平均行程生成耗时(ms): {summary['avg_plan_itinerary_ms']}")
    lines.append(f"- 平均景点检索耗时(ms): {summary['avg_search_attractions_ms']}")
    lines.append(f"- 平均景点解析耗时(ms): {summary['avg_parse_attractions_ms']}")
    lines.append("")

    if report.get("baseline_comparison"):
        base = report["baseline_comparison"]
        lines.append("## 与基线对比")
        lines.append("")
        for key, value in base.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("## 用例结果")
    lines.append("")
    lines.append("| case_id | 状态 | 约束通过 | 总耗时(ms) | 回退次数 | 违规说明 |")
    lines.append("|---|---|---:|---:|---:|---|")
    for item in report["results"]:
        violations = "; ".join(item["violations"][:2])
        status = item["status"]
        if status == "success":
            status = "成功"
        elif status == "runtime_error":
            status = "运行失败"
        elif status == "input_error":
            status = "输入失败"
        lines.append(
            f"| {item['case_id']} | {status} | {item['constraint_passed']} | "
            f"{item.get('workflow_ms') or '-'} | {item['fallback_hits']} | {violations or '-'} |"
        )
    lines.append("")

    lines.append("## 门禁结果")
    gate = report["gate"]
    lines.append("")
    lines.append(f"- 是否通过: {gate['passed']}")
    if gate["reasons"]:
        for reason in gate["reasons"]:
            lines.append(f"- 原因: {reason}")
    return "\n".join(lines)


def _compare_baseline(current_summary: Dict[str, Any], baseline_path: Path) -> Dict[str, str]:
    if not baseline_path.exists():
        return {"基线报告": f"缺失: {baseline_path}"}

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_summary = baseline.get("summary", {})

    keys = [
        "constraint_satisfaction_rate",
        "mcp_hit_rate",
        "fallback_rate",
        "failure_rate",
        "avg_workflow_ms",
    ]
    key_label = {
        "constraint_satisfaction_rate": "约束满足率",
        "mcp_hit_rate": "MCP命中率",
        "fallback_rate": "回退率",
        "failure_rate": "失败率",
        "avg_workflow_ms": "平均总耗时(ms)",
    }
    out: Dict[str, str] = {}
    for key in keys:
        cur = current_summary.get(key)
        old = baseline_summary.get(key)
        if cur is None or old is None:
            continue
        label = key_label.get(key, key)
        if isinstance(cur, float) and isinstance(old, float):
            delta = cur - old
            out[label] = f"{cur:.4f} (基线 {old:.4f}, 变化 {delta:+.4f})"
        else:
            try:
                c = float(cur)
                o = float(old)
                out[label] = f"{c:.2f} (基线 {o:.2f}, 变化 {c - o:+.2f})"
            except Exception:
                out[label] = f"{cur} (基线 {old})"
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline-style evaluator for trip planner workflow")
    parser.add_argument("--cases", default="evals/eval_cases.jsonl", help="Path to JSONL cases")
    parser.add_argument("--output", default="", help="Output report JSON path")
    parser.add_argument("--baseline", default="", help="Optional baseline report JSON path")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N cases")
    parser.add_argument(
        "--gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable threshold gate (default: disabled, same as --no-gate)",
    )
    parser.add_argument("--reset-workflow-each-case", action="store_true", help="Recreate workflow per case")

    parser.add_argument("--min-constraint-pass-rate", type=float, default=0.95)
    parser.add_argument("--max-fallback-rate", type=float, default=0.20)
    parser.add_argument("--max-failure-rate", type=float, default=0.05)
    parser.add_argument("--max-avg-latency-ms", type=float, default=0.0)

    args = parser.parse_args()

    setup_logging("INFO")

    cases_path = Path(args.cases)
    if not cases_path.is_absolute():
        cases_path = (BACKEND_DIR / cases_path).resolve()

    report_path = Path(args.output) if args.output else Path(
        f"evals/reports/eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    if not report_path.is_absolute():
        report_path = (BACKEND_DIR / report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(args.baseline) if args.baseline else None
    if baseline_path and not baseline_path.is_absolute():
        baseline_path = (BACKEND_DIR / baseline_path).resolve()

    log_path = (BACKEND_DIR / "logs" / "backend.out.log").resolve()

    cases = _read_cases(cases_path)
    if args.limit > 0:
        cases = cases[: args.limit]

    if not cases:
        raise ValueError("No cases found to evaluate.")

    results: List[CaseResult] = []

    workflow = None
    if not args.reset_workflow_each_case:
        reset_workflow()
        workflow = get_trip_planner_workflow()

    for idx, case in enumerate(cases, 1):
        case_id = str(case.get("id") or f"case_{idx:04d}")
        run_id = f"eval-{case_id}-{idx}-{int(time.time() * 1000)}"
        print(f"[{idx}/{len(cases)}] running {case_id} ...")

        constraints = _normalize_constraints(case.get("constraints") or {})

        if args.reset_workflow_each_case:
            reset_workflow()
            workflow = get_trip_planner_workflow()

        try:
            request = TripRequest(**(case.get("input") or {}))
        except Exception as exc:
            results.append(
                CaseResult(
                    case_id=case_id,
                    status="input_error",
                    error=str(exc),
                    constraint_passed=False,
                    violations=[f"invalid input: {exc}"],
                    workflow_ms=None,
                    plan_itinerary_ms=None,
                    search_attractions_ms=None,
                    parse_attractions_ms=None,
                    parse_hotels_ms=None,
                    retrieval_total=0,
                    mcp_hits=0,
                    fallback_hits=0,
                    weather_source="",
                    run_id=run_id,
                )
            )
            continue

        plan = None
        error_msg = ""
        status = "success"

        try:
            started = time.perf_counter()
            with logging_context(request_id=run_id, run_id=run_id):
                plan = workflow.plan_trip(request)
            _ = int((time.perf_counter() - started) * 1000)
        except Exception as exc:
            status = "runtime_error"
            error_msg = str(exc)

        metrics = _collect_run_metrics(log_path, run_id)
        violations = _evaluate_constraints(plan, constraints) if status == "success" else [f"runtime error: {error_msg}"]

        node_elapsed = metrics.get("node_elapsed", {})
        parse_elapsed = metrics.get("parse_elapsed", {})
        node_source = metrics.get("node_source", {})

        results.append(
            CaseResult(
                case_id=case_id,
                status=status,
                error=error_msg,
                constraint_passed=(len(violations) == 0),
                violations=violations,
                workflow_ms=metrics.get("workflow_ms"),
                plan_itinerary_ms=node_elapsed.get("plan_itinerary"),
                search_attractions_ms=node_elapsed.get("search_attractions"),
                parse_attractions_ms=parse_elapsed.get("attractions"),
                parse_hotels_ms=parse_elapsed.get("hotels"),
                retrieval_total=int(metrics.get("retrieval_total", 0)),
                mcp_hits=int(metrics.get("mcp_hits", 0)),
                fallback_hits=int(metrics.get("fallback_hits", 0)),
                weather_source=str(node_source.get("check_weather", "")),
                run_id=run_id,
            )
        )

    total_cases = len(results)
    success_cases = sum(1 for r in results if r.status == "success")
    failed_cases = total_cases - success_cases
    constraint_passed_cases = sum(1 for r in results if r.constraint_passed)

    retrieval_total = sum(r.retrieval_total for r in results)
    mcp_hits = sum(r.mcp_hits for r in results)
    fallback_hits = sum(r.fallback_hits for r in results)

    workflow_latencies = [r.workflow_ms for r in results if isinstance(r.workflow_ms, int)]

    summary = {
        "total_cases": total_cases,
        "success_cases": success_cases,
        "failed_cases": failed_cases,
        "constraint_passed_cases": constraint_passed_cases,
        "success_rate": _safe_pct(success_cases, total_cases),
        "failure_rate": _safe_pct(failed_cases, total_cases),
        "constraint_satisfaction_rate": _safe_pct(constraint_passed_cases, total_cases),
        "mcp_hit_rate": _safe_pct(mcp_hits, retrieval_total),
        "fallback_rate": _safe_pct(fallback_hits, retrieval_total),
        "avg_workflow_ms": _safe_mean([int(x) for x in workflow_latencies]),
        "avg_plan_itinerary_ms": _safe_mean(
            [int(r.plan_itinerary_ms) for r in results if isinstance(r.plan_itinerary_ms, int)]
        ),
        "avg_search_attractions_ms": _safe_mean(
            [int(r.search_attractions_ms) for r in results if isinstance(r.search_attractions_ms, int)]
        ),
        "avg_parse_attractions_ms": _safe_mean(
            [int(r.parse_attractions_ms) for r in results if isinstance(r.parse_attractions_ms, int)]
        ),
    }

    gate_reasons: List[str] = []
    if summary["constraint_satisfaction_rate"] < args.min_constraint_pass_rate:
        gate_reasons.append(
            f"约束满足率 {summary['constraint_satisfaction_rate']:.3f} < 阈值 {args.min_constraint_pass_rate:.3f}"
        )
    if summary["fallback_rate"] > args.max_fallback_rate:
        gate_reasons.append(f"回退率 {summary['fallback_rate']:.3f} > 阈值 {args.max_fallback_rate:.3f}")
    if summary["failure_rate"] > args.max_failure_rate:
        gate_reasons.append(f"失败率 {summary['failure_rate']:.3f} > 阈值 {args.max_failure_rate:.3f}")
    if args.max_avg_latency_ms > 0 and summary["avg_workflow_ms"] is not None:
        if float(summary["avg_workflow_ms"]) > float(args.max_avg_latency_ms):
            gate_reasons.append(
                f"平均总耗时(ms) {summary['avg_workflow_ms']} > 阈值 {float(args.max_avg_latency_ms):.2f}"
            )

    gate = {
        "enabled": bool(args.gate),
        "passed": len(gate_reasons) == 0,
        "reasons": gate_reasons,
        "thresholds": {
            "min_constraint_pass_rate": args.min_constraint_pass_rate,
            "max_fallback_rate": args.max_fallback_rate,
            "max_failure_rate": args.max_failure_rate,
            "max_avg_latency_ms": args.max_avg_latency_ms,
        },
    }

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases_path": str(cases_path),
        "summary": summary,
        "summary_cn": _summary_cn(summary),
        "gate": gate,
        "results": [asdict(r) for r in results],
    }

    if baseline_path:
        report["baseline_comparison"] = _compare_baseline(summary, baseline_path)

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = report_path.with_suffix(".md")
    md_path.write_text(_make_markdown(report), encoding="utf-8")

    print("\n=== 评测汇总（中文）===")
    print(json.dumps(report["summary_cn"], ensure_ascii=False, indent=2))
    print(f"\n报告 JSON: {report_path}")
    print(f"报告 MD  : {md_path}")

    if args.gate and gate_reasons:
        print("\n门禁未通过：")
        for reason in gate_reasons:
            print(f"- {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
