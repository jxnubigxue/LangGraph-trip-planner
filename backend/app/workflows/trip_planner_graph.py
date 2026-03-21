"""Trip planner workflow: decompose + parallel retrieval + quality check."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from ..models.schemas import (
    Attraction,
    Budget,
    DayPlan,
    Hotel,
    Location,
    Meal,
    TripPlan,
    TripRequest,
    WeatherInfo,
)
from ..services.llm_service import get_llm
from ..tools.amap_mcp_tools import get_cached_amap_tools
from .trip_planner_state import TripPlannerState, create_initial_state

logger = logging.getLogger(__name__)

CITY_ADCODE_MAP: Dict[str, str] = {
    "北京": "110000",
    "北京市": "110000",
    "上海": "310000",
    "上海市": "310000",
    "广州": "440100",
    "广州市": "440100",
    "深圳": "440300",
    "深圳市": "440300",
    "杭州": "330100",
    "杭州市": "330100",
    "南京": "320100",
    "南京市": "320100",
    "苏州": "320500",
    "苏州市": "320500",
    "成都": "510100",
    "成都市": "510100",
    "重庆": "500000",
    "重庆市": "500000",
    "武汉": "420100",
    "武汉市": "420100",
    "西安": "610100",
    "西安市": "610100",
    "天津": "120000",
    "天津市": "120000",
    "长沙": "430100",
    "长沙市": "430100",
    "青岛": "370200",
    "青岛市": "370200",
    "厦门": "350200",
    "厦门市": "350200",
    "福州": "350100",
    "福州市": "350100",
}


class TripPlannerWorkflow:
    """Trip planning workflow with MCP-first data retrieval."""

    def __init__(self):
        logger.info("Initializing LangGraph trip planner workflow...")
        self.tools = get_cached_amap_tools()
        self.llm = get_llm()
        self._tool_lookup_cache: Dict[str, Any] = {}
        self._tool_input_keys_cache: Dict[str, set[str]] = {}
        self._city_adcode_cache: Dict[str, str] = dict(CITY_ADCODE_MAP)
        self._poi_detail_cache: Dict[str, Dict[str, Any]] = {}
        self._poi_detail_cache_lock = threading.Lock()
        workers_raw = os.getenv("DETAIL_ENRICH_WORKERS", "3").strip()
        try:
            workers = int(workers_raw)
        except ValueError:
            workers = 3
        self._detail_enrich_workers = max(1, min(workers, 6))

        if self.tools:
            logger.info("Loaded %d MCP tools", len(self.tools))
        else:
            logger.warning("No MCP tools loaded, workflow will use LLM fallback only")

        self.graph = self._build_graph()
        logger.info("LangGraph workflow initialized")

    # ---------------------------------------------------------------------
    # Graph
    # ---------------------------------------------------------------------
    def _build_graph(self) -> StateGraph:
        """decompose -> parallel retrieval -> quality -> plan -> abnormal check."""
        workflow = StateGraph(TripPlannerState)

        workflow.add_node("decompose_request", self._decompose_request)
        workflow.add_node("search_attractions", self._search_attractions)
        workflow.add_node("check_weather", self._check_weather)
        workflow.add_node("find_hotels", self._find_hotels)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("plan_itinerary", self._plan_itinerary)
        workflow.add_node("abnormal_check", self._check_abnormal_conditions)
        workflow.add_node("handle_error", self._handle_error)

        workflow.set_entry_point("decompose_request")

        # fan-out
        workflow.add_edge("decompose_request", "search_attractions")
        workflow.add_edge("decompose_request", "check_weather")
        workflow.add_edge("decompose_request", "find_hotels")

        # fan-in barrier: wait until all three retrieval nodes finish.
        workflow.add_edge(
            ["search_attractions", "check_weather", "find_hotels"],
            "quality_check",
        )

        workflow.add_conditional_edges(
            "quality_check",
            self._route_after_quality_check,
            {"continue": "plan_itinerary", "error": "handle_error"},
        )

        workflow.add_edge("plan_itinerary", "abnormal_check")
        workflow.add_edge("abnormal_check", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    # ---------------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------------
    def _decompose_request(self, state: TripPlannerState) -> Dict[str, Any]:
        try:
            request = state["request"]
            breakdown = self._build_task_breakdown(request)
            return {
                "task_breakdown": breakdown,
                "current_step": "tasks_decomposed",
                "messages": [{"role": "assistant", "content": "任务拆分完成，开始并行取数。"}],
            }
        except Exception as exc:
            logger.error("Task decomposition failed: %s", exc, exc_info=True)
            return {"error": f"任务拆分失败: {exc}", "current_step": "decompose_failed"}

    def _search_attractions(self, state: TripPlannerState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        started = time.perf_counter()

        request = state["request"]
        breakdown = state.get("task_breakdown") or {}
        attraction_task = breakdown.get("attraction", {}) if isinstance(breakdown, dict) else {}
        keyword = self._safe_str(attraction_task.get("keywords"), "热门景点")
        target_count = min(30, max(3, self._safe_int(getattr(request, "travel_days", 1), 1) * 3))

        # MCP first
        source = "mcp"
        attractions: List[Attraction] = []
        raw = self._invoke_maps_text_search(request.city, keyword)
        if raw:
            attractions = self._parse_attractions(raw)
            non_food = [a for a in attractions if not self._is_food_poi(a)]
            if non_food:
                attractions = non_food

            attractions = self._merge_unique_attractions(
                [a for a in attractions if self._is_valid_location(a.location)],
                max_items=30,
            )

            if len(attractions) < target_count:
                preference_text = " ".join(
                    [self._safe_str(p, "") for p in (getattr(request, "preferences", None) or [])]
                )
                probe_keywords: List[str] = [
                    "热门景点",
                    "城市地标",
                    "历史文化景点",
                    "博物馆",
                    "风景名胜",
                    "城市公园",
                ]
                if "历史" in preference_text or "文化" in preference_text:
                    probe_keywords.insert(0, "历史古迹")
                if "自然" in preference_text:
                    probe_keywords.insert(0, "自然景观")

                seen_probe: set[str] = set()
                boost_rounds = 0
                before_boost = len(attractions)
                for probe in probe_keywords:
                    if len(attractions) >= target_count:
                        break
                    probe_key = self._normalize_name(probe)
                    if not probe_key or probe_key in seen_probe:
                        continue
                    seen_probe.add(probe_key)
                    if probe_key == self._normalize_name(keyword):
                        continue

                    probe_raw = self._invoke_maps_text_search(request.city, probe)
                    if not probe_raw:
                        continue
                    probe_items = self._parse_attractions(probe_raw)
                    probe_items = [
                        a
                        for a in probe_items
                        if self._is_valid_location(a.location) and not self._is_food_poi(a)
                    ]
                    if not probe_items:
                        continue
                    boost_rounds += 1
                    attractions = self._merge_unique_attractions(attractions + probe_items, max_items=30)

                if boost_rounds > 0:
                    logger.info(
                        "attraction_recall_boost city=%s before=%d after=%d target=%d rounds=%d",
                        request.city,
                        before_boost,
                        len(attractions),
                        target_count,
                        boost_rounds,
                    )

        # fallback: self-generate via LLM
        if not attractions:
            source = "llm_fallback"
            try:
                prompt = self._build_attraction_query(request, keyword)
                output = self._invoke_llm_text(prompt)
                attractions = self._parse_attractions(output)
            except Exception as exc:
                logger.error("Attraction fallback failed: %s", exc, exc_info=True)
                attractions = []

        # second-pass MCP fallback: avoid all-food attractions.
        if attractions and all(self._is_food_poi(a) for a in attractions):
            alt_raw = self._invoke_maps_text_search(request.city, "热门景点")
            if alt_raw:
                alt = [a for a in self._parse_attractions(alt_raw) if not self._is_food_poi(a)]
                if alt:
                    attractions = self._merge_unique_attractions(alt + attractions)

        # Keep MCP recall high; strict city filtering is done later by source matching in planner parse.
        if source != "mcp":
            in_city = [a for a in attractions if self._is_attraction_in_city(a, request.city)]
            if in_city:
                attractions = in_city

        attractions = self._merge_unique_attractions(
            [a for a in attractions if self._is_valid_location(a.location)],
            max_items=30,
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "node_done node=search_attractions source=%s count=%d target=%d elapsed_ms=%d",
            source,
            len(attractions),
            target_count,
            elapsed_ms,
        )
        return {
            "attractions": attractions,
            "current_step": "attractions_done",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"景点节点完成 source={source} count={len(attractions)}",
                }
            ],
        }

    def _check_weather(self, state: TripPlannerState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        started = time.perf_counter()

        request = state["request"]
        breakdown = state.get("task_breakdown") or {}
        weather_task = breakdown.get("weather", {}) if isinstance(breakdown, dict) else {}
        task_city = self._safe_str(weather_task.get("city"), "")
        city = task_city or self._safe_str(request.city, "")
        if not self._resolve_city_adcode(city):
            req_city = self._safe_str(request.city, "")
            if req_city and req_city != city and self._resolve_city_adcode(req_city):
                city = req_city

        source = "mcp"
        weather_info: List[WeatherInfo] = []
        raw = self._invoke_maps_weather(city)
        if raw:
            weather_info = self._parse_weather(raw)

        if not weather_info:
            if raw:
                logger.warning(
                    "Weather MCP parse produced no records: city=%s raw_preview=%s",
                    city,
                    self._safe_str(raw, "")[:240],
                )
            else:
                logger.warning("Weather MCP returned empty result: city=%s", city)
            source = "llm_fallback"
            try:
                prompt = (
                    f"请返回{city}未来3到5天的天气 JSON 数组。"
                    "字段必须有: date,day_weather,night_weather,day_temp,night_temp,wind_direction,wind_power。"
                    "只返回 JSON。"
                )
                output = self._invoke_llm_text(prompt)
                weather_info = self._parse_weather(output)
            except Exception as exc:
                logger.error("Weather fallback failed: %s", exc, exc_info=True)
                weather_info = []

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "node_done node=check_weather source=%s count=%d elapsed_ms=%d city=%s",
            source,
            len(weather_info),
            elapsed_ms,
            city,
        )
        return {
            "weather_info": weather_info,
            "current_step": "weather_done",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"天气节点完成 source={source} count={len(weather_info)}",
                }
            ],
        }

    def _find_hotels(self, state: TripPlannerState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        started = time.perf_counter()

        request = state["request"]
        breakdown = state.get("task_breakdown") or {}
        hotel_task = breakdown.get("hotel", {}) if isinstance(breakdown, dict) else {}
        keyword = self._safe_str(hotel_task.get("keywords"), f"{request.accommodation} 酒店")

        source = "mcp"
        hotels: List[Hotel] = []
        raw = self._invoke_maps_text_search(request.city, keyword)
        if raw:
            hotels = self._parse_hotels(raw)

        if not hotels:
            source = "llm_fallback"
            try:
                prompt = (
                    f"请为{request.city}推荐{request.accommodation}酒店，返回 JSON 数组。"
                    "字段: name,address,location{longitude,latitude},price_range,rating,distance,type,estimated_cost。"
                    "只返回 JSON。"
                )
                output = self._invoke_llm_text(prompt)
                hotels = self._parse_hotels(output)
            except Exception as exc:
                logger.error("Hotel fallback failed: %s", exc, exc_info=True)
                hotels = []

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "node_done node=find_hotels source=%s count=%d elapsed_ms=%d keyword=%s",
            source,
            len(hotels),
            elapsed_ms,
            keyword,
        )
        return {
            "hotels": hotels,
            "current_step": "hotels_done",
            "messages": [
                {
                    "role": "assistant",
                    "content": f"酒店节点完成 source={source} count={len(hotels)}",
                }
            ],
        }

    def _quality_check(self, state: TripPlannerState) -> Dict[str, Any]:
        started = time.perf_counter()
        attractions_count = len(state.get("attractions", []))
        weather_count = len(state.get("weather_info", []))
        hotels_count = len(state.get("hotels", []))

        logger.info(
            "Quality check: attractions=%d weather=%d hotels=%d",
            attractions_count,
            weather_count,
            hotels_count,
        )

        if attractions_count == 0 and weather_count == 0 and hotels_count == 0:
            return {
                "error": "三类数据均为空，检查节点未通过",
                "current_step": "quality_failed",
                "messages": [{"role": "assistant", "content": "检查失败：景点/天气/酒店全部为空。"}],
            }

        warns: List[str] = []
        if attractions_count == 0:
            warns.append("景点为空")
        if weather_count == 0:
            warns.append("天气为空")
        if hotels_count == 0:
            warns.append("酒店为空")

        msg = "检查通过" + (f"，但存在缺失: {', '.join(warns)}" if warns else "")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info("node_done node=quality_check elapsed_ms=%d warns=%d", elapsed_ms, len(warns))
        return {
            "current_step": "quality_passed",
            "messages": [{"role": "assistant", "content": msg}],
        }

    def _route_after_quality_check(self, state: TripPlannerState) -> str:
        return "error" if state.get("error") else "continue"

    def _plan_itinerary(self, state: TripPlannerState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        started = time.perf_counter()
        try:
            query = self._build_planner_query(
                state["request"],
                state.get("attractions", []),
                state.get("weather_info", []),
                state.get("hotels", []),
            )
            output = self._invoke_llm_text(query)
            trip_plan = self._parse_trip_plan(
                output,
                state["request"],
                source_attractions=state.get("attractions", []),
                source_hotels=state.get("hotels", []),
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info("node_done node=plan_itinerary status=success elapsed_ms=%d", elapsed_ms)
            return {
                "trip_plan": trip_plan,
                "current_step": "plan_completed",
                "messages": [{"role": "assistant", "content": "行程规划完成"}],
            }
        except Exception as exc:
            logger.error("Plan itinerary failed: %s", exc, exc_info=True)
            fallback = self._create_fallback_plan(state["request"])
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.warning("node_done node=plan_itinerary status=fallback elapsed_ms=%d", elapsed_ms)
            return {
                "trip_plan": fallback,
                "current_step": "plan_fallback",
                "messages": [{"role": "assistant", "content": f"规划失败，返回备用行程: {exc}"}],
            }

    def _check_abnormal_conditions(self, state: TripPlannerState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        started = time.perf_counter()

        trip_plan = state.get("trip_plan")
        if not isinstance(trip_plan, TripPlan):
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info("node_done node=abnormal_check status=skipped elapsed_ms=%d", elapsed_ms)
            return {
                "abnormal_alerts": [],
                "current_step": "abnormal_checked",
                "messages": [{"role": "assistant", "content": "异常检查跳过：暂无行程数据。"}],
            }

        request = state["request"]
        source_attractions = state.get("attractions", [])
        fallback_weather = state.get("weather_info", [])

        initial_alerts = self._collect_weather_activity_alerts(trip_plan, fallback_weather)
        if not initial_alerts:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info("node_done node=abnormal_check status=pass elapsed_ms=%d", elapsed_ms)
            return {
                "abnormal_alerts": [],
                "current_step": "abnormal_checked",
                "messages": [{"role": "assistant", "content": "异常检查通过，未发现天气与活动冲突。"}],
            }

        logger.warning(
            "Abnormal check found %d conflict(s) before repair: %s",
            len(initial_alerts),
            " | ".join(initial_alerts),
        )
        repaired_plan, repaired_count = self._auto_repair_weather_activity_conflicts(
            trip_plan,
            request,
            source_attractions,
            fallback_weather,
        )
        remaining_alerts = self._collect_weather_activity_alerts(repaired_plan, fallback_weather)

        merged_suggestions = repaired_plan.overall_suggestions
        if repaired_count > 0:
            merged_suggestions = self._merge_repair_note_into_suggestions(merged_suggestions, repaired_count)
        if remaining_alerts:
            merged_suggestions = self._merge_safety_alerts_into_suggestions(merged_suggestions, remaining_alerts)

        try:
            updated_plan = repaired_plan.model_copy(update={"overall_suggestions": merged_suggestions})
        except Exception:
            repaired_plan.overall_suggestions = merged_suggestions
            updated_plan = repaired_plan

        if remaining_alerts:
            if repaired_count > 0:
                msg = f"异常检查发现 {len(initial_alerts)} 项冲突，已自动修复 {repaired_count} 项，剩余 {len(remaining_alerts)} 项已写入建议。"
            else:
                msg = f"异常检查发现 {len(remaining_alerts)} 项天气冲突，已写入行程建议。"
        else:
            msg = f"异常检查发现 {len(initial_alerts)} 项冲突，已自动修复并通过复检。"

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "node_done node=abnormal_check status=checked initial_alerts=%d repaired=%d remaining=%d elapsed_ms=%d",
            len(initial_alerts),
            repaired_count,
            len(remaining_alerts),
            elapsed_ms,
        )
        return {
            "current_step": "abnormal_checked",
            "trip_plan": updated_plan,
            "abnormal_alerts": remaining_alerts,
            "messages": [
                {
                    "role": "assistant",
                    "content": msg,
                }
            ],
        }

    def _auto_repair_weather_activity_conflicts(
        self,
        trip_plan: TripPlan,
        request: TripRequest,
        source_attractions: List[Attraction],
        fallback_weather: List[WeatherInfo],
    ) -> Tuple[TripPlan, int]:
        weather_by_date = self._build_weather_text_by_date(trip_plan.weather_info or fallback_weather)
        candidate_pool = self._build_rain_safe_candidate_pool(request.city, source_attractions, trip_plan)
        if not candidate_pool:
            return trip_plan, 0

        replaced_total = 0
        updated_days: List[DayPlan] = []

        for day in trip_plan.days:
            weather_text = self._safe_str(weather_by_date.get(day.date), "")
            if not weather_text or not self._is_rainy_weather_text(weather_text):
                updated_days.append(day)
                continue

            repaired_attractions, replaced_count = self._replace_day_risky_attractions(day.attractions, candidate_pool)
            if replaced_count <= 0:
                updated_days.append(day)
                continue

            replaced_total += replaced_count
            desc_suffix = f"（考虑{weather_text}，已优先安排室内或低暴露景点）"
            day_desc = self._safe_str(day.description, "")
            if desc_suffix not in day_desc:
                day_desc = f"{day_desc}{desc_suffix}" if day_desc else desc_suffix.strip("（）")

            hotel = day.hotel
            if hotel is not None:
                hotel = self._complete_hotel_fields(hotel, request, repaired_attractions)

            try:
                updated_day = day.model_copy(
                    update={
                        "description": day_desc,
                        "attractions": repaired_attractions,
                        "hotel": hotel,
                    }
                )
            except Exception:
                updated_day = DayPlan(
                    date=day.date,
                    day_index=day.day_index,
                    description=day_desc,
                    transportation=day.transportation,
                    accommodation=day.accommodation,
                    hotel=hotel,
                    attractions=repaired_attractions,
                    meals=list(day.meals),
                )
            updated_days.append(updated_day)

        if replaced_total <= 0:
            return trip_plan, 0

        try:
            updated_plan = trip_plan.model_copy(update={"days": updated_days})
        except Exception:
            trip_plan.days = updated_days
            updated_plan = trip_plan
        return updated_plan, replaced_total

    def _build_rain_safe_candidate_pool(
        self,
        city: str,
        source_attractions: List[Attraction],
        trip_plan: TripPlan,
    ) -> List[Attraction]:
        seed: List[Attraction] = []
        for item in source_attractions:
            if isinstance(item, Attraction):
                seed.append(item)
        for day in trip_plan.days:
            for item in day.attractions:
                if isinstance(item, Attraction):
                    seed.append(item)

        indoor_candidates = self._merge_unique_attractions(
            [
                a
                for a in seed
                if self._is_valid_location(a.location)
                and not self._is_food_poi(a)
                and self._is_indoor_candidate_attraction(a)
                and self._is_attraction_in_city(a, city)
            ],
            max_items=80,
        )
        if indoor_candidates:
            return indoor_candidates

        # No usable local candidates, then query MCP for indoor alternatives.
        for keyword in ["博物馆", "美术馆"]:
            raw = self._invoke_maps_text_search(city, keyword)
            if not raw:
                continue
            parsed = self._parse_attractions(raw)
            if parsed:
                seed.extend(parsed)

        indoor_candidates = self._merge_unique_attractions(
            [
                a
                for a in seed
                if self._is_valid_location(a.location)
                and not self._is_food_poi(a)
                and self._is_indoor_candidate_attraction(a)
                and self._is_attraction_in_city(a, city)
            ],
            max_items=80,
        )
        if indoor_candidates:
            return indoor_candidates

        # fallback: at least choose non-food + non-high-risk attractions
        safe_candidates = self._merge_unique_attractions(
            [
                a
                for a in seed
                if self._is_valid_location(a.location)
                and not self._is_food_poi(a)
                and not self._is_outdoor_high_exposure_attraction(a)
            ],
            max_items=80,
        )
        return safe_candidates

    def _replace_day_risky_attractions(
        self,
        day_attractions: List[Attraction],
        candidate_pool: List[Attraction],
    ) -> Tuple[List[Attraction], int]:
        current = [a for a in day_attractions if isinstance(a, Attraction)]
        if not current:
            return [], 0

        risky = [a for a in current if self._is_outdoor_high_exposure_attraction(a)]
        if not risky:
            return current, 0

        safe = [a for a in current if not self._is_outdoor_high_exposure_attraction(a)]
        target_count = len(current)
        if target_count <= 0:
            target_count = 2
        target_count = max(1, min(3, target_count))

        result = self._merge_unique_attractions(safe, max_items=3)
        used_keys = {self._normalize_name(a.name) for a in result}

        for candidate in candidate_pool:
            key = self._normalize_name(candidate.name)
            if not key or key in used_keys:
                continue
            result.append(candidate)
            used_keys.add(key)
            if len(result) >= target_count:
                break

        # keep non-empty output even if candidate pool is insufficient
        if not result:
            result = safe[:1] if safe else current[:1]

        final_result = result[:3]
        remaining_risky = sum(1 for item in final_result if self._is_outdoor_high_exposure_attraction(item))
        repaired_count = max(0, len(risky) - remaining_risky)
        return final_result, repaired_count

    def _is_indoor_candidate_attraction(self, attraction: Attraction) -> bool:
        text = " ".join(
            [
                self._safe_str(attraction.name, ""),
                self._safe_str(attraction.category, ""),
                self._safe_str(attraction.description, ""),
            ]
        ).lower()
        if not text:
            return False
        if self._is_outdoor_high_exposure_attraction(attraction):
            return False

        indoor_terms = [
            "博物馆",
            "美术馆",
            "科技馆",
            "纪念馆",
            "展览馆",
            "艺术馆",
            "文化馆",
            "图书馆",
            "剧院",
            "museum",
            "gallery",
            "indoor",
        ]
        if any(term in text for term in indoor_terms):
            return True

        neutral_terms = ["历史文化", "文化街区", "古迹", "遗址", "祠堂", "宫", "寺", "塔", "故居"]
        return any(term in text for term in neutral_terms)

    def _merge_repair_note_into_suggestions(self, base_suggestions: str, repaired_count: int) -> str:
        note = f"【异常自动修复】已根据天气风险自动替换 {repaired_count} 个高暴露户外景点。"
        base = self._safe_str(base_suggestions, "")
        if note in base:
            return base
        if not base:
            return note
        return f"{base}\n{note}"

    def _collect_weather_activity_alerts(self, trip_plan: TripPlan, fallback_weather: List[WeatherInfo]) -> List[str]:
        weather_by_date = self._build_weather_text_by_date(trip_plan.weather_info or fallback_weather)
        alerts: List[str] = []

        for day in trip_plan.days:
            weather_text = self._safe_str(weather_by_date.get(day.date), "")
            if not weather_text or not self._is_rainy_weather_text(weather_text):
                continue

            risky_names: List[str] = []
            seen: set[str] = set()
            for attraction in day.attractions:
                if not self._is_outdoor_high_exposure_attraction(attraction):
                    continue
                name = self._safe_str(attraction.name, "景点")
                key = self._normalize_name(name)
                if key in seen:
                    continue
                seen.add(key)
                risky_names.append(name)

            if risky_names:
                alerts.append(
                    f"{day.date} 预报有雨（{weather_text}），当天包含高暴露户外项目: {', '.join(risky_names[:3])}"
                )

        return alerts

    def _build_weather_text_by_date(self, weather_info: List[WeatherInfo]) -> Dict[str, str]:
        weather_by_date: Dict[str, str] = {}
        for item in weather_info:
            date = self._safe_str(getattr(item, "date", ""), "")
            if not date:
                continue
            day_weather = self._safe_str(getattr(item, "day_weather", ""), "")
            night_weather = self._safe_str(getattr(item, "night_weather", ""), "")
            weather_by_date[date] = f"{day_weather}/{night_weather}".strip("/")
        return weather_by_date

    def _is_rainy_weather_text(self, text: str) -> bool:
        value = self._safe_str(text, "").lower()
        if not value:
            return False
        rainy_terms = ["雨", "阵雨", "雷雨", "雷阵雨", "暴雨", "中雨", "大雨", "小雨", "雨夹雪", "thunder", "rain"]
        return any(term in value for term in rainy_terms)

    def _is_outdoor_high_exposure_attraction(self, attraction: Attraction) -> bool:
        text = " ".join(
            [
                self._safe_str(attraction.name, ""),
                self._safe_str(attraction.category, ""),
                self._safe_str(attraction.description, ""),
            ]
        ).lower()
        if not text:
            return False

        indoor_terms = ["博物馆", "美术馆", "科技馆", "展览馆", "商场", "购物中心", "室内", "剧院", "图书馆", "museum"]
        if any(term in text for term in indoor_terms):
            return False

        outdoor_terms = [
            "长城",
            "登山",
            "爬山",
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
            "海滩",
            "沙滩",
            "海岛",
            "风景区",
            "hiking",
        ]
        return any(term in text for term in outdoor_terms)

    def _merge_safety_alerts_into_suggestions(self, base_suggestions: str, alerts: List[str]) -> str:
        note = "【异常检查提醒】" + "；".join(alerts[:3]) + "。建议改为室内项目或调整至无雨时段。"
        base = self._safe_str(base_suggestions, "")
        if note in base:
            return base
        if not base:
            return note
        return f"{base}\n{note}"

    def _handle_error(self, state: TripPlannerState) -> Dict[str, Any]:
        error_msg = self._safe_str(state.get("error"), "未知错误")
        logger.warning("Handle error: %s", error_msg)
        fallback = self._create_fallback_plan(state["request"])
        return {
            "trip_plan": fallback,
            "current_step": "error_handled",
            "messages": [{"role": "assistant", "content": f"流程异常，已返回备用行程: {error_msg}"}],
        }

    # ---------------------------------------------------------------------
    # Prompt builders
    # ---------------------------------------------------------------------
    def _build_task_breakdown(self, request: TripRequest) -> Dict[str, Any]:
        prefs = [self._safe_str(p, "") for p in (request.preferences or []) if self._safe_str(p, "")]
        attraction_kw = self._pick_attraction_keyword(request, prefs)
        meal_focus = [p for p in prefs if self._contains_food_intent(p)]
        free_text = self._safe_str(request.free_text_input, "")
        if self._contains_food_intent(free_text):
            meal_focus.append(free_text)
        hotel_kw = f"{request.accommodation} 酒店".strip() or "酒店"
        return {
            "attraction": {"city": request.city, "keywords": attraction_kw},
            "weather": {"city": request.city},
            "hotel": {"city": request.city, "keywords": hotel_kw},
            "constraints": {
                "days": request.travel_days,
                "transportation": request.transportation,
                "accommodation": request.accommodation,
                "meal_focus": "，".join(meal_focus[:3]) if meal_focus else "",
            },
        }

    def _build_attraction_query(self, request: TripRequest, keyword: str) -> str:
        return (
            f"请推荐{request.city}的{keyword}，只返回 JSON 数组。"
            "景点必须是城市地标、历史文化景区、博物馆、公园等，不要输出餐厅或小吃店。"
            "字段: name,address,location{longitude,latitude},visit_duration,description,category,ticket_price。"
        )

    def _build_planner_query(
        self,
        request: TripRequest,
        attractions: List[Attraction],
        weather: List[WeatherInfo],
        hotels: List[Hotel],
    ) -> str:
        def as_list(items: List[Any], limit: int = 20) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for item in items[:limit]:
                if hasattr(item, "model_dump") and callable(getattr(item, "model_dump")):
                    out.append(item.model_dump())
                elif isinstance(item, dict):
                    out.append(item)
            return out

        attractions_json = json.dumps(as_list(attractions, 30), ensure_ascii=False)
        weather_json = json.dumps(as_list(weather, 10), ensure_ascii=False)
        hotels_json = json.dumps(as_list(hotels, 20), ensure_ascii=False)

        return (
            "你是旅行规划助手，只返回 JSON 对象。\n"
            f"城市:{request.city}\n"
            f"开始:{request.start_date}\n"
            f"结束:{request.end_date}\n"
            f"天数:{request.travel_days}\n"
            f"交通:{request.transportation}\n"
            f"住宿:{request.accommodation}\n"
            f"偏好:{', '.join(request.preferences) if request.preferences else '无'}\n"
            f"额外要求:{request.free_text_input or '无'}\n\n"
            f"景点数据:{attractions_json}\n"
            f"天气数据:{weather_json}\n"
            f"酒店数据:{hotels_json}\n\n"
            "输出字段必须包含: city,start_date,end_date,days,weather_info,overall_suggestions,budget。\n"
            "days 每天包含: date,day_index,description,transportation,accommodation,hotel,attractions,meals。\n"
            "硬性要求:\n"
            "1) day_index 必须从0开始连续递增。\n"
            "2) date 必须从开始日期按天连续递增。\n"
            "3) 每天 attractions 2-3 个，且以城市地标/文化景点为主，不要把餐厅小吃店当景点。\n"
            "4) 餐饮请写入 meals，meals 每天至少包含 breakfast/lunch/dinner 三项。\n"
            "5) 若当天天气含雨，避免安排长城、登山、徒步、湿地等高暴露户外项目。"
        )

    # ---------------------------------------------------------------------
    # LLM
    # ---------------------------------------------------------------------
    def _invoke_llm_text(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", response)
        return self._value_to_text(content)

    # ---------------------------------------------------------------------
    # Safe helpers
    # ---------------------------------------------------------------------
    def _safe_str(self, value: Any, default: str = "") -> str:
        if value is None:
            return default
        try:
            s = str(value).strip()
            return s if s else default
        except Exception:
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        if value is None or isinstance(value, bool):
            return default
        try:
            return int(float(str(value).strip()))
        except Exception:
            return default

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None or isinstance(value, bool):
            return default
        try:
            return float(str(value).strip())
        except Exception:
            return default

    def _value_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    def _is_adcode(self, value: str) -> bool:
        return bool(re.fullmatch(r"\d{6}", self._safe_str(value, "")))

    def _normalize_city_name(self, city: str) -> str:
        text = self._safe_str(city, "")
        if not text:
            return ""
        text = text.replace(" ", "").replace("\u3000", "")
        if text.startswith("中国") and len(text) > 2:
            text = text[2:]

        suffixes = (
            "特别行政区",
            "维吾尔自治区",
            "壮族自治区",
            "回族自治区",
            "自治区",
            "省",
            "市",
        )
        for suffix in suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                break
        return text

    def _build_city_candidates(self, city: str) -> List[str]:
        city_text = self._safe_str(city, "")
        if not city_text:
            return []
        normalized = self._normalize_city_name(city_text)

        candidates: List[str] = []
        adcode = self._resolve_city_adcode(city_text)
        if adcode:
            candidates.append(adcode)
        candidates.append(city_text)
        if normalized and normalized != city_text:
            candidates.append(normalized)
        if normalized and not self._is_adcode(normalized):
            candidates.append(f"{normalized}市")
        # Keep order and dedupe.
        output: List[str] = []
        seen: set[str] = set()
        for item in candidates:
            key = self._safe_str(item, "")
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(key)
        return output

    def _resolve_city_adcode(self, city: str) -> str:
        city_text = self._safe_str(city, "")
        if not city_text:
            return ""
        if self._is_adcode(city_text):
            return city_text

        short_city = self._normalize_city_name(city_text)
        probes: List[str] = []
        for value in (city_text, short_city, f"{short_city}市" if short_city else ""):
            key = self._safe_str(value, "")
            if key and key not in probes:
                probes.append(key)

        for key in probes:
            ad = self._safe_str(self._city_adcode_cache.get(key), "")
            if ad:
                self._city_adcode_cache[city_text] = ad
                return ad

        # Fuzzy map matching, e.g. “杭州市西湖区” -> “杭州”.
        for key, value in self._city_adcode_cache.items():
            if not key:
                continue
            if key in city_text or city_text in key or (short_city and (key in short_city or short_city in key)):
                ad = self._safe_str(value, "")
                if ad:
                    self._city_adcode_cache[city_text] = ad
                    if short_city:
                        self._city_adcode_cache[short_city] = ad
                    return ad

        tool = self._find_tool("maps_geo")
        if tool is None:
            return ""

        for address in probes:
            payload = {"address": address}
            try:
                raw = self._tool_result_to_text(tool.invoke(payload))
                parsed = self._safe_parse_json_payload(raw)
                if isinstance(parsed, dict):
                    adcode = self._safe_str(parsed.get("adcode"), "")
                    if not adcode:
                        adcode = self._safe_str(parsed.get("citycode"), "")
                    if adcode:
                        self._city_adcode_cache[city_text] = adcode
                        if short_city:
                            self._city_adcode_cache[short_city] = adcode
                        return adcode

                records: Any = None
                if isinstance(parsed, dict):
                    records = parsed.get("return") or parsed.get("geocodes") or parsed.get("data")
                elif isinstance(parsed, list):
                    records = parsed
                if isinstance(records, list) and records:
                    first = records[0]
                    if isinstance(first, dict):
                        adcode = self._safe_str(first.get("adcode"), "")
                        if not adcode:
                            adcode = self._safe_str(first.get("citycode"), "")
                        if adcode:
                            self._city_adcode_cache[city_text] = adcode
                            if short_city:
                                self._city_adcode_cache[short_city] = adcode
                            return adcode
            except Exception:
                continue
        return ""

    def _contains_food_intent(self, text: str) -> bool:
        value = self._safe_str(text, "").lower()
        if not value:
            return False
        food_terms = [
            "美食",
            "小吃",
            "餐厅",
            "餐馆",
            "饭店",
            "饭馆",
            "酒楼",
            "酒家",
            "早茶",
            "夜宵",
            "海鲜",
            "火锅",
            "烧烤",
            "咖啡",
            "奶茶",
            "甜品",
            "吃",
        ]
        return any(term in value for term in food_terms)

    def _pick_attraction_keyword(self, request: TripRequest, preferences: List[str]) -> str:
        non_food = [p for p in preferences if not self._contains_food_intent(p)]
        if non_food:
            base = self._safe_str(non_food[0], "景点")
            scenic_terms = ["景点", "景区", "博物馆", "公园", "地标"]
            if not any(term in base for term in scenic_terms):
                base = f"{base} 景点"
            return base

        return "景点"

    def _is_food_poi(self, attraction: Attraction) -> bool:
        text = " ".join(
            [
                self._safe_str(attraction.name, ""),
                self._safe_str(attraction.category, ""),
                self._safe_str(attraction.description, ""),
                self._safe_str(attraction.address, ""),
            ]
        ).lower()
        if not text:
            return False
        food_terms = [
            "美食",
            "小吃",
            "餐厅",
            "餐馆",
            "饭店",
            "饭馆",
            "酒楼",
            "酒家",
            "面店",
            "奶茶",
            "咖啡",
            "火锅",
            "海鲜",
            "烧烤",
            "茶餐厅",
            "饺子",
            "粥",
            "food",
            "restaurant",
            "cafe",
        ]
        return any(term in text for term in food_terms)

    def _is_attraction_in_city(self, attraction: Attraction, city: str) -> bool:
        city_full = self._safe_str(city, "")
        city_short = city_full.replace("市", "")
        text = " ".join(
            [
                self._safe_str(attraction.name, ""),
                self._safe_str(attraction.address, ""),
                self._safe_str(attraction.description, ""),
            ]
        )
        if not text:
            return False
        if city_full and city_full in text:
            return True
        if city_short and city_short in text:
            return True
        return False

    def _merge_unique_attractions(self, items: List[Attraction], max_items: int = 30) -> List[Attraction]:
        result: List[Attraction] = []
        seen: set[str] = set()
        for item in items:
            key = self._normalize_name(self._safe_str(item.name, ""))
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
            if len(result) >= max_items:
                break
        return result

    # ---------------------------------------------------------------------
    # MCP tool helpers
    # ---------------------------------------------------------------------
    def _find_tool(self, fragment: str):
        key = fragment.lower()
        if key in self._tool_lookup_cache:
            return self._tool_lookup_cache[key]
        for tool in self.tools:
            name = self._safe_str(getattr(tool, "name", ""), "").lower()
            if key in name:
                self._tool_lookup_cache[key] = tool
                return tool
        self._tool_lookup_cache[key] = None
        return None

    def _tool_input_keys(self, tool: Any) -> set[str]:
        tool_name = self._safe_str(getattr(tool, "name", ""), "")
        if tool_name and tool_name in self._tool_input_keys_cache:
            return set(self._tool_input_keys_cache[tool_name])

        keys: set[str] = set()
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            if isinstance(args_schema, dict):
                props = args_schema.get("properties")
                if isinstance(props, dict):
                    keys.update(str(k).lower() for k in props.keys())
            else:
                model_fields = getattr(args_schema, "model_fields", None)
                if isinstance(model_fields, dict):
                    keys.update(str(k).lower() for k in model_fields.keys())
                else:
                    schema_getter = getattr(args_schema, "model_json_schema", None)
                    try:
                        schema_data = schema_getter() if callable(schema_getter) else None
                    except Exception:
                        schema_data = None
                    if isinstance(schema_data, dict) and isinstance(schema_data.get("properties"), dict):
                        keys.update(str(k).lower() for k in schema_data["properties"].keys())
        if tool_name:
            self._tool_input_keys_cache[tool_name] = set(keys)
        return keys

    def _pick_input_key(self, input_keys: set[str], candidates: List[str], default: Optional[str]) -> Optional[str]:
        if not input_keys:
            return default
        for candidate in candidates:
            if candidate in input_keys:
                return candidate
        return default

    def _dedupe_payloads(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for payload in payloads:
            try:
                key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            except Exception:
                key = str(payload)
            if key in seen:
                continue
            seen.add(key)
            unique.append(payload)
        return unique

    def _tool_result_to_text(self, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result.strip()

        texts: List[str] = []

        def visit(value: Any):
            if value is None:
                return
            if isinstance(value, str):
                s = value.strip()
                if s:
                    texts.append(s)
                return
            if isinstance(value, dict):
                t = value.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
                for k, v in value.items():
                    if k != "text":
                        visit(v)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                return
            visit(getattr(value, "content", None))

        visit(result)
        if texts:
            for t in texts:
                if t.startswith("{") or t.startswith("["):
                    return t
            return "\n".join(texts)

        return self._safe_str(result, "")

    def _extract_json_candidate(self, text: str) -> str:
        start = -1
        stack: List[str] = []
        in_string = False
        escape = False
        for idx, ch in enumerate(text):
            if start == -1:
                if ch in "{[":
                    start = idx
                    stack = [ch]
                continue
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue
            if ch == "\"":
                in_string = True
                continue
            if ch in "{[":
                stack.append(ch)
                continue
            if ch in "}]":
                if not stack:
                    continue
                left = stack[-1]
                if (left == "{" and ch == "}") or (left == "[" and ch == "]"):
                    stack.pop()
                    if not stack and start >= 0:
                        return text[start : idx + 1]
        return ""

    def _safe_parse_json_payload(self, payload: Any) -> Optional[Any]:
        if isinstance(payload, (dict, list)):
            return payload
        text = self._safe_str(payload, "").strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:
            pass

        if "```" in text:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
            try:
                return json.loads(text)
            except Exception:
                pass

        candidate = self._extract_json_candidate(text)
        if candidate:
            try:
                return json.loads(candidate)
            except Exception:
                try:
                    value = ast.literal_eval(candidate)
                    if isinstance(value, (dict, list)):
                        return value
                except Exception:
                    pass

        try:
            value = ast.literal_eval(text)
            if isinstance(value, (dict, list)):
                return value
            reparsed_text = self._tool_result_to_text(value)
            if reparsed_text and reparsed_text != text:
                return self._safe_parse_json_payload(reparsed_text)
        except Exception:
            pass
        return None

    def _extract_tool_error(self, parsed: Any) -> str:
        if isinstance(parsed, dict):
            return self._safe_str(parsed.get("error"), "")
        return ""

    def _invoke_maps_text_search(self, city: str, keywords: str) -> str:
        tool = self._find_tool("maps_text_search")
        if tool is None:
            return ""

        keys = self._tool_input_keys(tool)
        query_key = self._pick_input_key(keys, ["keywords", "query", "keyword"], "keywords")
        city_key = self._pick_input_key(keys, ["city", "cityname", "region", "name"], "city")
        citylimit_key = self._pick_input_key(keys, ["citylimit"], None)

        base_query = self._safe_str(keywords, "").strip() or "热门景点"
        city_candidates = self._build_city_candidates(city)
        payloads: List[Dict[str, Any]] = []
        for city_value in city_candidates:
            payloads.append({query_key: base_query, city_key: city_value})
            payloads.append({query_key: f"{city_value} {base_query}", city_key: city_value})
            if citylimit_key:
                payloads.append({query_key: base_query, city_key: city_value, citylimit_key: "true"})
        payloads.append({query_key: f"{city} {base_query}"})
        payloads.append({query_key: base_query})
        payloads = self._dedupe_payloads(payloads)

        last_error = ""
        for payload in payloads:
            try:
                raw = self._tool_result_to_text(tool.invoke(payload))
                parsed = self._safe_parse_json_payload(raw)
                if isinstance(parsed, dict):
                    error_text = self._extract_tool_error(parsed)
                    if error_text:
                        last_error = error_text
                        continue
                    pois = parsed.get("pois")
                    if isinstance(pois, list) and len(pois) > 0:
                        return json.dumps(parsed, ensure_ascii=False)
                if isinstance(parsed, list) and parsed:
                    return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                continue
        if last_error:
            logger.warning("maps_text_search returned empty/error: city=%s keywords=%s error=%s", city, base_query, last_error)
        else:
            logger.warning("maps_text_search returned empty: city=%s keywords=%s payloads=%d", city, base_query, len(payloads))
        return ""

    def _invoke_maps_weather(self, city: str) -> str:
        tool = self._find_tool("maps_weather")
        if tool is None:
            return ""

        keys = self._tool_input_keys(tool)
        city_key = self._pick_input_key(keys, ["city", "cityname", "name"], "city")
        ext_key = self._pick_input_key(keys, ["extensions"], None)

        payloads: List[Dict[str, Any]] = []
        for city_value in self._build_city_candidates(city):
            payloads.append({city_key: city_value})
            if ext_key:
                payloads.append({city_key: city_value, ext_key: "all"})
        payloads = self._dedupe_payloads(payloads)

        last_error = ""
        for payload in payloads:
            try:
                raw = self._tool_result_to_text(tool.invoke(payload))
                parsed = self._safe_parse_json_payload(raw)
                if isinstance(parsed, dict):
                    error_text = self._extract_tool_error(parsed)
                    if error_text:
                        last_error = error_text
                        continue
                    has_forecasts = isinstance(parsed.get("forecasts"), list) and len(parsed.get("forecasts", [])) > 0
                    has_casts = isinstance(parsed.get("casts"), list) and len(parsed.get("casts", [])) > 0
                    has_lives = isinstance(parsed.get("lives"), list) and len(parsed.get("lives", [])) > 0
                    if has_forecasts or has_casts or has_lives:
                        return json.dumps(parsed, ensure_ascii=False)
                if isinstance(parsed, list) and parsed:
                    return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                continue
        if last_error:
            logger.warning("maps_weather returned empty/error: city=%s error=%s", city, last_error)
        else:
            logger.warning("maps_weather returned empty: city=%s payloads=%d", city, len(payloads))
        return ""

    def _invoke_maps_search_detail(self, poi_id: str) -> Optional[Dict[str, Any]]:
        pid = self._safe_str(poi_id, "")
        if not pid:
            return None
        with self._poi_detail_cache_lock:
            cached = self._poi_detail_cache.get(pid)
        if cached is not None:
            return cached

        tool = self._find_tool("maps_search_detail")
        if tool is None:
            return None

        keys = self._tool_input_keys(tool)
        id_key = self._pick_input_key(keys, ["id", "ids", "poi_id"], "id")
        payload = {id_key: pid}
        try:
            raw = self._tool_result_to_text(tool.invoke(payload))
            parsed = self._safe_parse_json_payload(raw)
            if isinstance(parsed, dict):
                error_text = self._extract_tool_error(parsed)
                if error_text:
                    return None
                detail_payload = self._normalize_detail_payload(parsed)
                with self._poi_detail_cache_lock:
                    self._poi_detail_cache[pid] = detail_payload
                return detail_payload
            if isinstance(parsed, list):
                for record in parsed:
                    if isinstance(record, dict):
                        detail_payload = self._normalize_detail_payload(record)
                        with self._poi_detail_cache_lock:
                            self._poi_detail_cache[pid] = detail_payload
                        return detail_payload
        except Exception:
            return None
        return None

    def _normalize_detail_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}

        # Some providers wrap detail fields under pois/data/items.
        candidates: List[Dict[str, Any]] = [payload]
        for key in ("pois", "data", "results", "items", "list"):
            value = payload.get(key)
            if isinstance(value, dict):
                candidates.append(value)
            elif isinstance(value, list):
                candidates.extend([x for x in value if isinstance(x, dict)])

        for item in candidates:
            has_core = any(
                self._safe_str(item.get(k), "")
                for k in ("name", "address", "location", "type", "typecode", "adname", "district")
            )
            if has_core:
                return item
        return payload

    def _enrich_item_with_detail(self, item: Dict[str, Any]) -> Dict[str, Any]:
        detail = self._invoke_maps_search_detail(item.get("id") or item.get("poi_id") or "")
        if not isinstance(detail, dict):
            return item
        merged = dict(item)
        for key in (
            "name",
            "address",
            "location",
            "city",
            "cityname",
            "adname",
            "district",
            "type",
            "typecode",
            "rating",
            "description",
            "photos",
            "tel",
            "business_area",
        ):
            value = detail.get(key)
            if value not in (None, "", []):
                merged[key] = value
        return merged

    def _enrich_items_with_detail(
        self,
        items: List[Dict[str, Any]],
        should_enrich: Callable[[Dict[str, Any]], bool],
        detail_budget: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if not items or detail_budget <= 0:
            return items, 0

        candidates: List[Tuple[int, Dict[str, Any]]] = []
        for idx, item in enumerate(items):
            if len(candidates) >= detail_budget:
                break
            if not isinstance(item, dict):
                continue
            poi_id = self._safe_str(item.get("id") or item.get("poi_id"), "")
            if not poi_id:
                continue
            if should_enrich(item):
                candidates.append((idx, item))

        if not candidates:
            return items, 0

        updated = list(items)
        workers = min(self._detail_enrich_workers, len(candidates))
        workers = max(1, workers)

        if workers <= 1:
            for idx, item in candidates:
                updated[idx] = self._enrich_item_with_detail(item)
            return updated, len(candidates)

        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="poi-detail") as executor:
                future_to_idx = {
                    executor.submit(self._enrich_item_with_detail, item): idx
                    for idx, item in candidates
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        if isinstance(result, dict):
                            updated[idx] = result
                    except Exception as exc:
                        logger.debug("POI detail enrich failed idx=%d: %s", idx, exc)
        except Exception as exc:
            logger.warning("POI detail parallel enrich failed, fallback to serial: %s", exc)
            for idx, item in candidates:
                try:
                    updated[idx] = self._enrich_item_with_detail(item)
                except Exception:
                    continue

        return updated, len(candidates)

    # ---------------------------------------------------------------------
    # Data parsers
    # ---------------------------------------------------------------------
    def _unwrap_records(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("pois", "data", "results", "items", "list"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [data]
        return []

    def _parse_location(self, raw_location: Any, item: Optional[Dict[str, Any]] = None) -> Location:
        item = item or {}
        lng = 0.0
        lat = 0.0
        if isinstance(raw_location, dict):
            lng = raw_location.get("longitude") or raw_location.get("lng") or raw_location.get("lon") or 0.0
            lat = raw_location.get("latitude") or raw_location.get("lat") or 0.0
        elif isinstance(raw_location, str):
            parts = [p.strip() for p in raw_location.split(",")]
            if len(parts) >= 2:
                lng = self._safe_float(parts[0], 0.0)
                lat = self._safe_float(parts[1], 0.0)
        elif isinstance(raw_location, (list, tuple)) and len(raw_location) >= 2:
            lng = self._safe_float(raw_location[0], 0.0)
            lat = self._safe_float(raw_location[1], 0.0)
        if lng == 0.0 and lat == 0.0:
            lng = item.get("longitude") or item.get("lng") or item.get("lon") or 0.0
            lat = item.get("latitude") or item.get("lat") or 0.0
        return Location(longitude=self._safe_float(lng, 0.0), latitude=self._safe_float(lat, 0.0))

    def _is_valid_location(self, location: Optional[Location]) -> bool:
        if location is None:
            return False
        lng = self._safe_float(getattr(location, "longitude", None), 0.0)
        lat = self._safe_float(getattr(location, "latitude", None), 0.0)
        if not (-180 <= lng <= 180 and -90 <= lat <= 90):
            return False
        return not (abs(lng) < 1e-8 and abs(lat) < 1e-8)

    def _is_incomplete_address(self, address: str) -> bool:
        text = re.sub(r"\s+", "", self._safe_str(address, ""))
        if not text:
            return True
        if text in {"未知", "暂无", "不详", "市内", "城区"}:
            return True
        return len(text) < 4

    def _is_incomplete_description(self, description: str) -> bool:
        text = re.sub(r"\s+", "", self._safe_str(description, ""))
        if not text:
            return True
        if len(text) < 14:
            return True
        low_info_tokens = ("著名景点", "值得一游", "风景优美", "打卡地", "热门景点")
        if any(token in text for token in low_info_tokens) and len(text) < 26:
            return True
        return False

    def _fallback_attraction_address(self, item: Dict[str, Any], city: str = "") -> str:
        candidates = [
            item.get("address"),
            item.get("adname"),
            item.get("district"),
            item.get("business_area"),
            item.get("cityname"),
            item.get("city"),
        ]
        for value in candidates:
            text = self._safe_str(value, "")
            if not self._is_incomplete_address(text):
                return text
        city_text = self._safe_str(city, "") or self._safe_str(item.get("city") or item.get("cityname"), "")
        return f"{city_text}市区" if city_text else "地址待补充"

    def _fallback_attraction_description(
        self,
        item: Dict[str, Any],
        name: str,
        category: str,
        address: str,
        city: str = "",
        visit_duration: int = 120,
    ) -> str:
        for key in ("description", "intro", "summary"):
            text = self._safe_str(item.get(key), "")
            if not self._is_incomplete_description(text):
                return text
        city_text = self._safe_str(city, "") or self._safe_str(item.get("city") or item.get("cityname"), "")
        location_text = address if not self._is_incomplete_address(address) else (city_text or "当地")
        category_text = self._safe_str(category, "") or "景点"
        minutes = max(30, self._safe_int(visit_duration, 120))
        return f"{name}位于{location_text}，属于{category_text}类景点，建议安排约{minutes}分钟游览。"

    def _needs_attraction_detail(self, item: Dict[str, Any]) -> bool:
        if not isinstance(item, dict):
            return False
        location = self._parse_location(item.get("location"), item)
        if not self._is_valid_location(location):
            return True
        address = self._safe_str(item.get("address"), "")
        description = self._safe_str(item.get("description"), "")
        return self._is_incomplete_address(address) or self._is_incomplete_description(description)

    def _parse_attractions(self, response: str) -> List[Attraction]:
        started = time.perf_counter()
        parsed = self._safe_parse_json_payload(response)
        if parsed is None:
            return []
        records = self._unwrap_records(parsed)
        attractions: List[Attraction] = []
        raw_items = [dict(item) for item in records[:40] if isinstance(item, dict)]
        detail_budget = 10
        raw_items, detail_calls = self._enrich_items_with_detail(
            raw_items,
            should_enrich=self._needs_attraction_detail,
            detail_budget=detail_budget,
        )
        for enriched in raw_items:
            location = self._parse_location(enriched.get("location"), enriched)
            if not self._is_valid_location(location):
                continue

            name = self._safe_str(enriched.get("name"), "未知景点")
            category = self._safe_str(enriched.get("category"), "")
            if not category:
                category = self._safe_str(enriched.get("type") or enriched.get("typecode"), "景点")
            city_name = self._safe_str(enriched.get("city") or enriched.get("cityname"), "")
            address = self._fallback_attraction_address(enriched, city=city_name)
            description = self._fallback_attraction_description(
                enriched,
                name=name,
                category=category,
                address=address,
                city=city_name,
                visit_duration=self._safe_int(enriched.get("visit_duration"), 120),
            )
            attractions.append(
                Attraction(
                    name=name,
                    address=address,
                    location=location,
                    visit_duration=self._safe_int(enriched.get("visit_duration"), 120),
                    description=description,
                    category=category or "景点",
                    rating=self._safe_float(enriched.get("rating"), 0.0) if enriched.get("rating") is not None else None,
                    photos=enriched.get("photos") if isinstance(enriched.get("photos"), list) else [],
                    poi_id=self._safe_str(enriched.get("id") or enriched.get("poi_id"), ""),
                    image_url=self._safe_str(enriched.get("image_url"), "") or None,
                    ticket_price=self._safe_int(enriched.get("ticket_price"), 0),
                )
            )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "parse_done type=attractions records=%d output=%d detail_calls=%d elapsed_ms=%d",
            len(records),
            len(attractions),
            detail_calls,
            elapsed_ms,
        )
        return attractions

    def _parse_weather(self, response: str) -> List[WeatherInfo]:
        parsed = self._safe_parse_json_payload(response)
        if parsed is None:
            return []

        records: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            records = [x for x in parsed if isinstance(x, dict)]
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("forecasts"), list):
                for block in parsed["forecasts"]:
                    if not isinstance(block, dict):
                        continue
                    if isinstance(block.get("casts"), list):
                        records.extend([x for x in block["casts"] if isinstance(x, dict)])
                        continue
                    # Some MCP responses already flatten daily forecast records in forecasts.
                    if block.get("date") or block.get("dayweather") or block.get("nightweather"):
                        records.append(block)
            elif isinstance(parsed.get("casts"), list):
                records = [x for x in parsed["casts"] if isinstance(x, dict)]
            elif isinstance(parsed.get("lives"), list):
                records = [x for x in parsed["lives"] if isinstance(x, dict)]
            else:
                records = [parsed]

        weather_info: List[WeatherInfo] = []
        for item in records[:15]:
            date_str = self._safe_str(item.get("date") or item.get("reporttime"), "")
            if len(date_str) >= 10:
                date_str = date_str[:10]
            weather_info.append(
                WeatherInfo(
                    date=date_str,
                    day_weather=self._safe_str(item.get("day_weather") or item.get("dayweather") or item.get("weather"), ""),
                    night_weather=self._safe_str(
                        item.get("night_weather") or item.get("nightweather") or item.get("weather"),
                        "",
                    ),
                    day_temp=item.get("day_temp") or item.get("daytemp") or item.get("temperature") or 0,
                    night_temp=item.get("night_temp") or item.get("nighttemp") or item.get("temperature") or 0,
                    wind_direction=self._safe_str(
                        item.get("wind_direction") or item.get("daywind") or item.get("winddirection"),
                        "",
                    ),
                    wind_power=self._safe_str(
                        item.get("wind_power") or item.get("daypower") or item.get("windpower"),
                        "",
                    ),
                )
            )
        return weather_info

    def _parse_hotels(self, response: str) -> List[Hotel]:
        started = time.perf_counter()
        parsed = self._safe_parse_json_payload(response)
        if parsed is None:
            return []
        records = self._unwrap_records(parsed)
        hotels: List[Hotel] = []
        raw_items = [dict(item) for item in records[:30] if isinstance(item, dict)]
        detail_budget = 4
        raw_items, detail_calls = self._enrich_items_with_detail(
            raw_items,
            should_enrich=lambda item: (
                not self._safe_str(item.get("name"), "") or not self._safe_str(item.get("address"), "")
            ),
            detail_budget=detail_budget,
        )
        for enriched in raw_items:
            location = None
            loc = self._parse_location(enriched.get("location"), enriched)
            if self._is_valid_location(loc):
                location = loc
            hotels.append(
                Hotel(
                    name=self._safe_str(enriched.get("name"), "推荐酒店"),
                    address=self._safe_str(enriched.get("address"), ""),
                    location=location,
                    price_range=self._safe_str(enriched.get("price_range"), ""),
                    rating=self._safe_str(enriched.get("rating"), ""),
                    distance=self._safe_str(enriched.get("distance"), ""),
                    type=self._safe_str(enriched.get("type"), ""),
                    estimated_cost=self._safe_int(enriched.get("estimated_cost"), 0),
                )
            )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "parse_done type=hotels records=%d output=%d detail_calls=%d elapsed_ms=%d",
            len(records),
            len(hotels),
            detail_calls,
            elapsed_ms,
        )
        return hotels

    def _parse_trip_plan(
        self,
        response: str,
        request: TripRequest,
        source_attractions: Optional[List[Attraction]] = None,
        source_hotels: Optional[List[Hotel]] = None,
    ) -> TripPlan:
        parsed = self._safe_parse_json_payload(response)
        if not isinstance(parsed, dict):
            return self._create_fallback_plan(request)

        source_attractions = source_attractions or []
        source_hotels = source_hotels or []
        attraction_index = self._build_name_index(source_attractions)
        hotel_index = self._build_name_index(source_hotels)

        trip_plan = TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=[],
            weather_info=[],
            overall_suggestions=self._safe_str(parsed.get("overall_suggestions"), ""),
            budget=None,
        )

        for w in parsed.get("weather_info", []):
            if not isinstance(w, dict):
                continue
            trip_plan.weather_info.append(
                WeatherInfo(
                    date=self._safe_str(w.get("date"), ""),
                    day_weather=self._safe_str(w.get("day_weather") or w.get("dayweather"), ""),
                    night_weather=self._safe_str(w.get("night_weather") or w.get("nightweather"), ""),
                    day_temp=w.get("day_temp") or w.get("daytemp") or 0,
                    night_temp=w.get("night_temp") or w.get("nighttemp") or 0,
                    wind_direction=self._safe_str(w.get("wind_direction") or w.get("daywind"), ""),
                    wind_power=self._safe_str(w.get("wind_power") or w.get("daypower"), ""),
                )
            )

        parsed_days = parsed.get("days", [])
        if not isinstance(parsed_days, list):
            parsed_days = []

        try:
            start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
        except ValueError:
            start_dt = datetime.now()

        target_days = max(1, self._safe_int(request.travel_days, 1))
        for idx in range(target_days):
            raw_day = parsed_days[idx] if idx < len(parsed_days) else {}
            day_data = raw_day if isinstance(raw_day, dict) else {}

            attractions: List[Attraction] = []
            dropped_attractions = 0
            day_attractions_raw = day_data.get("attractions", [])
            for ad in day_attractions_raw:
                if not isinstance(ad, dict):
                    continue

                name = self._safe_str(ad.get("name"), "未知景点")
                matched = self._match_index_item(attraction_index, name)
                location = self._parse_location(ad.get("location"), ad)
                if not self._is_valid_location(location):
                    matched_loc = self._item_location(matched)
                    if matched_loc is not None:
                        location = matched_loc
                if not self._is_valid_location(location):
                    dropped_attractions += 1
                    logger.debug("trip_plan_attraction_drop day=%d reason=invalid_location name=%s", idx, name)
                    continue

                if source_attractions and matched is None:
                    logger.debug("trip_plan_attraction_unmatched day=%d name=%s", idx, name)

                category = self._safe_str(ad.get("category"), "")
                if not category:
                    category = self._safe_str(ad.get("type") or ad.get("typecode"), "景点")

                address = self._fallback_attraction_address(ad, city=request.city)
                matched_address = self._item_address(matched)
                if self._is_incomplete_address(address) and not self._is_incomplete_address(matched_address):
                    address = matched_address

                description = self._fallback_attraction_description(
                    ad,
                    name=name,
                    category=category,
                    address=address,
                    city=request.city,
                    visit_duration=self._safe_int(ad.get("visit_duration"), 120),
                )
                matched_desc = self._item_description(matched)
                if self._is_incomplete_description(description) and not self._is_incomplete_description(matched_desc):
                    description = matched_desc

                ticket_price = self._safe_int(ad.get("ticket_price"), 0)
                if ticket_price <= 0:
                    ticket_price = self._item_ticket_price(matched)

                attractions.append(
                    Attraction(
                        name=name,
                        address=address,
                        location=location,
                        visit_duration=self._safe_int(ad.get("visit_duration"), 120),
                        description=description,
                        category=category or "景点",
                        ticket_price=ticket_price,
                    )
                )

            if dropped_attractions > 0:
                logger.info(
                    "trip_plan_attraction_drop_summary day=%d input=%d dropped=%d",
                    idx,
                    len(day_attractions_raw) if isinstance(day_attractions_raw, list) else 0,
                    dropped_attractions,
                )

            non_food_attractions = [a for a in attractions if not self._is_food_poi(a)]
            if non_food_attractions:
                attractions = non_food_attractions
            if not source_attractions:
                city_attractions = [a for a in attractions if self._is_attraction_in_city(a, request.city)]
                if city_attractions:
                    attractions = city_attractions

            if not attractions and source_attractions:
                attractions = self._pick_day_attraction_fallback(source_attractions, idx)
            attractions = self._ensure_day_attractions_count(attractions, source_attractions, idx)

            meals: List[Meal] = []
            for md in day_data.get("meals", []):
                if not isinstance(md, dict):
                    continue
                meal_loc = None
                if md.get("location") is not None:
                    loc = self._parse_location(md.get("location"), md)
                    if self._is_valid_location(loc):
                        meal_loc = loc
                meals.append(
                    Meal(
                        type=self._safe_str(md.get("type"), "lunch"),
                        name=self._safe_str(md.get("name"), "餐食推荐"),
                        address=self._safe_str(md.get("address"), "") or None,
                        location=meal_loc,
                        description=self._safe_str(md.get("description"), "") or None,
                        estimated_cost=self._safe_int(md.get("estimated_cost"), 0),
                    )
                )
            meals = self._ensure_day_meals(meals, request, attractions, idx)

            hotel = None
            hd = day_data.get("hotel")
            if source_hotels:
                matched_hotel = None
                if isinstance(hd, dict):
                    hotel_name = self._safe_str(hd.get("name"), "")
                    if hotel_name:
                        matched_hotel = self._match_index_item(hotel_index, hotel_name)
                if isinstance(matched_hotel, Hotel):
                    hotel = matched_hotel
                else:
                    hotel = source_hotels[min(idx, len(source_hotels) - 1)]
            elif isinstance(hd, dict):
                hotel_name = self._safe_str(hd.get("name"), "推荐酒店")
                hotel_loc = self._parse_location(hd.get("location"), hd)
                hotel = Hotel(
                    name=hotel_name,
                    address=self._safe_str(hd.get("address"), ""),
                    location=hotel_loc if self._is_valid_location(hotel_loc) else None,
                    price_range=self._safe_str(hd.get("price_range"), ""),
                    rating=self._safe_str(hd.get("rating"), ""),
                    distance=self._safe_str(hd.get("distance"), ""),
                    type=self._safe_str(hd.get("type"), ""),
                    estimated_cost=self._safe_int(hd.get("estimated_cost"), 0),
                )

            if hotel is not None:
                hotel = self._complete_hotel_fields(hotel, request, attractions)

            day_date = (start_dt + timedelta(days=idx)).strftime("%Y-%m-%d")
            day_desc = self._safe_str(day_data.get("description"), "")
            if not day_desc:
                day_desc = f"第{idx + 1}天城市探索与休闲体验。"

            trip_plan.days.append(
                DayPlan(
                    date=day_date,
                    day_index=idx,
                    description=day_desc,
                    transportation=self._safe_str(day_data.get("transportation"), request.transportation),
                    accommodation=self._safe_str(day_data.get("accommodation"), request.accommodation),
                    hotel=hotel,
                    attractions=attractions,
                    meals=meals,
                )
            )

        if not trip_plan.days:
            return self._create_fallback_plan(request)

        parsed_budget = None
        budget_data = parsed.get("budget")
        if isinstance(budget_data, dict):
            parsed_budget = Budget(
                total_attractions=self._safe_int(budget_data.get("total_attractions"), 0),
                total_hotels=self._safe_int(budget_data.get("total_hotels"), 0),
                total_meals=self._safe_int(budget_data.get("total_meals"), 0),
                total_transportation=self._safe_int(budget_data.get("total_transportation"), 0),
                total=self._safe_int(budget_data.get("total"), 0),
            )

        estimated_budget = self._estimate_budget_from_plan(trip_plan, request)
        if parsed_budget is None:
            trip_plan.budget = estimated_budget
        else:
            trip_plan.budget = Budget(
                total_attractions=parsed_budget.total_attractions or estimated_budget.total_attractions,
                total_hotels=parsed_budget.total_hotels or estimated_budget.total_hotels,
                total_meals=parsed_budget.total_meals or estimated_budget.total_meals,
                total_transportation=parsed_budget.total_transportation or estimated_budget.total_transportation,
                total=parsed_budget.total or estimated_budget.total,
            )

        return trip_plan

    def _normalize_name(self, name: str) -> str:
        normalized = self._safe_str(name, "").lower()
        if not normalized:
            return ""
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(r"[()（）\[\]【】,，.。:：·•\-_/\\]", "", normalized)
        return normalized

    def _name_aliases(self, name: str) -> List[str]:
        raw = self._safe_str(name, "")
        base_key = self._normalize_name(raw)
        if not base_key:
            return []

        aliases: List[str] = [base_key]
        base_text = re.split(r"[（(]", raw, maxsplit=1)[0]
        part_candidates = re.split(r"[-—·•/｜|]", base_text)
        for part in part_candidates:
            key = self._normalize_name(part)
            if key and len(key) >= 2:
                aliases.append(key)

        suffixes = (
            "景区",
            "风景区",
            "旅游区",
            "国家公园",
            "湿地公园",
            "公园",
            "博物馆",
            "美术馆",
            "展览馆",
            "遗址",
            "古镇",
            "景点",
        )
        for suffix in suffixes:
            sfx = self._normalize_name(suffix)
            if sfx and base_key.endswith(sfx) and len(base_key) > len(sfx) + 1:
                aliases.append(base_key[: -len(sfx)])

        unique: List[str] = []
        seen: set[str] = set()
        for alias in aliases:
            if alias in seen:
                continue
            seen.add(alias)
            unique.append(alias)
        return unique

    def _build_name_index(self, items: List[Any]) -> Dict[str, List[Any]]:
        index: Dict[str, List[Any]] = {}
        for item in items:
            item_name = self._item_name(item)
            keys = self._name_aliases(item_name)
            for key in keys:
                index.setdefault(key, []).append(item)
        return index

    def _match_index_item(self, index: Dict[str, List[Any]], name: str) -> Optional[Any]:
        keys = self._name_aliases(name)
        if not keys:
            return None

        for key in keys:
            if key in index and index[key]:
                return index[key][0]

        # Fall back to substring match with longer aliases first.
        for key in sorted(keys, key=len, reverse=True):
            for idx_key, values in index.items():
                if not values:
                    continue
                if key in idx_key or idx_key in key:
                    return values[0]
        return None

    def _item_name(self, item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, dict):
            return self._safe_str(item.get("name"), "")
        return self._safe_str(getattr(item, "name", ""), "")

    def _item_address(self, item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, dict):
            return self._safe_str(item.get("address"), "")
        return self._safe_str(getattr(item, "address", ""), "")

    def _item_description(self, item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, dict):
            return self._safe_str(item.get("description"), "")
        return self._safe_str(getattr(item, "description", ""), "")

    def _item_ticket_price(self, item: Any) -> int:
        if item is None:
            return 0
        if isinstance(item, dict):
            return self._safe_int(item.get("ticket_price"), 0)
        return self._safe_int(getattr(item, "ticket_price", 0), 0)

    def _item_estimated_cost(self, item: Any) -> int:
        if item is None:
            return 0
        if isinstance(item, dict):
            return self._safe_int(item.get("estimated_cost"), 0)
        return self._safe_int(getattr(item, "estimated_cost", 0), 0)

    def _item_location(self, item: Any) -> Optional[Location]:
        if item is None:
            return None
        if isinstance(item, dict):
            loc = self._parse_location(item.get("location"), item)
            return loc if self._is_valid_location(loc) else None
        loc = getattr(item, "location", None)
        if isinstance(loc, Location) and self._is_valid_location(loc):
            return loc
        if loc is not None:
            parsed = self._parse_location(loc)
            if self._is_valid_location(parsed):
                return parsed
        return None

    def _first_valid_hotel_location(self, hotels: List[Hotel]) -> Optional[Location]:
        for hotel in hotels:
            loc = self._item_location(hotel)
            if loc is not None:
                return loc
        return None

    def _pick_day_attraction_fallback(self, source_attractions: List[Attraction], day_index: int) -> List[Attraction]:
        valid = [a for a in source_attractions if self._is_valid_location(a.location)]
        if not valid:
            return []
        pick_count = min(2, len(valid))
        start = (day_index * pick_count) % len(valid)
        result: List[Attraction] = []
        for offset in range(pick_count):
            result.append(valid[(start + offset) % len(valid)])
        return result

    def _ensure_day_attractions_count(
        self,
        attractions: List[Attraction],
        source_attractions: List[Attraction],
        day_index: int,
    ) -> List[Attraction]:
        result = self._merge_unique_attractions(attractions, max_items=3)
        if len(result) >= 2:
            return result[:3]

        pool = self._merge_unique_attractions(
            [
                a
                for a in source_attractions
                if self._is_valid_location(a.location) and not self._is_food_poi(a)
            ],
            max_items=60,
        )
        if not pool:
            return result

        target = 2 if len(pool) >= 2 else 1
        start = (day_index * 2) % max(1, len(pool))
        idx = 0
        while len(result) < target and idx < len(pool) * 2:
            candidate = pool[(start + idx) % len(pool)]
            exists = any(self._normalize_name(x.name) == self._normalize_name(candidate.name) for x in result)
            if not exists:
                result.append(candidate)
            idx += 1
        return result[:3]

    def _distance_km(self, a: Location, b: Location) -> float:
        from math import radians, sin, cos, asin, sqrt

        lon1, lat1, lon2, lat2 = map(
            radians,
            [
                self._safe_float(a.longitude, 0.0),
                self._safe_float(a.latitude, 0.0),
                self._safe_float(b.longitude, 0.0),
                self._safe_float(b.latitude, 0.0),
            ],
        )
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        x = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 6371.0 * 2 * asin(sqrt(x))

    def _hotel_price_range(self, hotel: Hotel, accommodation: str) -> str:
        if self._safe_str(hotel.price_range, ""):
            return hotel.price_range
        estimated = self._safe_int(hotel.estimated_cost, 0)
        if estimated > 0:
            low = max(80, int(estimated * 0.85))
            high = max(low + 50, int(estimated * 1.15))
            return f"¥{low}-¥{high}/晚"
        text = self._safe_str(accommodation, "")
        if "豪华" in text:
            return "¥900-¥1500/晚"
        if "舒适" in text:
            return "¥450-¥900/晚"
        if "经济" in text:
            return "¥220-¥450/晚"
        return "¥350-¥700/晚"

    def _hotel_type(self, hotel: Hotel, accommodation: str) -> str:
        if self._safe_str(hotel.type, ""):
            return hotel.type
        text = self._safe_str(accommodation, "")
        if "豪华" in text:
            return "高档酒店"
        if "舒适" in text:
            return "舒适型酒店"
        if "经济" in text:
            return "经济型酒店"
        return "酒店"

    def _hotel_rating(self, hotel: Hotel) -> str:
        rating = self._safe_str(hotel.rating, "")
        return rating or "4.4"

    def _hotel_distance(self, hotel: Hotel, attractions: List[Attraction]) -> str:
        distance = self._safe_str(hotel.distance, "")
        if distance:
            return distance
        if hotel.location and self._is_valid_location(hotel.location):
            for attraction in attractions:
                if attraction.location and self._is_valid_location(attraction.location):
                    km = self._distance_km(hotel.location, attraction.location)
                    return f"距首个景点约{km:.1f}km"
        return "约3km内"

    def _complete_hotel_fields(
        self,
        hotel: Hotel,
        request: TripRequest,
        attractions: List[Attraction],
    ) -> Hotel:
        estimated = self._safe_int(hotel.estimated_cost, 0)
        if estimated <= 0:
            estimated = self._default_hotel_cost(request.accommodation)

        return Hotel(
            name=self._safe_str(hotel.name, "推荐酒店"),
            address=self._safe_str(hotel.address, ""),
            location=hotel.location if self._is_valid_location(hotel.location) else None,
            price_range=self._hotel_price_range(hotel, request.accommodation),
            rating=self._hotel_rating(hotel),
            distance=self._hotel_distance(hotel, attractions),
            type=self._hotel_type(hotel, request.accommodation),
            estimated_cost=estimated,
        )

    def _meal_focus_text(self, request: TripRequest) -> str:
        candidates: List[str] = []
        for pref in request.preferences or []:
            pref_text = self._safe_str(pref, "")
            if pref_text and self._contains_food_intent(pref_text):
                candidates.append(pref_text)
        free_text = self._safe_str(request.free_text_input, "")
        if free_text and self._contains_food_intent(free_text):
            candidates.append(free_text)
        return candidates[0] if candidates else ""

    def _build_default_meal(
        self,
        meal_type: str,
        request: TripRequest,
        attractions: List[Attraction],
        day_index: int,
    ) -> Meal:
        focus = self._meal_focus_text(request)
        area = attractions[0].name if attractions else request.city

        if meal_type == "breakfast":
            base_name = "广式早茶"
            description = f"建议在{area}附近安排清淡早餐，便于后续游览。"
            cost = 25
        elif meal_type == "lunch":
            base_name = "本地特色午餐"
            description = f"在{area}周边安排午餐，减少往返时间。"
            cost = 45
        else:
            base_name = "本地特色晚餐"
            description = f"晚间可在{area}附近用餐并休整。"
            cost = 60

        if focus:
            if meal_type == "breakfast":
                description = f"可在{area}附近尝试{focus}相关早餐。"
            elif meal_type == "lunch":
                description = f"中午推荐在{area}附近安排{focus}主题用餐。"
            else:
                description = f"晚餐建议安排{focus}，与当天行程衔接。"

        return Meal(
            type=meal_type,
            name=f"第{day_index + 1}天{base_name}",
            description=description,
            estimated_cost=cost,
        )

    def _ensure_day_meals(
        self,
        meals: List[Meal],
        request: TripRequest,
        attractions: List[Attraction],
        day_index: int,
    ) -> List[Meal]:
        required_types = ["breakfast", "lunch", "dinner"]
        by_type: Dict[str, Meal] = {}
        extra_snacks: List[Meal] = []

        for meal in meals:
            meal_type = self._safe_str(getattr(meal, "type", ""), "").lower()
            if meal_type not in {"breakfast", "lunch", "dinner", "snack"}:
                meal_type = "snack"
            fixed = Meal(
                type=meal_type,
                name=self._safe_str(getattr(meal, "name", ""), "餐食推荐"),
                address=getattr(meal, "address", None),
                location=getattr(meal, "location", None),
                description=self._safe_str(getattr(meal, "description", ""), "") or None,
                estimated_cost=self._safe_int(getattr(meal, "estimated_cost", 0), 0),
            )
            if meal_type in required_types and meal_type not in by_type:
                by_type[meal_type] = fixed
            elif meal_type == "snack":
                extra_snacks.append(fixed)

        ensured: List[Meal] = []
        for meal_type in required_types:
            meal = by_type.get(meal_type)
            if meal is None or not self._safe_str(meal.name, ""):
                meal = self._build_default_meal(meal_type, request, attractions, day_index)
            ensured.append(meal)

        ensured.extend(extra_snacks[:1])
        return ensured

    def _default_hotel_cost(self, accommodation: str) -> int:
        text = self._safe_str(accommodation, "")
        if "豪华" in text:
            return 900
        if "舒适" in text:
            return 500
        if "经济" in text:
            return 280
        return 450

    def _default_transport_cost(self, transportation: str) -> int:
        text = self._safe_str(transportation, "")
        if "打车" in text or "出租" in text or "网约车" in text:
            return 120
        if "自驾" in text or "租车" in text:
            return 220
        if "地铁" in text or "公交" in text:
            return 35
        return 60

    def _estimate_budget_from_plan(self, trip_plan: TripPlan, request: TripRequest) -> Budget:
        total_attractions = 0
        attraction_count = 0
        total_hotels = 0
        total_meals = 0

        for day in trip_plan.days:
            for attr in day.attractions:
                attraction_count += 1
                total_attractions += max(0, self._safe_int(attr.ticket_price, 0))
            if day.hotel is not None:
                total_hotels += max(0, self._safe_int(day.hotel.estimated_cost, 0))
            for meal in day.meals:
                total_meals += max(0, self._safe_int(meal.estimated_cost, 0))

        if total_attractions <= 0 and attraction_count > 0:
            total_attractions = attraction_count * 40

        if total_hotels <= 0:
            total_hotels = max(1, request.travel_days) * self._default_hotel_cost(request.accommodation)

        if total_meals <= 0:
            total_meals = max(1, request.travel_days) * 3 * 45

        total_transportation = max(1, request.travel_days) * self._default_transport_cost(request.transportation)
        total = total_attractions + total_hotels + total_meals + total_transportation

        return Budget(
            total_attractions=total_attractions,
            total_hotels=total_hotels,
            total_meals=total_meals,
            total_transportation=total_transportation,
            total=total,
        )

    # ---------------------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------------------
    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        try:
            start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        except ValueError:
            start_date = datetime.now()

        days: List[DayPlan] = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            base_lng = 113.264385 + i * 0.01
            base_lat = 23.129112 + i * 0.01
            attractions = [
                Attraction(
                    name=f"{request.city}景点{j + 1}",
                    address=f"{request.city}市",
                    location=Location(longitude=base_lng + j * 0.005, latitude=base_lat + j * 0.005),
                    visit_duration=120,
                    description=f"{request.city}推荐景点",
                    category="景点",
                    ticket_price=0,
                )
                for j in range(2)
            ]
            meals = [
                Meal(type="breakfast", name=f"第{i + 1}天早餐", description="本地早餐推荐"),
                Meal(type="lunch", name=f"第{i + 1}天午餐", description="本地午餐推荐"),
                Meal(type="dinner", name=f"第{i + 1}天晚餐", description="本地晚餐推荐"),
            ]
            days.append(
                DayPlan(
                    date=current_date.strftime("%Y-%m-%d"),
                    day_index=i,
                    description=f"第{i + 1}天行程",
                    transportation=request.transportation,
                    accommodation=request.accommodation,
                    attractions=attractions,
                    meals=meals,
                )
            )

        fallback_plan = TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"当前返回备用行程，请核验{request.city}实时开放时间和天气。",
            budget=None,
        )
        fallback_plan.budget = self._estimate_budget_from_plan(fallback_plan, request)
        return fallback_plan

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def plan_trip(self, request: TripRequest) -> TripPlan:
        started = time.perf_counter()
        logger.info("\n%s", "=" * 60)
        logger.info(
            "workflow_start city=%s start=%s end=%s days=%d",
            request.city,
            request.start_date,
            request.end_date,
            request.travel_days,
        )
        logger.info("%s\n", "=" * 60)

        initial_state: TripPlannerState = create_initial_state(request)
        final_state = self.graph.invoke(initial_state)

        if final_state.get("error") and not final_state.get("trip_plan"):
            error_msg = self._safe_str(final_state.get("error"), "unknown error")
            logger.error("Trip planning failed: %s", error_msg)
            raise Exception(error_msg)

        if not final_state.get("trip_plan"):
            logger.warning("No trip_plan returned, using fallback plan")
            return self._create_fallback_plan(request)

        alerts = final_state.get("abnormal_alerts") or []
        trip_plan = final_state["trip_plan"]
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if alerts:
            logger.warning("Abnormal alerts in final plan: %d", len(alerts))

        logger.info("\n%s", "=" * 60)
        logger.info(
            "workflow_done city=%s days=%d alerts=%d elapsed_ms=%d",
            request.city,
            len(trip_plan.days) if trip_plan else 0,
            len(alerts),
            elapsed_ms,
        )
        logger.info("%s\n", "=" * 60)
        return trip_plan


_trip_planner_workflow: Optional[TripPlannerWorkflow] = None


def get_trip_planner_workflow() -> TripPlannerWorkflow:
    """Get singleton workflow instance."""
    global _trip_planner_workflow
    if _trip_planner_workflow is None:
        _trip_planner_workflow = TripPlannerWorkflow()
    return _trip_planner_workflow


def reset_workflow():
    """Reset singleton workflow instance."""
    global _trip_planner_workflow
    _trip_planner_workflow = None
    logger.info("Workflow singleton reset")
