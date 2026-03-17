"""旅行规划API路由 (LangGraph 版本)"""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException
from ...logging_config import logging_context
from ...models.schemas import (
    TripPlanResponse,
    TripRequest,
)
# 从新的工作流导入
from ...workflows.trip_planner_graph import get_trip_planner_workflow

router = APIRouter(prefix="/trip", tags=["旅行规划"])
logger = logging.getLogger(__name__)


@router.post(
    "/plan",
    response_model=TripPlanResponse,
    summary="生成旅行计划",
    description="根据用户输入的旅行需求,生成详细的旅行计划"
)
async def plan_trip(request: TripRequest):
    """
    生成旅行计划 (LangGraph 版本)

    Args:
        request: 旅行请求参数

    Returns:
        旅行计划响应
    """
    run_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        with logging_context(run_id=run_id):
            logger.info(
                "trip_request_start city=%s start=%s end=%s days=%d",
                request.city,
                request.start_date,
                request.end_date,
                request.travel_days,
            )

            # 获取工作流实例
            workflow = get_trip_planner_workflow()

            # 执行工作流
            trip_plan = workflow.plan_trip(request)

            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "trip_request_success city=%s days=%d elapsed_ms=%d",
                request.city,
                len(trip_plan.days) if trip_plan else 0,
                elapsed_ms,
            )

            return TripPlanResponse(
                success=True,
                message="旅行计划生成成功 (LangGraph)",
                data=trip_plan
            )

    except Exception as exc:
        with logging_context(run_id=run_id):
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.error("trip_request_failed elapsed_ms=%d error=%s", elapsed_ms, str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"生成旅行计划失败: {str(exc)}"
        )


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    """健康检查"""
    try:
        workflow = get_trip_planner_workflow()

        return {
            "status": "healthy",
            "service": "trip-planner-langgraph",
            "framework": "LangGraph",
            "graph_compiled": True,
            "tools_loaded": len(workflow.tools) if hasattr(workflow, 'tools') else 0
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"服务不可用: {str(e)}"
        )

