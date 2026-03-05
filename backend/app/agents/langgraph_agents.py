"""LangChain/LangGraph 智能体定义"""

from typing import Dict, Any, List, Optional
import logging
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from ..services.llm_service import get_llm

# 设置日志记录
logger = logging.getLogger(__name__)

# ============ 智能体提示词 ============

ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。

**重要提示:**
你必须使用工具来搜索景点!不要自己编造景点信息!

**可用工具:**
{tools}

**工具调用要求:**
1. 根据用户查询选择合适的工具
2. 提供完整的参数
3. 不要在没有工具的情况下直接回答
4. 工具调用完成后，请直接返回工具返回的原始JSON数据，不要进行总结或格式化
5. 返回的JSON应该是一个景点列表，每个景点包含name、address、location、visit_duration、description、category、ticket_price等字段
"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。你的任务是查询指定城市的天气信息。

**重要提示:**
你必须使用工具来查询天气!不要自己编造天气信息!

**可用工具:**
{tools}

**工具调用要求:**
1. 根据用户查询选择合适的工具
2. 提供完整的参数
3. 不要在没有工具的情况下直接回答
4. 工具调用完成后，请直接返回工具返回的原始JSON数据，不要进行总结或格式化
5. 返回的JSON应该是一个天气信息列表，每个天气信息包含date、day_weather、night_weather、day_temp、night_temp、wind_direction、wind_power等字段
"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。你的任务是根据城市和景点位置推荐合适的酒店。

**重要提示:**
你必须使用工具来搜索酒店!不要自己编造酒店信息!

**可用工具:**
{tools}

**工具调用要求:**
1. 根据用户查询选择合适的工具
2. 提供完整的参数
3. 不要在没有工具的情况下直接回答
4. 关键词使用"酒店"或"宾馆"
5. 工具调用完成后，请直接返回工具返回的原始JSON数据，不要进行总结或格式化
6. 返回的JSON应该是一个酒店列表，每个酒店包含name、address、location、price_range、rating、distance、type、estimated_cost等字段
"""

PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划:
```json
{{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {{
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {{"longitude": 116.397128, "latitude": 39.916527}},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离景点2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      }},
      "attractions": [
        {{
          "name": "景点名称",
          "address": "详细地址",
          "location": {{"longitude": 116.397128, "latitude": 39.916527}},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30}},
        {{"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50}},
        {{"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}}
      ]
    }}
  ],
  "weather_info": [
    {{
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }}
  ],
  "overall_suggestions": "总体建议",
  "budget": {{
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }}
}}
```

**重要提示:**
1. weather_info数组必须包含每一天的天气信息
2. 温度必须是纯数字(不要带°C等单位)
3. 每天安排2-3个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. **必须包含预算信息**:
   - 景点门票价格(ticket_price)
   - 餐饮预估费用(estimated_cost)
   - 酒店预估费用(estimated_cost)
   - 预算汇总(budget)包含各项总费用
"""


def create_attraction_search_agent(tools: List[BaseTool]):
    """创建景点搜索智能体"""
    print("   创建景点搜索智能体...")
    try:
        llm = get_llm()

        # 格式化系统提示词，包含工具信息
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        system_prompt = ATTRACTION_AGENT_PROMPT.format(tools=tools_description)

        # 创建智能体图
        agent_graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=False  # 设置为 True 可启用详细日志
        )

        print(f"   [OK] 景点搜索智能体创建成功，工具数量: {len(tools)}")
        logger.info("[OK] 景点搜索智能体创建成功")
        return agent_graph

    except Exception as e:
        print(f"   [ERROR] 创建景点搜索智能体失败: {str(e)}")
        logger.error(f"[ERROR] 创建景点搜索智能体失败: {str(e)}", exc_info=True)
        raise


def create_weather_agent(tools: List[BaseTool]):
    """创建天气查询智能体"""
    print("   创建天气查询智能体...")
    try:
        llm = get_llm()

        # 格式化系统提示词，包含工具信息
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        system_prompt = WEATHER_AGENT_PROMPT.format(tools=tools_description)

        # 创建智能体图
        agent_graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=False  # 设置为 True 可启用详细日志
        )

        print(f"   [OK] 天气查询智能体创建成功，工具数量: {len(tools)}")
        logger.info("[OK] 天气查询智能体创建成功")
        return agent_graph

    except Exception as e:
        print(f"   [ERROR] 创建天气查询智能体失败: {str(e)}")
        logger.error(f"[ERROR] 创建天气查询智能体失败: {str(e)}", exc_info=True)
        raise


def create_hotel_agent(tools: List[BaseTool]):
    """创建酒店推荐智能体"""
    try:
        llm = get_llm()

        # 格式化系统提示词，包含工具信息
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        system_prompt = HOTEL_AGENT_PROMPT.format(tools=tools_description)

        # 创建智能体图
        agent_graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=False  # 设置为 True 可启用详细日志
        )

        logger.info("[OK] 酒店推荐智能体创建成功")
        return agent_graph

    except Exception as e:
        logger.error(f"[ERROR] 创建酒店推荐智能体失败: {str(e)}", exc_info=True)
        raise


def create_planner_agent(tools: List[BaseTool]):
    """创建行程规划智能体"""
    try:
        llm = get_llm()

        # 格式化系统提示词，包含工具信息（如果提示词中包含 {tools}）
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        system_prompt = PLANNER_AGENT_PROMPT
        if "{tools}" in system_prompt:
            system_prompt = system_prompt.format(tools=tools_description)

        # 创建智能体图
        agent_graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=False  # 设置为 True 可启用详细日志
        )

        logger.info("[OK] 行程规划智能体创建成功")
        return agent_graph

    except Exception as e:
        logger.error(f"[ERROR] 创建行程规划智能体失败: {str(e)}", exc_info=True)
        raise


# 智能体实例缓存
_agent_instances: Dict[str, Any] = {}


def get_agent(agent_type: str, tools: List[BaseTool]) -> Any:
    """获取智能体实例（带缓存）"""
    cache_key = f"{agent_type}_{len(tools)}"

    if cache_key not in _agent_instances:
        logger.info(f"创建 {agent_type} 智能体...")

        if agent_type == "attraction_search":
            agent = create_attraction_search_agent(tools)
        elif agent_type == "weather":
            agent = create_weather_agent(tools)
        elif agent_type == "hotel":
            agent = create_hotel_agent(tools)
        elif agent_type == "planner":
            agent = create_planner_agent(tools)
        else:
            raise ValueError(f"未知的智能体类型: {agent_type}")

        _agent_instances[cache_key] = agent

    return _agent_instances[cache_key]


def clear_agent_cache():
    """清空智能体缓存"""
    global _agent_instances
    _agent_instances.clear()
    logger.info("智能体缓存已清空")