"""高德地图MCP工具 (LangChain MCP适配器版本)"""

from typing import List, Dict, Any, Optional, Type
import asyncio
import logging
import nest_asyncio
import inspect
import json
try:
    from langchain_mcp_adapters.tools import load_mcp_tools
    MCP_ADAPTERS_AVAILABLE = True
except ImportError:
    MCP_ADAPTERS_AVAILABLE = False
    load_mcp_tools = None
from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field
from ..config import get_settings

# 设置日志记录
logger = logging.getLogger(__name__)


def wrap_async_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """包装异步工具以支持同步调用

    某些 MCP 工具可能只实现了异步方法 (_arun)，
    但 LangGraph 工具节点需要同步调用。
    此函数检查工具是否有 _arun 方法但没有 _run 方法，
    并创建一个支持同步调用的包装器。
    """
    wrapped_tools = []

    for tool in tools:
        # 检查工具是否已经是 StructuredTool 且有 _arun 但没有 _run
        has_arun = hasattr(tool, '_arun') and callable(tool._arun)
        has_run = hasattr(tool, '_run') and callable(tool._run)

        # 检查是否是 StructuredTool 实例（即使有 _run 方法，也可能抛出 NotImplementedError）
        is_structured_tool = isinstance(tool, StructuredTool)

        # 需要包装的情况：
        # 1. 有 _arun 但没有 _run
        # 2. 是 StructuredTool 且有 _arun（因为 StructuredTool._run 会抛出 NotImplementedError）
        if (has_arun and not has_run) or (is_structured_tool and has_arun):
            logger.debug(f"包装异步工具: {tool.name} (类型: {type(tool).__name__})")

            # 创建一个新类，继承自原始工具类
            class SyncWrapper(tool.__class__):
                def _run(self, *args, **kwargs):
                    """同步运行方法，内部调用异步方法"""
                    import asyncio
                    # 确保 kwargs 中有 config 参数
                    if 'config' not in kwargs:
                        kwargs['config'] = None
                    try:
                        # 使用 nest_asyncio 允许在已有事件循环中运行
                        nest_asyncio.apply()
                        return asyncio.run(self._arun(*args, **kwargs))
                    except RuntimeError as e:
                        if "cannot be called from a running event loop" in str(e):
                            # 如果已经有运行中的事件循环，尝试使用当前循环
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # 在已有循环中运行
                                future = asyncio.run_coroutine_threadsafe(
                                    self._arun(*args, **kwargs), loop
                                )
                                return future.result()
                        raise

            # 创建包装器实例，复制所有属性
            wrapper = SyncWrapper(
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema if hasattr(tool, 'args_schema') else None,
                return_direct=tool.return_direct if hasattr(tool, 'return_direct') else False,
                verbose=tool.verbose if hasattr(tool, 'verbose') else False,
                callbacks=tool.callbacks if hasattr(tool, 'callbacks') else None,
                tags=tool.tags if hasattr(tool, 'tags') else None,
                metadata=tool.metadata if hasattr(tool, 'metadata') else None,
            )

            # 复制其他可能需要的属性
            for attr in ['func', 'coroutine']:
                if hasattr(tool, attr):
                    try:
                        setattr(wrapper, attr, getattr(tool, attr))
                    except AttributeError:
                        # 某些属性可能是只读的，跳过
                        pass

            wrapped_tools.append(wrapper)
        else:
            # 工具已经支持同步调用，直接使用
            wrapped_tools.append(tool)

    return wrapped_tools


async def create_amap_mcp_tools() -> List[BaseTool]:
    """创建高德地图MCP工具列表"""
    settings = get_settings()

    # 验证必要的配置
    if not settings.amap_api_key:
        logger.error("AMAP_API_KEY 未配置")
        return []

    # 如果 MCP 适配器不可用，则返回模拟工具
    if not MCP_ADAPTERS_AVAILABLE:
        logger.info("MCP适配器不可用，返回模拟工具")
        return create_mock_tools()

    try:

        # 创建连接配置
        connection = {
            "command": "uvx",
            "args": ["amap-mcp-server"],
            "transport": "stdio",
            "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key}
        }

        logger.info("正在连接高德地图MCP服务器...")

        # 使用 load_mcp_tools 直接加载工具
        tools = await load_mcp_tools(
            session=None,
            connection=connection,
            server_name="amap",
            tool_name_prefix=False
        )

        logger.info(f"✅ 从MCP服务器加载了 {len(tools)} 个工具")

        # 为工具添加自定义描述，增强可读性
        tool_descriptions = {
            "maps_text_search": "搜索高德地图的POI（兴趣点）信息，如景点、餐厅、酒店等",
            "maps_weather": "查询指定城市的天气信息，包括温度、天气状况、风力等",
            "maps_geocode": "地址编码，将地址转换为经纬度坐标",
            "maps_reverse_geocode": "逆地址编码，将经纬度坐标转换为地址",
            "maps_route_planning": "路线规划，提供驾车、步行、公交等出行方式的路线规划"
        }

        for tool in tools:
            tool_name = tool.name.lower()
            for key, description in tool_descriptions.items():
                if key in tool_name:
                    tool.description = description
                    break

        # 包装异步工具以支持同步调用
        tools = wrap_async_tools(tools)
        logger.info(f"包装后工具数量: {len(tools)}")

        return tools

    except Exception as e:
        logger.error(f"❌ 加载MCP工具失败: {str(e)}", exc_info=True)
        # 返回空列表，调用方应处理空工具情况
        return []


def get_amap_mcp_tools() -> List[BaseTool]:
    """同步获取MCP工具"""
    try:
        # 应用 nest_asyncio 以允许在已有事件循环中运行
        nest_asyncio.apply()
        return asyncio.run(create_amap_mcp_tools())
    except Exception as e:
        logger.error(f"❌ 同步获取MCP工具失败: {str(e)}", exc_info=True)
        return []


def get_amap_essential_tools() -> List[BaseTool]:
    """获取主要的高德地图工具（手动配置备用方案）"""
    settings = get_settings()

    if not settings.amap_api_key:
        logger.error("AMAP_API_KEY 未配置")
        return []

    try:
        # 使用异步函数加载工具，然后过滤出主要工具
        async def load_and_filter():
            connection = {
                "command": "uvx",
                "args": ["amap-mcp-server"],
                "transport": "stdio",
                "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key}
            }

            tools = await load_mcp_tools(
                session=None,
                connection=connection,
                server_name="amap",
                tool_name_prefix=False
            )

            # 过滤出主要工具
            essential_tool_names = {"maps_text_search", "maps_weather"}
            filtered_tools = []
            for tool in tools:
                tool_name = tool.name.lower()
                for essential_name in essential_tool_names:
                    if essential_name in tool_name:
                        # 添加描述
                        if "maps_text_search" in tool_name:
                            tool.description = "搜索高德地图的POI（兴趣点）信息，如景点、餐厅、酒店等"
                        elif "maps_weather" in tool_name:
                            tool.description = "查询指定城市的天气信息，包括温度、天气状况、风力等"
                        filtered_tools.append(tool)
                        break

            return filtered_tools

        nest_asyncio.apply()
        tools = asyncio.run(load_and_filter())
        logger.info(f"✅ 加载了 {len(tools)} 个主要高德地图工具")

        # 包装异步工具以支持同步调用
        tools = wrap_async_tools(tools)
        logger.info(f"包装后工具数量: {len(tools)}")

        return tools

    except Exception as e:
        logger.error(f"❌ 加载主要工具失败: {str(e)}", exc_info=True)
        return []


# 全局工具缓存
_cached_tools: Optional[List[BaseTool]] = None


def get_cached_amap_tools() -> List[BaseTool]:
    """获取缓存的高德地图工具（避免重复创建）"""
    global _cached_tools

    if _cached_tools is None:
        logger.info("首次加载高德地图工具，建立缓存...")
        tools = get_amap_mcp_tools()

        # 如果自动加载失败，使用主要工具备用
        if not tools:
            logger.warning("自动加载工具失败，尝试使用主要工具...")
            tools = get_amap_essential_tools()

        # 如果主要工具也失败，使用模拟工具
        if not tools:
            logger.warning("所有真实工具加载失败，使用模拟工具...")
            tools = create_mock_tools()

        _cached_tools = tools

    return _cached_tools


def clear_tools_cache():
    """清空工具缓存（用于测试或重新加载）"""
    global _cached_tools
    _cached_tools = None
    logger.info("工具缓存已清空")


# ============ 模拟工具（当MCP服务器不可用时）============

class SearchInput(BaseModel):
    """景点搜索输入参数"""
    query: str = Field(description="搜索查询，如'北京景点'")
    city: str = Field(description="城市名称")


class WeatherInput(BaseModel):
    """天气查询输入参数"""
    city: str = Field(description="城市名称")


def create_mock_tools() -> List[BaseTool]:
    """创建模拟工具用于测试和开发"""
    logger.info("创建模拟工具...")

    @tool("maps_text_search", args_schema=SearchInput)
    def mock_search_tool(query: str, city: str) -> str:
        """模拟景点搜索工具"""
        logger.info(f"模拟搜索: {query} in {city}")

        # 返回模拟的景点数据
        mock_results = [
            {
                "name": f"{city}著名景点1",
                "address": f"{city}市某区某路1号",
                "location": {"longitude": 116.397128, "latitude": 39.916527},
                "visit_duration": 120,
                "description": f"这是{city}的著名景点，历史悠久，值得一游",
                "category": "历史文化",
                "ticket_price": 60
            },
            {
                "name": f"{city}著名景点2",
                "address": f"{city}市某区某路2号",
                "location": {"longitude": 116.407128, "latitude": 39.926527},
                "visit_duration": 90,
                "description": f"这是{city}的另一个著名景点，风景优美",
                "category": "公园",
                "ticket_price": 40
            }
        ]

        return json.dumps(mock_results, ensure_ascii=False)

    @tool("maps_weather", args_schema=WeatherInput)
    def mock_weather_tool(city: str) -> str:
        """模拟天气查询工具"""
        logger.info(f"模拟天气查询: {city}")

        # 返回模拟的天气数据
        mock_weather = [
            {
                "date": "2024-10-01",
                "day_weather": "晴",
                "night_weather": "多云",
                "day_temp": 25,
                "night_temp": 15,
                "wind_direction": "南风",
                "wind_power": "1-3级"
            },
            {
                "date": "2024-10-02",
                "day_weather": "多云",
                "night_weather": "阴",
                "day_temp": 23,
                "night_temp": 16,
                "wind_direction": "北风",
                "wind_power": "2-4级"
            },
            {
                "date": "2024-10-03",
                "day_weather": "小雨",
                "night_weather": "阴",
                "day_temp": 20,
                "night_temp": 14,
                "wind_direction": "东风",
                "wind_power": "1-2级"
            }
        ]

        return json.dumps(mock_weather, ensure_ascii=False)

    @tool("maps_hotel_search")
    def mock_hotel_tool(query: str) -> str:
        """模拟酒店搜索工具"""
        logger.info(f"模拟酒店搜索: {query}")

        # 返回模拟的酒店数据
        mock_hotels = [
            {
                "name": f"{query.split()[0]}经济型酒店1",
                "address": f"{query.split()[0]}市某区某路10号",
                "location": {"longitude": 116.387128, "latitude": 39.906527},
                "price_range": "200-400元",
                "rating": "4.2",
                "distance": "距离市中心2公里",
                "type": "经济型酒店",
                "estimated_cost": 300
            },
            {
                "name": f"{query.split()[0]}经济型酒店2",
                "address": f"{query.split()[0]}市某区某路20号",
                "location": {"longitude": 116.417128, "latitude": 39.936527},
                "price_range": "150-350元",
                "rating": "4.0",
                "distance": "距离景点1公里",
                "type": "经济型酒店",
                "estimated_cost": 250
            }
        ]

        return json.dumps(mock_hotels, ensure_ascii=False)

    # 创建工具列表
    tools = [
        mock_search_tool,
        mock_weather_tool,
        mock_hotel_tool
    ]

    logger.info(f"✅ 创建了 {len(tools)} 个模拟工具")
    return tools