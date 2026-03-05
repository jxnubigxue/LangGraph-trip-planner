"""高德地图MCP服务封装 (LangChain 版本)"""

from typing import List, Dict, Any, Optional, Union
import logging
from langchain_core.tools import BaseTool
from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo
from ..tools.amap_mcp_tools import get_cached_amap_tools

logger = logging.getLogger(__name__)

# 全局工具缓存
_amap_tools_cache: Optional[List[BaseTool]] = None
_tool_map: Dict[str, BaseTool] = {}


def _get_tool_by_name(tool_name: str) -> Optional[BaseTool]:
    """根据工具名称获取工具实例"""
    global _tool_map

    if not _tool_map:
        tools = get_cached_amap_tools()
        for tool in tools:
            _tool_map[tool.name] = tool

    return _tool_map.get(tool_name)


def _execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Union[str, Dict, List]:
    """
    执行指定工具

    Args:
        tool_name: 工具名称
        arguments: 工具参数

    Returns:
        工具执行结果
    """
    tool = _get_tool_by_name(tool_name)
    if not tool:
        raise ValueError(f"未找到工具: {tool_name}")

    try:
        # 使用 LangChain 工具调用方式
        result = tool.invoke(arguments)
        logger.debug(f"工具 {tool_name} 执行成功，参数: {arguments}")
        return result
    except Exception as e:
        logger.error(f"工具 {tool_name} 执行失败: {str(e)}", exc_info=True)
        raise


class AmapService:
    """高德地图服务封装类 (LangChain 版本)"""

    def __init__(self):
        """初始化服务"""
        # 预加载工具
        self.tools = get_cached_amap_tools()
        logger.info(f"高德地图服务初始化完成，加载了 {len(self.tools)} 个工具")

    def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """
        搜索POI

        Args:
            keywords: 搜索关键词
            city: 城市
            citylimit: 是否限制在城市范围内

        Returns:
            POI信息列表
        """
        try:
            # 调用工具
            result = _execute_tool("maps_text_search", {
                "keywords": keywords,
                "city": city,
                "citylimit": str(citylimit).lower()
            })

            logger.info(f"POI搜索成功: {keywords} in {city}")

            # 解析结果
            # 注意: MCP工具返回的是字符串,需要解析
            # 这里简化处理,实际应该解析JSON
            if isinstance(result, str):
                logger.debug(f"POI搜索结果 (前200字符): {result[:200]}...")
            else:
                logger.debug(f"POI搜索结果类型: {type(result)}")

            # TODO: 解析实际的POI数据
            return []

        except Exception as e:
            logger.error(f"❌ POI搜索失败: {str(e)}", exc_info=True)
            return []

    def get_weather(self, city: str) -> List[WeatherInfo]:
        """
        查询天气

        Args:
            city: 城市名称

        Returns:
            天气信息列表
        """
        try:
            result = _execute_tool("maps_weather", {"city": city})

            logger.info(f"天气查询成功: {city}")

            if isinstance(result, str):
                logger.debug(f"天气查询结果 (前200字符): {result[:200]}...")
            else:
                logger.debug(f"天气查询结果类型: {type(result)}")

            # TODO: 解析实际的天气数据
            return []

        except Exception as e:
            logger.error(f"❌ 天气查询失败: {str(e)}", exc_info=True)
            return []

    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking"
    ) -> Dict[str, Any]:
        """
        规划路线

        Args:
            origin_address: 起点地址
            destination_address: 终点地址
            origin_city: 起点城市
            destination_city: 终点城市
            route_type: 路线类型 (walking/driving/transit)

        Returns:
            路线信息
        """
        try:
            # 根据路线类型选择工具
            tool_map = {
                "walking": "maps_direction_walking_by_address",
                "driving": "maps_direction_driving_by_address",
                "transit": "maps_direction_transit_integrated_by_address"
            }

            tool_name = tool_map.get(route_type, "maps_direction_walking_by_address")

            # 构建参数
            arguments = {
                "origin_address": origin_address,
                "destination_address": destination_address
            }

            # 公共交通需要城市参数
            if route_type == "transit":
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city
            else:
                # 其他路线类型也可以提供城市参数提高准确性
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city

            result = _execute_tool(tool_name, arguments)

            logger.info(f"路线规划成功: {origin_address} -> {destination_address} ({route_type})")

            if isinstance(result, str):
                logger.debug(f"路线规划结果 (前200字符): {result[:200]}...")

            # TODO: 解析实际的路线数据
            return {}

        except Exception as e:
            logger.error(f"❌ 路线规划失败: {str(e)}", exc_info=True)
            return {}

    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """
        地理编码(地址转坐标)

        Args:
            address: 地址
            city: 城市

        Returns:
            经纬度坐标
        """
        try:
            arguments = {"address": address}
            if city:
                arguments["city"] = city

            result = _execute_tool("maps_geo", arguments)

            logger.info(f"地理编码成功: {address} in {city or '未知城市'}")

            if isinstance(result, str):
                logger.debug(f"地理编码结果 (前200字符): {result[:200]}...")

            # TODO: 解析实际的坐标数据
            return None

        except Exception as e:
            logger.error(f"❌ 地理编码失败: {str(e)}", exc_info=True)
            return None

    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """
        获取POI详情

        Args:
            poi_id: POI ID

        Returns:
            POI详情信息
        """
        try:
            result = _execute_tool("maps_search_detail", {"id": poi_id})

            logger.info(f"获取POI详情成功: {poi_id}")

            # 解析结果并提取图片
            import json
            import re

            if isinstance(result, str):
                logger.debug(f"POI详情结果 (前200字符): {result[:200]}...")

                # 尝试从结果中提取JSON
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        return data
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析JSON: {json_match.group()[:100]}...")

                return {"raw": result}
            else:
                # 如果工具直接返回字典
                return result if isinstance(result, dict) else {"result": result}

        except Exception as e:
            logger.error(f"❌ 获取POI详情失败: {str(e)}", exc_info=True)
            return {}


# 创建全局服务实例
_amap_service = None


def get_amap_service() -> AmapService:
    """获取高德地图服务实例(单例模式)"""
    global _amap_service
    
    if _amap_service is None:
        _amap_service = AmapService()
    
    return _amap_service

