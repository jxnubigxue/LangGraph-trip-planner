"""Unsplash图片服务"""

import requests
from typing import List, Optional
from ..config import get_settings

class UnsplashService:
    """Unsplash图片服务类"""
    
    def __init__(self):
        """初始化服务"""
        settings = get_settings()
        self.access_key = settings.unsplash_access_key
        self.base_url = "https://api.unsplash.com"
    
    def search_photos(self, query: str, per_page: int = 5) -> List[dict]:
        """
        搜索图片

        Args:
            query: 搜索关键词
            per_page: 每页数量

        Returns:
            图片列表
        """
        try:
            # 清理查询字符串 - 修复全角括号导致的410错误
            query = self._clean_query(query)

            url = f"{self.base_url}/search/photos"
            params = {
                "query": query,
                "per_page": per_page,
                "client_id": self.access_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # 提取图片URL
            photos = []
            for photo in results:
                photos.append({
                    "id": photo.get("id"),
                    "url": photo.get("urls", {}).get("regular"),
                    "thumb": photo.get("urls", {}).get("thumb"),
                    "description": photo.get("description") or photo.get("alt_description"),
                    "photographer": photo.get("user", {}).get("name")
                })
            
            return photos

        except Exception as e:
            print(f"❌ Unsplash搜索失败: {str(e)}")
            return []

    def _clean_query(self, query: str) -> str:
        """清理查询字符串，避免Unsplash API的410错误"""
        import re

        # 主要问题：替换全角括号为半角括号
        query = query.replace('（', '(').replace('）', ')')

        # 可选：替换其他常见中文标点，但可能不是必需的
        # 只处理已知可能引起问题的字符
        replacements = {
            '，': ',',  # 中文逗号 -> 英文逗号
            '；': ';',  # 中文分号 -> 英文分号
            '：': ':',  # 中文冒号 -> 英文冒号
        }

        for old, new in replacements.items():
            query = query.replace(old, new)

        # 压缩多个空格为单个空格（避免URL编码问题）
        query = re.sub(r'\s+', ' ', query).strip()

        return query

    def get_photo_url(self, query: str) -> Optional[str]:
        """
        获取单张图片URL

        Args:
            query: 搜索关键词

        Returns:
            图片URL
        """
        photos = self.search_photos(query, per_page=1)
        if photos:
            return photos[0].get("url")
        return None


# 全局服务实例
_unsplash_service = None


def get_unsplash_service() -> UnsplashService:
    """获取Unsplash服务实例(单例模式)"""
    global _unsplash_service
    
    if _unsplash_service is None:
        _unsplash_service = UnsplashService()
    
    return _unsplash_service

