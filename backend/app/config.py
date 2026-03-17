"""Configuration management for the backend service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings

# Load local .env first.
load_dotenv()

# Then optionally load sibling HelloAgents/.env without overriding existing vars.
helloagents_env = Path(__file__).parent.parent.parent.parent / "HelloAgents" / ".env"
if helloagents_env.exists():
    load_dotenv(helloagents_env, override=False)


class Settings(BaseSettings):
    """Application settings."""

    # App
    app_name: str = "多agent的智能旅行助手"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS (comma separated)
    cors_origins: str = (
        "http://localhost:5173,http://localhost:3000,"
        "http://127.0.0.1:5173,http://127.0.0.1:3000"
    )

    # AMap
    amap_api_key: str = ""

    # Unsplash
    unsplash_access_key: str = ""
    unsplash_secret_key: str = ""

    # LLM defaults (env variables can override these)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4"

    # LangChain / LangSmith
    langchain_tracing: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "trip-planner"

    # Agent behavior
    agent_max_iterations: int = 3
    agent_temperature: float = 0.7
    agent_timeout: float = 30.0

    # Logging
    log_level: str = "INFO"

    @field_validator("debug", mode="before")
    @classmethod
    def _coerce_debug(cls, value: Any) -> bool:
        """Accept non-bool DEBUG values like 'release' from parent env."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production", ""}:
                return False
        return bool(value)

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

    def get_cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()


def get_settings() -> Settings:
    return settings


def _first_non_empty(*values: str | None) -> str:
    """Return first non-empty string; otherwise empty string."""
    for value in values:
        if value is None:
            continue
        stripped = value.strip()
        if stripped:
            return stripped
    return ""


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_effective_llm_config() -> dict[str, Any]:
    """Resolve effective LLM config with backward-compatible aliases.

    Supported aliases:
    - OPENAI_API_KEY or LLM_API_KEY
    - OPENAI_BASE_URL or LLM_BASE_URL
    - OPENAI_MODEL or LLM_MODEL_ID
    - OPENAI_TIMEOUT or LLM_TIMEOUT
    - OPENAI_MAX_RETRIES or LLM_MAX_RETRIES
    - AGENT_TEMPERATURE or LLM_TEMPERATURE
    """

    api_key = _first_non_empty(
        os.getenv("OPENAI_API_KEY"),
        os.getenv("LLM_API_KEY"),
        settings.openai_api_key,
    )
    base_url = _first_non_empty(
        os.getenv("OPENAI_BASE_URL"),
        os.getenv("LLM_BASE_URL"),
        settings.openai_base_url,
    ) or "https://api.openai.com/v1"
    model = _first_non_empty(
        os.getenv("OPENAI_MODEL"),
        os.getenv("LLM_MODEL_ID"),
        settings.openai_model,
    ) or "gpt-4"

    timeout_raw = _first_non_empty(
        os.getenv("OPENAI_TIMEOUT"),
        os.getenv("LLM_TIMEOUT"),
    )
    timeout = _to_float(timeout_raw, 60.0) if timeout_raw else 60.0

    max_retries_raw = _first_non_empty(
        os.getenv("OPENAI_MAX_RETRIES"),
        os.getenv("LLM_MAX_RETRIES"),
    )
    max_retries = _to_int(max_retries_raw, 0) if max_retries_raw else 0

    temperature_raw = _first_non_empty(
        os.getenv("AGENT_TEMPERATURE"),
        os.getenv("LLM_TEMPERATURE"),
    )
    temperature = _to_float(temperature_raw, settings.agent_temperature) if temperature_raw else float(
        settings.agent_temperature
    )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout": timeout,
        "max_retries": max_retries,
        "temperature": temperature,
    }


def validate_config() -> bool:
    """Validate required runtime config."""
    errors: list[str] = []
    warnings: list[str] = []

    if not settings.amap_api_key:
        errors.append("AMAP_API_KEY 未配置")

    llm_config = get_effective_llm_config()
    if not llm_config["api_key"]:
        errors.append("OPENAI_API_KEY/LLM_API_KEY 未配置（LangChain 必需）")

    if settings.langchain_tracing and not settings.langchain_api_key:
        warnings.append("启用了 LangSmith 追踪但未配置 LANGCHAIN_API_KEY")

    if errors:
        error_msg = "配置错误:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    if warnings:
        print("\n⚠️  配置警告:")
        for w in warnings:
            print(f"  - {w}")

    return True


def print_config() -> None:
    """Print non-sensitive effective configuration for debugging."""
    llm_config = get_effective_llm_config()

    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德地图API Key: {'已配置' if settings.amap_api_key else '未配置'}")
    print(f"OpenAI API Key: {'已配置' if llm_config['api_key'] else '未配置'}")
    print(f"OpenAI Base URL: {llm_config['base_url']}")
    print(f"OpenAI Model: {llm_config['model']}")
    print(f"OpenAI Max Retries: {llm_config['max_retries']}")
    print(f"LangChain 追踪: {'启用' if settings.langchain_tracing else '禁用'}")
    print(f"智能体最大迭代次数: {settings.agent_max_iterations}")
    print(f"智能体温度: {llm_config['temperature']}")
    print(f"智能体超时: {llm_config['timeout']}秒")
    print(f"日志级别: {settings.log_level}")
