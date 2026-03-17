"""LLM service wrapper (LangChain)."""

from __future__ import annotations

from typing import Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config import get_effective_llm_config

# Global singleton LLM instance
_llm_instance: Optional[BaseChatModel] = None
_llm_signature: Optional[Tuple[str, str, str, float, float, int]] = None


def _make_signature(llm_config: dict) -> Tuple[str, str, str, float, float, int]:
    """Build a stable config signature to detect .env changes at runtime."""
    return (
        str(llm_config["api_key"]),
        str(llm_config["base_url"]),
        str(llm_config["model"]),
        float(llm_config["temperature"]),
        float(llm_config["timeout"]),
        int(llm_config["max_retries"]),
    )


def get_llm() -> BaseChatModel:
    """Get singleton LangChain ChatOpenAI instance."""
    global _llm_instance, _llm_signature

    llm_config = get_effective_llm_config()
    current_signature = _make_signature(llm_config)

    # Rebuild instance when .env/runtime config changes, not only on first call.
    if _llm_instance is None or _llm_signature != current_signature:
        if _llm_instance is not None and _llm_signature != current_signature:
            print("[INFO] LLM configuration changed, rebuilding ChatOpenAI instance...")

        api_key = llm_config["api_key"]
        base_url = llm_config["base_url"]
        model = llm_config["model"]
        temperature = llm_config["temperature"]
        timeout = llm_config["timeout"]
        max_retries = llm_config["max_retries"]

        if not api_key:
            raise ValueError("OPENAI_API_KEY/LLM_API_KEY 未配置（LangChain 必需）")

        _llm_instance = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=2000,
            timeout=timeout,
            max_retries=max_retries,
        )

        print("[SUCCESS] LangChain LLM 初始化成功")
        print(f"   模型: {model}")
        print(f"   Base URL: {base_url}")
        print(f"   Temperature: {temperature}")
        print(f"   Timeout: {timeout}s")
        print(f"   Max retries: {max_retries}")
        _llm_signature = current_signature

    return _llm_instance


def reset_llm() -> None:
    """Reset LLM singleton (for tests or reload)."""
    global _llm_instance, _llm_signature
    _llm_instance = None
    _llm_signature = None
