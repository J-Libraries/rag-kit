from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class LLMProviderError(ValueError):
    pass


def get_llm(
    provider: str = "sarvam",
    model: str | None = None,
    temperature: float = 0.2,
    **kwargs: Any,
):
    provider = provider.strip().lower()

    if provider == "sarvam":
        return _build_sarvam_llm(
            model=model or "sarvam-m",
            temperature=temperature,
            **kwargs,
        )

    if provider == "openai":
        return _build_openai_llm(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            **kwargs,
        )

    if provider in {"anthropic", "claude"}:
        return _build_anthropic_llm(
            model=model or "claude-3-5-haiku-latest",
            temperature=temperature,
            **kwargs,
        )

    raise LLMProviderError(
        f"Unsupported llm provider: {provider}. "
        f"Supported providers: sarvam, openai, anthropic, claude"
    )


def _build_sarvam_llm(model: str, temperature: float, **kwargs: Any):
    """
    Priority order:
    1. vendored/local langchain_sarvam package
    2. existing root config.get_llm()
    3. raise helpful error
    """

    # Option 1: local vendored source inside your project
    sarvam_import_attempts = [
        "langchain_sarvam",
        "third_party.langchain_sarvam",
        "vendor.langchain_sarvam",
    ]

    for module_name in sarvam_import_attempts:
        try:
            module = __import__(module_name, fromlist=["ChatSarvam"])
            ChatSarvam = getattr(module, "ChatSarvam")
            return ChatSarvam(
                model=model,
                temperature=temperature,
                api_key=os.getenv("SARVAM_API_KEY"),
                **kwargs,
            )
        except Exception:
            pass

    # Option 2: fallback to your older tutorial config.py
    try:
        from config import get_llm as legacy_get_llm
        return legacy_get_llm()
    except Exception as exc:
        raise LLMProviderError(
            "Sarvam provider selected, but no working Sarvam integration was found.\n"
            "Do one of these:\n"
            "1. keep your existing root-level config.py with get_llm(), or\n"
            "2. place your downloaded langchain_sarvam source in importable path, or\n"
            "3. replace _build_sarvam_llm() with your direct Sarvam wrapper.\n"
        ) from exc


def _build_openai_llm(model: str, temperature: float, **kwargs: Any):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMProviderError(
            "OpenAI support is not installed. Install with: pip install 'rag-kit[openai]'"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMProviderError("OPENAI_API_KEY not found in environment.")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )


def _build_anthropic_llm(model: str, temperature: float, **kwargs: Any):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise LLMProviderError(
            "Anthropic support is not installed. Install with: pip install 'rag-kit[anthropic]'"
        ) from exc

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMProviderError("ANTHROPIC_API_KEY not found in environment.")

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )