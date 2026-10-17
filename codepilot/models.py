"""Per-role model selection.

Every agent role reads its model from the environment as a `provider:model`
string, so swapping Groq, Gemini, Anthropic or a local Ollama model is a
config change, not a code change:

    CODEPILOT_MODEL_CODER=anthropic:claude-sonnet-5
    CODEPILOT_MODEL_CLASSIFIER=groq:openai/gpt-oss-20b

A role with no override falls back to CODEPILOT_MODEL_DEFAULT, and then to the
free Groq default, so a fresh clone runs with nothing but GROQ_API_KEY set.
"""

import os
from enum import StrEnum

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


class Role(StrEnum):
    ORCHESTRATOR = "orchestrator"
    CLASSIFIER = "classifier"
    REPO_EXPLORER = "repo_explorer"
    CODER = "coder"
    TESTER = "tester"
    PR = "pr"
    SUMMARIZER = "summarizer"
    REVIEWER = "reviewer"
    INPUT_RAIL = "input_rail"


FALLBACK_MODEL = "groq:openai/gpt-oss-120b"

# Groq's key is GROQ_API_KEY; the other providers read their own standard variables.
PROVIDER_KEYS = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "ollama": None,
}


def model_spec(role: Role) -> str:
    """Resolve the `provider:model` string for a role."""
    spec = (
        os.getenv(f"CODEPILOT_MODEL_{role.name}")
        or os.getenv("CODEPILOT_MODEL_DEFAULT")
        or FALLBACK_MODEL
    )
    provider, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"Model for role {role} must look like 'provider:model', got {spec!r}")
    if provider not in PROVIDER_KEYS:
        raise ValueError(f"Unsupported provider {provider!r} for role {role}; use one of {sorted(PROVIDER_KEYS)}")
    return spec


GATEWAY_PROVIDERS = {"anthropic", "groq"}  # the providers deploy/llm-gateway/config.yaml serves


def gateway() -> tuple[str, str] | None:
    """(url, virtual key) when CodePilot is configured to call models through the LiteLLM gateway."""
    url, key = os.getenv("CODEPILOT_LLM_GATEWAY_URL"), os.getenv("CODEPILOT_LLM_GATEWAY_KEY")
    return (url.rstrip("/"), key) if url and key else None


def chat_model(provider: str, model: str, **kwargs) -> BaseChatModel:
    """A chat model for `provider:model`, through the gateway when one is configured.

    Through the gateway, CodePilot holds only a spend-capped virtual key. Anthropic models use the gateway's
    Anthropic-format endpoint, so prompt caching and streaming work as they do direct; Groq models use its
    OpenAI-compatible endpoint under the same names.
    """
    if (via := gateway()) and provider in GATEWAY_PROVIDERS:
        url, key = via
        if provider == "anthropic":
            return init_chat_model(model, model_provider="anthropic", base_url=url, api_key=key, **kwargs)
        return init_chat_model(model, model_provider="openai", base_url=f"{url}/v1", api_key=key, **kwargs)
    env_key = PROVIDER_KEYS[provider]
    if env_key and not os.getenv(env_key):
        raise RuntimeError(f"{provider}:{model} needs {env_key} (or CodePilot's LLM gateway settings)")
    return init_chat_model(model, model_provider=provider, **kwargs)


def build_model(role: Role, **kwargs) -> BaseChatModel:
    """Instantiate the chat model configured for a role."""
    provider, _, model = model_spec(role).partition(":")
    return chat_model(provider, model, **kwargs)
