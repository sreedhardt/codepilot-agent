import pytest

from codepilot.models import FALLBACK_MODEL, Role, build_model, model_spec


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for role in Role:
        monkeypatch.delenv(f"CODEPILOT_MODEL_{role.name}", raising=False)
    monkeypatch.delenv("CODEPILOT_MODEL_DEFAULT", raising=False)


def test_falls_back_to_free_groq_model():
    assert model_spec(Role.CODER) == FALLBACK_MODEL


def test_default_applies_to_roles_without_override(monkeypatch):
    monkeypatch.setenv("CODEPILOT_MODEL_DEFAULT", "google_genai:gemma-4-31b-it")
    assert model_spec(Role.PR) == "google_genai:gemma-4-31b-it"


def test_role_override_beats_default(monkeypatch):
    monkeypatch.setenv("CODEPILOT_MODEL_DEFAULT", "groq:openai/gpt-oss-120b")
    monkeypatch.setenv("CODEPILOT_MODEL_CODER", "anthropic:claude-sonnet-5")
    assert model_spec(Role.CODER) == "anthropic:claude-sonnet-5"
    assert model_spec(Role.CLASSIFIER) == "groq:openai/gpt-oss-120b"


def test_groq_model_ids_keep_their_slash(monkeypatch):
    monkeypatch.setenv("CODEPILOT_MODEL_CLASSIFIER", "groq:openai/gpt-oss-20b")
    assert model_spec(Role.CLASSIFIER) == "groq:openai/gpt-oss-20b"


@pytest.mark.parametrize("bad", ["claude-sonnet-5", "anthropic:", "openai:gpt-5"])
def test_rejects_malformed_or_unsupported_specs(monkeypatch, bad):
    monkeypatch.setenv("CODEPILOT_MODEL_CODER", bad)
    with pytest.raises(ValueError):
        model_spec(Role.CODER)


def test_missing_key_fails_before_any_api_call(monkeypatch):
    monkeypatch.setenv("CODEPILOT_MODEL_CODER", "anthropic:claude-sonnet-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        build_model(Role.CODER)


def test_builds_each_provider_without_network(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("CODEPILOT_MODEL_CODER", "anthropic:claude-sonnet-5")
    assert type(build_model(Role.CODER)).__name__ == "ChatAnthropic"
    assert type(build_model(Role.CLASSIFIER)).__name__ == "ChatGroq"
