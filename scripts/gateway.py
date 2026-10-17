"""Run the LiteLLM gateway and give CodePilot a spend-capped virtual key.

    uv run python scripts/gateway.py up                # start the gateway (generates its own keys once)
    uv run python scripts/gateway.py key --budget 10   # issue CodePilot a virtual key capped at $10, into .env
    uv run python scripts/gateway.py status            # health, and the key's spend against its budget
    uv run python scripts/gateway.py down

Provider keys are read from CodePilot's .env and handed to Docker Compose through the environment, never
written anywhere else; the gateway's master, salt and database keys are generated into deploy/llm-gateway/.env.
No secret is printed. With CODEPILOT_LLM_GATEWAY_URL and CODEPILOT_LLM_GATEWAY_KEY in .env, every model call
goes through the gateway.
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import urllib.request
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "llm-gateway"
GATEWAY_ENV = DEPLOY / ".env"
CODEPILOT_ENV = ROOT / ".env"
URL = "http://localhost:4000"
PROVIDER_KEYS = ("ANTHROPIC_API_KEY", "GROQ_API_KEY")
MODELS = ["claude-sonnet-5", "claude-haiku-4-5", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def gateway_secrets() -> dict[str, str]:
    """The gateway's own keys, generated on first use."""
    if not GATEWAY_ENV.exists():
        GATEWAY_ENV.write_text("\n".join([f"LITELLM_MASTER_KEY=sk-{secrets.token_urlsafe(32)}",
                                          f"LITELLM_SALT_KEY=sk-{secrets.token_urlsafe(32)}",
                                          f"LITELLM_DB_PASSWORD={secrets.token_urlsafe(24)}", ""]))
        GATEWAY_ENV.chmod(0o600)
    return {k: v for k, v in dotenv_values(GATEWAY_ENV).items() if v}


def compose(*args: str) -> int:
    env = {**os.environ, **gateway_secrets()}
    codepilot = dotenv_values(CODEPILOT_ENV)
    env.update({k: codepilot[k] for k in PROVIDER_KEYS if codepilot.get(k)})
    return subprocess.run(["docker", "compose", *args], cwd=DEPLOY, env=env).returncode


def call(path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        URL + path, data=json.dumps(payload).encode() if payload is not None else None, method="POST" if payload else "GET",
        headers={"Authorization": f"Bearer {gateway_secrets()['LITELLM_MASTER_KEY']}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def set_env_lines(values: dict[str, str]) -> None:
    """Set lines in CodePilot's .env, replacing existing ones, without echoing them."""
    lines = CODEPILOT_ENV.read_text().splitlines() if CODEPILOT_ENV.exists() else []
    lines = [line for line in lines if line.split("=", 1)[0] not in values]
    lines += [f"{k}={v}" for k, v in values.items()]
    CODEPILOT_ENV.write_text("\n".join(lines) + "\n")


def issue_key(budget: float) -> None:
    created = call("/key/generate", {"key_alias": f"codepilot-{secrets.token_hex(3)}", "models": MODELS,
                                     "max_budget": budget, "metadata": {"app": "codepilot"}})
    set_env_lines({"CODEPILOT_LLM_GATEWAY_URL": URL, "CODEPILOT_LLM_GATEWAY_KEY": created["key"]})
    print(f"virtual key {created.get('key_alias')} created with a ${budget:g} budget and written to .env")


def status() -> None:
    health = call("/health/liveliness")
    print(f"gateway: {health}")
    key = dotenv_values(CODEPILOT_ENV).get("CODEPILOT_LLM_GATEWAY_KEY")
    if not key:
        print("no CodePilot key in .env yet: run `scripts/gateway.py key`")
        return
    info = call(f"/key/info?key={key}")["info"]
    print(f"key {info.get('key_alias')}: spent ${info.get('spend', 0):.4f} of ${info.get('max_budget')} budget")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["up", "down", "key", "status"])
    parser.add_argument("--budget", type=float, default=10.0, help="Spend cap for the virtual key, in USD")
    args = parser.parse_args()
    if args.command == "up":
        return compose("up", "-d", "--wait")
    if args.command == "down":
        return compose("down")
    if args.command == "key":
        issue_key(args.budget)
    else:
        status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
