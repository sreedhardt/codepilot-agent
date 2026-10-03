# CodePilot — a multi-agent coding platform

A terminal-based multi-agent system that picks up GitHub issues, plans and
implements fixes in a sandbox, verifies them with tests, and opens pull
requests, with human approval gates on anything risky.

> Work in progress — AI Engineering Cohort capstone (codebasics.io).

## Setup

```bash
uv sync
cp .env.example .env   # add GROQ_API_KEY at minimum
uv run pytest
```

Every agent role reads its model from `.env` as `provider:model` (Groq,
Anthropic, Gemini or local Ollama), so the whole system runs on Groq's free
tier, and individual roles can be upgraded independently.
