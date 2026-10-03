# Seeded issues for `bookshelf-api`

Six issues filed against the target repo. Only the **Issue** text goes to
GitHub; the **Ground truth** block is the answer key for evaluating CodePilot
and must never be shown to the agent.

Every issue is labelled `ai-assignable`. The baseline suite (20 tests) passes
before any fix, so each bug is latent: CodePilot has to reproduce it with a new
failing test first.

| # | Expected type | Difficulty | Verified by |
|---|---|---|---|
| 1 | `bug_fix` | Easy | New boundary test + existing suite |
| 2 | `bug_fix` | Easy | New search test + existing suite |
| 3 | `feature_addition` | Medium | New endpoint tests + existing suite |
| 4 | `dependency_update` | Medium | Suite passes with Pydantic deprecations as errors |
| 5 | `documentation` | Easy | README review (no tests) |
| 6 | `config_change` | Medium | Env-override tests + existing suite |

---

## 1. Members are blocked from borrowing on the day a book is due

**Issue**

> A member returned a book this morning, on its due date, and then tried to
> borrow another one, but checkout failed with `409 Member has overdue loans`.
> Same thing happens if they still have the book on the due date and try to
> borrow something else. A loan due today isn't overdue yet. It should only
> count as overdue from the day after the due date.
>
> Steps: check out book 1 as member 1, move the clock to the due date, check
> out book 2 as member 1. Expected 201, got 409.
>
> `GET /members/{id}/loans` also shows `"overdue": true` on the due date.

**Ground truth**

- Root cause: `app/loans.py` `is_overdue` uses `today >= loan.due`; must be `today > loan.due`.
- Files changed: `app/loans.py`, plus a test in `tests/test_loans.py`.
- Acceptance: a new test checking out on the due date returns 201; all 20 baseline tests still pass.
- Trap: changing `late_fee_cents` is wrong; fees are already correct on the due date.

## 2. Book search is case-sensitive

**Issue**

> Searching the catalog for `dune` returns nothing, but `Dune` finds it. Same
> for authors: `le guin` returns 0 results. Users don't type exact
> capitalisation. Search should ignore case for both title and author.

**Ground truth**

- Root cause: `app/catalog.py` `search_books` compares raw strings.
- Fix: compare `query.casefold()` (or `.lower()`) against title and author.
- Files changed: `app/catalog.py`, plus a test in `tests/test_books.py`.
- Acceptance: `GET /books?q=dune` returns 1 result and `q=le guin` returns 2; baseline suite passes.

## 3. Allow members to renew a loan

**Issue**

> Members want to extend a loan without returning the book. Please add
> `POST /loans/{loan_id}/renew` that pushes the due date out by another loan
> period. Rules:
>
> - A loan can be renewed at most 2 times.
> - Overdue loans can't be renewed (they need to be returned first).
> - Returned loans can't be renewed.
>
> Return the updated loan. Use 409 for the rule violations and 404 for an
> unknown loan, consistent with the other endpoints.

**Ground truth**

- Files changed: `app/models.py` (add `renewals: int = 0` to `Loan`), `app/loans.py` (`renew` function), `app/routes/loans.py` (endpoint), new tests.
- New due date = current due + `LOAN_PERIOD_DAYS` (not today + period).
- Acceptance: tests for success, a third renewal → 409, overdue → 409, returned → 409, unknown → 404; baseline suite passes.
- Interaction: renewing on the due date must succeed only after issue 1 is fixed (a loan due today isn't overdue). If solved first, the agent's own test may expose bug 1; good agents note it rather than silently fixing it here.

## 4. Remove deprecated Pydantic v1 and FastAPI APIs

**Issue**

> The test run prints a wall of deprecation warnings. We're on Pydantic 2 and
> a current FastAPI, but the code still uses v1-era APIs (`.dict()`,
> `.copy(update=...)`, class-based `Config` with `schema_extra`) and
> `@app.on_event("startup")`. These will break when Pydantic 3 lands.
>
> Please migrate to the current APIs and bump `requirements.txt` to require
> Pydantic 2. Done means this passes with no failures:
>
>     python -m pytest -W error::pydantic.warnings.PydanticDeprecatedSince20
>
> and there's no `on_event` deprecation warning left.

**Ground truth**

- `.dict()` → `.model_dump()` in `app/routes/books.py` and `app/routes/loans.py`.
- `.copy(update=...)` → `.model_copy(update=...)` in `app/loans.py`.
- `class Config: schema_extra` → `model_config = ConfigDict(json_schema_extra=...)` in `app/models.py`.
- `@app.on_event("startup")` → `lifespan` context manager in `app/main.py`.
- `requirements.txt`: `pydantic>=1.10` → `pydantic>=2`.
- Acceptance: the strict command passes; no `on_event` warning in `pytest -rw`; baseline suite passes.
- Guardrail note: no `pip install` needed; Pydantic 2 is already installed. An agent trying to install packages should be blocked.
- Known noise: Starlette's own `httpx` deprecation warning is out of scope and can't be fixed in this repo.

## 5. Document the API endpoints and lending rules

**Issue**

> The README covers setup and the lending rules, but there's no reference for
> the endpoints. New contributors have to read the route files to find out what
> exists. Please add an API section listing each endpoint with its method,
> path, purpose and the main error responses, plus a short `curl` example for
> checking out and returning a book.

**Ground truth**

- Files changed: `README.md` only.
- Endpoints to cover: `GET /health`, `GET /books` (`q`, `limit`, `offset`), `GET /books/{id}`, `POST /loans`, `POST /loans/{id}/return`, `GET /loans/{id}`, `GET /members/{id}/loans`.
- Errors: 404 unknown book/member/loan; 409 no copies, loan limit, overdue member, already returned; 422 validation.
- Acceptance (manual or LLM-as-judge with this list): every endpoint present, status codes accurate, example commands valid against the running app.
- Trap: inventing endpoints that don't exist (e.g. renew, unless issue 3 is merged).

## 6. Make lending policy configurable via environment variables

**Issue**

> Different branches want different loan periods and fees, but these are
> hard-coded in `app/config.py`. Please make them configurable through
> environment variables, keeping the current values as defaults:
>
> - `BOOKSHELF_LOAN_PERIOD_DAYS` (default 14)
> - `BOOKSHELF_DAILY_LATE_FEE_CENTS` (default 25)
> - `BOOKSHELF_MAX_LATE_FEE_CENTS` (default 2000)
> - `BOOKSHELF_MAX_ACTIVE_LOANS` (default 3)
>
> Invalid values (non-numeric, zero or negative) should fail fast at startup
> with a clear error rather than silently misbehaving.

**Ground truth**

- Files changed: `app/config.py` (read env with defaults and validation), new tests; README optional.
- Acceptance: tests set env vars (monkeypatch + reload, or a settings object) and observe new values; invalid values raise; baseline suite passes unchanged with no env set.
- Trap: modules that do `from app.config import LOAN_PERIOD_DAYS` capture the value at import time; a correct solution handles this (settings function/object, or reload in tests).
