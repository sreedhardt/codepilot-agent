---
name: dependency-update-skill
description: Upgrade, pin or migrate off a library version or deprecated API, updating requirement files and verifying the whole suite. Load for dependency and deprecation work.
metadata:
  skill_id: dependency_update_skill
  task_type: dependency_update
  workflow_steps: [check_changelog, update, resolve_conflicts, test_all]
  example_prompts:
    - "Remove deprecated Pydantic v1 and FastAPI APIs"
    - "Bump requests from 2.28 to 2.32"
  forbidden_actions:
    - Installing packages (network installs are blocked; use what is already installed)
    - Silencing warnings with filters instead of migrating the code
    - Mixing unrelated refactors into the migration
---

# Dependency update

## Workflow

1. **Check what changed.** List every call site of the deprecated API (`grep`) and note the
   replacement for each. Run the suite once and read the warning summary: it is the to-do list.
2. **Update.** Migrate call sites one pattern at a time with `edit_file`, then update version
   constraints in `requirements.txt` / `pyproject.toml` to match what the code now needs.
3. **Resolve conflicts.** If a replacement API behaves differently (defaults, return types), adjust
   the call site so behaviour is unchanged; existing tests are the contract.
4. **Test everything.** Run the full suite, then the strict check the issue specifies, if any.

## Common breaking changes

**Pydantic 1 → 2**
- `.dict()` → `.model_dump()`; `.json()` → `.model_dump_json()`; `.copy(update=)` → `.model_copy(update=)`
- `parse_obj()` → `model_validate()`; `class Config:` → `model_config = ConfigDict(...)`
- Config keys: `orm_mode` → `from_attributes`, `schema_extra` → `json_schema_extra`,
  `allow_population_by_field_name` → `populate_by_name`
- `@validator` → `@field_validator`; `@root_validator` → `@model_validator`

**FastAPI / Starlette**
- `@app.on_event("startup"/"shutdown")` → a `lifespan` async context manager passed to `FastAPI(lifespan=...)`

**Python standard library**
- `datetime.utcnow()` → `datetime.now(timezone.utc)`

**npm / cargo**
- Read the changelog's "Breaking changes" section between the two versions; update the lockfile
  with the package manager, never by hand.

## Tooling note

Package installation is blocked by the guardrail. If the migration genuinely needs a new package
version, stop and report it instead of attempting an install.
