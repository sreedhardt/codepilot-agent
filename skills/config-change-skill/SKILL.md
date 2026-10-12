---
name: config-change-skill
description: Make hard-coded values configurable or change settings and tooling config, keeping current defaults and validating input. Load for configuration tasks.
metadata:
  skill_id: config_change_skill
  task_type: config_change
  workflow_steps: [locate_values, design_settings, implement, test_overrides]
  example_prompts:
    - "Make lending policy configurable via environment variables"
    - "Log level is hard-coded to DEBUG"
  forbidden_actions:
    - Changing default values (unset configuration must behave exactly as before)
    - Committing real secrets or environment-specific values
    - Editing .env files, keys or credentials
---

# Config change

## Workflow

1. **Locate values.** Find every hard-coded value the issue names and every place that reads it.
2. **Design settings.** One variable per value, named as the issue specifies, defaults equal to the
   current constants. Decide validation: type, range, and the error raised for bad input.
3. **Implement.** Read configuration in one place. Watch for import-time capture:
   `from config import X` copies the value when the module is imported, so later environment
   changes are invisible. Prefer a settings function or object read at call time.
4. **Test overrides.** With no variables set, the full suite passes unchanged. Add tests that set
   each variable (pytest `monkeypatch.setenv`) and observe the new behaviour, plus one per invalid
   value showing a clear error.
