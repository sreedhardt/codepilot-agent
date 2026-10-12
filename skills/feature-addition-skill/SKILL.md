---
name: feature-addition-skill
description: Add new behaviour by following the patterns of similar existing features, with tests and documentation. Load when the task adds an endpoint, option or capability.
metadata:
  skill_id: feature_addition_skill
  task_type: feature_addition
  workflow_steps: [explore_pattern, design, implement, test, document]
  example_prompts:
    - "Allow members to renew a loan"
    - "Filter orders by date range"
  forbidden_actions:
    - Changing the behaviour or response shape of existing endpoints
    - Adding dependencies or new frameworks for the feature
    - Implementing rules the issue does not ask for
    - Skipping tests for the error paths the issue lists
---

# Feature addition

## Workflow

1. **Explore the pattern.** Find the closest existing feature (an endpoint with the same shape, a
   rule in the same module). Read its route, domain function, model and tests. New code should look
   like it was written by the same author.
2. **Design.** Write down, before coding: the interface (path, method, request, response), every rule
   in the issue, and the status code for each failure. Map each rule to one test.
3. **Implement.** Domain logic in the domain module, HTTP mapping in the route module, schema changes
   in the models module. Keep new fields backwards compatible: give them defaults.
4. **Test.** One test per rule and per error status, plus the happy path. Run the full suite.
5. **Document.** If the README lists endpoints or rules, add the new one in the same style.

## Interface checklist

- Status codes consistent with neighbouring endpoints (404 unknown id, 409 rule conflict, 422 validation).
- Error `detail` strings in the same voice as existing ones.
- Existing clients keep working: no renamed fields, no new required fields.
