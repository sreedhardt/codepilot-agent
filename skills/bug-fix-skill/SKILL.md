---
name: bug-fix-skill
description: Reproduce a reported bug with a failing test, fix the root cause with a minimal edit, and verify the full suite passes. Load when the task is a bug fix.
metadata:
  skill_id: bug_fix_skill
  task_type: bug_fix
  workflow_steps: [reproduce, localize, fix, verify]
  example_prompts:
    - "Members are blocked from borrowing on the day a book is due"
    - "Unknown user id returns 500 instead of 404"
  forbidden_actions:
    - Fixing before a failing test reproduces the bug
    - Weakening or deleting existing tests to make the suite pass
    - Changing behaviour the issue does not mention
    - Catching broad exceptions to hide the symptom
---

# Bug fix

## Workflow

1. **Reproduce.** Write a test that encodes the behaviour the issue says is correct. Run only that
   test and confirm it fails for the reason the issue describes. If it passes, the bug is somewhere
   else: re-read the issue before going further.
2. **Localize.** Trace from the failing assertion to the code that produces the wrong value. Read the
   function and its callers. Name the root cause in one sentence before editing.
3. **Fix.** Make the smallest change that corrects the root cause, with `edit_file`. Fix the cause,
   not the symptom: if a comparison is off by one, fix the comparison rather than adjusting the input.
4. **Verify.** Run the new test, then the full suite. Every pre-existing test must still pass.

## Debugging checklist

- Boundaries: `<` vs `<=`, off-by-one in ranges and slices, inclusive vs exclusive dates.
- Normalization: case, whitespace, Unicode (`casefold()` over `lower()` for user text), time zones.
- Empty and missing input: empty strings, empty lists, `None`, zero.
- Mutation: shared default arguments, objects mutated in place and reused.
- Error mapping: domain errors that should become 404/409/422 rather than 500.

## Reading a stack trace

Start from the **last** frame inside this repository, not the last frame overall: library frames
show where the failure surfaced, repository frames show what passed it bad input.
