---
name: documentation-skill
description: Write or fix documentation that matches the existing style and is verified against the code. Load when the task changes only docs, READMEs, docstrings or examples.
metadata:
  skill_id: documentation_skill
  task_type: documentation
  workflow_steps: [read_existing, draft, review_accuracy, update_index]
  example_prompts:
    - "Document the API endpoints and lending rules"
    - "Public helpers in utils.py have no docstrings"
  forbidden_actions:
    - Changing code behaviour in a documentation task
    - Documenting endpoints, parameters or options that do not exist
    - Rewriting sections the issue does not ask about
---

# Documentation

## Workflow

1. **Read existing docs.** Match heading levels, table style, tone and example format already in use.
2. **Draft.** Write from the code, not from memory: every endpoint, parameter, default and status
   code in the docs must be found in the source. Include runnable examples.
3. **Review accuracy.** For each documented item, cite the file that defines it. Check examples
   against the real paths and parameter names. Remove anything you could not verify.
4. **Update the index.** If the document has a table of contents or a layout table, add the new section.

## Verification

Run the test suite at the end even though only docs changed: it proves nothing else was touched.
