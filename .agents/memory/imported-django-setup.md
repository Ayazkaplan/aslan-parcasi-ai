---
name: Imported Django setup
description: Environment-specific setup behavior for imported Django projects
---

Imported Django projects may include a correct requirements file while the Replit Python environment has none of those packages installed yet.

**Why:** The first workflow failure was a missing Django module, so application debugging could not begin until dependencies were installed through the project package manager.

**How to apply:** When an imported Django workflow fails with `ModuleNotFoundError` before application code runs, install the declared requirements first, then restart the existing workflow and inspect application errors.