---
description: Initialize a new project from templates/project.
allowed-tools: Read, Write, Bash, Glob
argument-hint: --project <slug>
---
# /project-init
Copy `templates/project/*` to `projects/<slug>/` and replace `<slug>` placeholder in pipeline.json.
Also append the project to `projects/PROJECTS.md`.
