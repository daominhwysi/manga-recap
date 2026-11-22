# Memory Index

## Project
- [project] Before and after modifying architecture, workflow, or filesystem structure, check `docs/ARCHITECTURE.md` (Section 6 Impact Checklist) and keep both `docs/ARCHITECTURE.md` and `AGENTS.md` updated → docs/ARCHITECTURE.md
- [project] Never build overfitted/hardcoded workflows, magic page-index cutoffs, or series-specific prompts/heuristics; enforce via `uv run pytest` → AGENTS.md
- [project] Commit every completed and tested feature/fix to the local Git repository using Conventional Commits → AGENTS.md
- [project] Do not load root `RULES.md` (out-of-context academic ML paper style guide) into context → AGENTS.md
- [project] Always create a new dedicated branch for major code changes → project-conventions.md
- [project] AG Kit only supports Gemini CLI and Google Antigravity (not other AI coding tools) → project-conventions.md
- [project] Component metadata uses SemVer while toolkit releases use CalVer → tech-decisions.md
