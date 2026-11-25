---
name: universal-rules
version: 1.0.0
priority: P0
trigger: always_on
---

# Universal Rules (TIER 0) - AG Kit

> Always-active rules that apply to every request, regardless of domain.

---

## 🌐 Language Handling

When user's prompt is NOT in English:

1. **Internally translate** for better comprehension
2. **Respond in user's language** - match their communication
3. **Code comments/variables** remain in English

---

## 🧹 Clean Code (Global Mandatory)

**ALL code MUST follow `@[skills/clean-code]` rules. No exceptions.**

- **Code**: Concise, direct, no over-engineering. Self-documenting.
- **Testing**: Mandatory. Pyramid (Unit > Int > E2E) + AAA Pattern.
- **Performance**: Measure first. Adhere to current Core Web Vitals standards.
- **Infra/Safety**: 5-Phase Deployment. Verify secrets security.

---

## 📐 Project Architecture, Generalization & Git Rules (MANDATORY)

> 🔴 **Read `AGENTS.md` and `docs/ARCHITECTURE.md` before modifying pipeline stages, prompts, or file layout.**

1. **Architecture Check & `AGENTS.md` / `docs/ARCHITECTURE.md` Sync Rule**: Whenever you modify the architecture, stage logic, schemas, or add/remove/rename files, you MUST check the **Architecture Impact Checklist** in `docs/ARCHITECTURE.md` (Section 6) to verify what downstream components/invariants are affected, and update both `docs/ARCHITECTURE.md` and `AGENTS.md` in the same task.
2. **Zero Overfitting / No Hardcoded Heuristics Rule**: Never introduce series-specific names, quotes, regexes, or hardcoded page-index/pixel/beat-step cutoffs (e.g., `idx < 6`, `idx in (0, 2, 3, 4)`, single-word counters) into `src/` or `configs/`. All logic must rely on structural state machines and adaptive relative formulas across any manga/manhwa and pass `uv run pytest`.
3. **Atomic Feature Git Commit Rule**: After completing and verifying any feature, bugfix, or architectural change (with `uv run pytest` passing), immediately stage and commit the changes to the Git repository with a clear Conventional Commit message (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).

