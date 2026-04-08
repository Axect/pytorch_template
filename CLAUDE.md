# pytorch_template

## Migration Skill Maintenance Rule

When modifying any of these template source files, you **MUST** also update the migration skill documentation:

**Tracked source files:**
- `callbacks.py`, `config.py`, `util.py`, `cli.py`
- `checkpoint.py`, `pruner.py`, `provenance.py`, `main.py`

**Migration docs to update:**
- `.claude/skills/pytorch-migrate/SKILL.md` (version detection table + migration summary)
- `.claude/skills/pytorch-migrate/references/migrations.md` (detailed migration steps)

### When adding a new feature:
1. Create a new migration section (M7, M8, ...) in `references/migrations.md`
2. Add detection marker(s) to the version table in `SKILL.md`
3. Include: detection grep pattern, which files/classes/methods to copy from `$TEMPLATE_DIR`, wiring instructions
4. Do NOT embed code snippets — the migration clones the latest template and reads code from there

### When modifying an existing feature:
- Update the relevant migration section's structural description (class names, method signatures, field names) if they changed
- The actual code is always read from the template clone, so minor code changes don't require migration doc updates

## Development Workflow

- Use Gitflow: feature branches from `main`
- Run `pytest tests/ -x -q` before committing
- Run `python -m cli preflight configs/run_template.yaml --device cpu` to verify template health
- A pre-push hook enforces migration doc updates when source files change
