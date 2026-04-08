---
name: pytorch-migrate
description: >
  Migrate an existing project that uses pytorch_template to the latest version.
  Use this skill when the user says: "update template", "migrate", "upgrade",
  "apply latest changes", "sync with template", "update pytorch_template",
  or when the user's project is missing new features (preflight, hpo-report,
  data field, diagnostic callbacks).
allowed-tools: Bash, Write, Read, Glob, Grep, Edit
---

# pytorch-migrate

Detect the current version of a pytorch_template-based project and apply all necessary migrations to bring it up to date.

## Usage

```
/pytorch-migrate [project_path]
```

If `project_path` is omitted, uses the current working directory.

---

## Step 0: Clone the Template

Clone the latest template into a temporary directory. All code references during migration come from this clone — never from embedded snippets.

```bash
TEMPLATE_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/Axect/pytorch_template.git "$TEMPLATE_DIR"
```

Use `$TEMPLATE_DIR` as the source of truth for all file contents throughout the migration. After migration is complete, clean up:

```bash
rm -rf "$TEMPLATE_DIR"
```

---

## Step 1: Detect Current Version

Read the project's files and determine which version it's based on by checking for feature markers.

Run these checks **in order** — the first missing feature determines the starting migration point:

| Check | How to detect | Version if MISSING |
|-------|---------------|-------------------|
| `config.py` has `RunConfig` dataclass | `class RunConfig` exists | Pre-template (not migratable) |
| `callbacks.py` exists | File exists | v0 (monolithic, pre-callback refactor) |
| `pruner.py` has `PFLPruner` | `class PFLPruner` in pruner.py | v1 (pre-PFL pruner, before 2024-12) |
| `callbacks.py` has `NaNDetectionCallback` | Class exists | v2 (pre-NaN detection, before 2024-09) |
| `callbacks.py` has `CheckpointCallback` | Class exists | v3 (pre-checkpoint, before 2025-04) |
| `config.py` has `data` field in RunConfig | `data: str` in RunConfig | v4 (pre-data-decoupling) |
| `callbacks.py` has `GradientMonitorCallback` | Class exists | v5 (pre-diagnostics) |
| `cli.py` has `preflight` command | `def preflight` exists | v5 (pre-preflight) |
| `cli.py` has `hpo_report` command | `def hpo_report` exists | v5 (pre-hpo-report) |
| `callbacks.py` has `CSVLoggingCallback` | Class exists | v6 (pre-dual-logging) |
| `config.py` has `logging` field in RunConfig | `logging: str` in RunConfig | v6 (pre-dual-logging) |
| `provenance.py` exists | File exists | v6 (pre-provenance) |
| `cli.py` has `update_skills` command | `def update_skills` exists | v7 (pre-TUI-tabs) |
| All checks pass | — | Current (up to date) |

```bash
# Quick detection script
grep -c "class PFLPruner" pruner.py 2>/dev/null || echo "0"
grep -c "class NaNDetectionCallback" callbacks.py 2>/dev/null || echo "0"
grep -c "class CheckpointCallback" callbacks.py 2>/dev/null || echo "0"
grep -c "data: str" config.py 2>/dev/null || echo "0"
grep -c "class GradientMonitorCallback" callbacks.py 2>/dev/null || echo "0"
grep -c "def preflight" cli.py 2>/dev/null || echo "0"
grep -c "def hpo_report" cli.py 2>/dev/null || echo "0"
grep -c "class CSVLoggingCallback" callbacks.py 2>/dev/null || echo "0"
grep -c "logging: str" config.py 2>/dev/null || echo "0"
test -f provenance.py && echo "1" || echo "0"
grep -c "def update_skills" cli.py 2>/dev/null || echo "0"
```

---

## Step 2: Apply Migrations

Apply only the migrations that are needed, in order. Each migration is independent and idempotent.

Read `references/migrations.md` for the structural migration guide. Each migration describes WHICH files, classes, and methods to copy from `$TEMPLATE_DIR` and HOW to wire them into the user's project. Read the actual source files from `$TEMPLATE_DIR` to get the current code.

### Migration Summary

| Migration | From | Changes |
|-----------|------|---------|
| M1: PFL Pruner | v1→v2 | Add `pruner.py`, update `optimize_template.yaml` |
| M2: NaN Detection + Checkpoint | v2→v3 | Add NaN/Checkpoint callbacks, add `CheckpointConfig` to `config.py` |
| M3: Modular CLI | v3→v4 | Add `cli.py` with typer, refactor `main.py` |
| M4: Data Decoupling | v4→v5 | Add `data` field to `RunConfig`, add `load_data()` method, update CLI |
| M5: Diagnostics + Preflight + HPO Report | v5→v6 | Add callbacks, CLI commands, `validate_semantics()` |
| M6: Dual Logging + TUI Monitor + Provenance | v6→v7 | Add `CSVLoggingCallback`, `TUILoggingCallback`, `LatestModelCallback`, `provenance.py`, `logging` field, `doctor`/`monitor` CLI |
| M7: TUI Monitor Tabs + CLI Enhancements | v7→current | TUI monitor dynamic column tabs, `monitor --list`, `update-skills` CLI command |

---

## Step 3: Migrate User's Config Files

After updating the template code, scan for existing YAML configs and update them:

```bash
# Find all run configs
find configs/ -name "*.yaml" -type f 2>/dev/null
```

For each YAML config file:

### Add `data` field (M4+)

If the config lacks a `data:` field, add it after `criterion_config:`:

```yaml
data: util.load_data  # Default — change to your project's data loader
```

**Important:** If the project has a custom `load_data()` in `util.py`, the user should either:
1. Keep `data: util.load_data` (works as-is)
2. Move their data loader to a dedicated module and use that path

### Verify config compatibility

After migration, run:
```bash
python -m cli preflight <config_path> --device cpu
```

---

## Step 4: Update Dependencies

Check if new dependencies are needed:

```bash
# Required since v3 (CLI refactor)
pip show typer rich beaupy 2>/dev/null || echo "Need: uv pip install typer rich beaupy"

# Required since v1 (template inception)
pip show pytorch-optimizer pytorch-scheduler 2>/dev/null || echo "Need: uv pip install pytorch-optimizer pytorch-scheduler"
```

---

## Step 5: Verify

```bash
# Run tests
pytest tests/ -x -q

# Run preflight on existing configs
python -m cli preflight <config> --device cpu

# Check system health
python -m cli doctor
```

---

## Important Notes

- **Never overwrite user's custom `load_data()`** in util.py. The data decoupling migration adds the `data` field to RunConfig but preserves the existing function.
- **Never overwrite user's custom callbacks.** Add new callbacks alongside existing ones.
- **Preserve user's custom model code.** Migrations only touch template infrastructure files.
- **Back up first.** Recommend `git stash` or `git commit` before migrating.
- **Config files need the `data` field** to use the new pluggable data loading. But if omitted, the default `util.load_data` is used automatically (backward compatible).
