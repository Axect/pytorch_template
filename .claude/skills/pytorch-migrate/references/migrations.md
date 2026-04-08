# Migration Reference

Detailed steps for each migration. Apply in order, skip migrations that are already present.

The template was cloned into `$TEMPLATE_DIR` in Step 0. All source files and YAML configs referenced below exist there. Read them before editing the user's project files.

---

## M1: PFL Pruner (v1 → v2)

**Detect:** `grep -c "class PFLPruner" pruner.py` returns 0

### pruner.py

**Action:** Create from `$TEMPLATE_DIR/pruner.py`

Key components:
- `Trial` (dataclass) — holds per-trial intermediate state: `trial_id`, `current_epoch`, `seed_values` dict; method `add_value(seed, value)`
- `BasePruner` — Optuna-like pruner base class; methods `register_trial(trial_id)`, `complete_trial(trial_id)`, `report(trial_id, seed, epoch, value)`, `should_prune()`, abstract `_should_prune_trial(trial)`
- `PFLPruner(BasePruner)` — predicted-final-loss pruner; constructor params `n_startup_trials`, `n_warmup_epochs`, `top_k`, `target_epoch`; maintains `top_pairs` list; overrides `complete_trial` and `_should_prune_trial`

### callbacks.py

**Action:** Modify existing file

Changes needed:
- Add `OptimizerModeCallback` class (priority=10) before `EarlyStoppingCallback`; copy from `$TEMPLATE_DIR/callbacks.py` — search for `class OptimizerModeCallback`; hook `on_train_epoch_begin` calls `trainer.optimizer.train()` if the optimizer has a callable `train` method (needed for SPlus, ScheduleFree); hook `on_val_begin` calls `trainer.optimizer.eval()` similarly
- Add `LossPredictionCallback` class (priority=70) after `EarlyStoppingCallback`; copy from `$TEMPLATE_DIR/callbacks.py` — search for `class LossPredictionCallback`; constructor param `max_epochs: int`; hook `on_val_end` appends val_loss to history, calls `predict_final_loss(self.val_losses, self.max_epochs)` after epoch 10, writes result to `trainer._loss_prediction`
- Add `PrunerCallback` class (priority=85) after `WandbLoggingCallback`
- Copy from `$TEMPLATE_DIR/callbacks.py` — search for `class PrunerCallback`
- Constructor: `__init__(self, pruner, trial, seed)`
- Hook: `on_val_end` — calls `pruner.report(...)` then raises `optuna.TrialPruned` if `pruner.should_prune()` returns True

### util.py

**Action:** Modify existing file

Changes needed:
- Copy `predict_final_loss(losses, max_epochs)` function from `$TEMPLATE_DIR/util.py` (placed before the `Trainer` class)
- Add imports: `OptimizerModeCallback`, `LossPredictionCallback`, `PrunerCallback` from `callbacks`
- In `run()` callbacks list: insert `OptimizerModeCallback()` as the FIRST callback (it must run before all others to toggle optimizer train/eval mode)
- In `run()` callbacks list: insert `LossPredictionCallback(run_config.epochs)` after `NaNDetectionCallback()`
- In `run()`: call `pruner.register_trial(trial.number)` at the start if pruner is not None
- In `run()`: append `PrunerCallback(pruner, trial, seed)` to callbacks list when pruner is not None
- In `run()` finally block: call `pruner.complete_trial(trial.number)` once after all seeds finish

### config.py

**Action:** Modify existing file

Changes needed:
- Add `pruner: dict` field (default empty dict) to `OptimizeConfig` dataclass
- Copy `create_pruner()` method from `$TEMPLATE_DIR/config.py` into `OptimizeConfig` (after `_create_sampler()`); it uses `importlib` to instantiate the pruner class from `pruner["name"]` and `pruner.get("kwargs", {})`

### configs/optimize_template.yaml

**Action:** Add section from `$TEMPLATE_DIR/configs/optimize_template.yaml`

Fields to add under `pruner:`:
- `name:` — fully qualified class path (e.g., `pruner.PFLPruner`)
- `kwargs:` — sub-fields `n_startup_trials`, `n_warmup_epochs`, `top_k`, `target_epoch`

### cli.py / main.py

**Action:** Modify existing file

Changes needed:
- In the `train` command's HPO branch: call `opt_config.create_pruner()` to create the pruner object
- Pass `pruner=pruner` to every `run(...)` call inside the objective function

---

## M2: NaN Detection + Checkpoint (v2 → v3)

**Detect:** `grep -c "class NaNDetectionCallback" callbacks.py` returns 0

### callbacks.py

**Action:** Modify existing file

Changes needed:
- Add `import math` at the top if not already present
- Add `NaNDetectionCallback` (priority=5) before `EarlyStoppingCallback`; copy from `$TEMPLATE_DIR/callbacks.py`; constructor sets `self.nan_detected = False`; hook `on_epoch_end` checks `math.isnan(train_loss) or math.isnan(val_loss)` and sets flag
- Add `CheckpointCallback` (priority=95) after `EarlyStoppingCallback`; copy from `$TEMPLATE_DIR/callbacks.py`; constructor `__init__(self, checkpoint_manager, config_hash: str = "")`; hook `on_epoch_end` calls `checkpoint_manager.maybe_save(...)` passing early stopping state extracted from the callbacks list

### config.py

**Action:** Modify existing file

Changes needed:
- Add `EarlyStoppingConfig` dataclass (before `RunConfig`) — copy from `$TEMPLATE_DIR/config.py`; fields: `enabled: bool`, `patience: int`, `mode: str`, `min_delta: float`
- Add `CheckpointConfig` dataclass (after `EarlyStoppingConfig`) — copy from `$TEMPLATE_DIR/config.py`; fields: `enabled: bool`, `save_every_n_epochs: int`, `keep_last_k: int`, `save_best: bool`, `monitor: str`, `mode: str`
- Add `early_stopping_config: EarlyStoppingConfig` field to `RunConfig` (default factory)
- Add `checkpoint_config: CheckpointConfig` field to `RunConfig` (default factory)
- In `RunConfig.__post_init__`: add dict-to-dataclass coercion for both new fields (if value is a dict, construct the dataclass from it)

### checkpoint.py

**Action:** Create from `$TEMPLATE_DIR/checkpoint.py`

Key components:
- `CheckpointManager` — constructor params `run_dir`, `save_every_n`, `keep_last_k`, `save_best`, `monitor`, `mode`; class constant `CHECKPOINT_VERSION = 1`; methods: `save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, metrics, early_stopping_state, config_hash)`, `load_checkpoint(path, model, optimizer, scheduler, device, config_hash)`, `maybe_save(epoch, model, optimizer, scheduler, val_loss, metrics, early_stopping_state, config_hash)`, `find_latest_checkpoint()`, `_cleanup_old_checkpoints()`, `_capture_rng_states()`, `_restore_rng_states(rng)`
- `SeedManifest` — tracks completed seeds for multi-seed resume; constructor param `group_path`; methods: `mark_complete(seed, val_loss, wandb_run_id, metrics)`, `is_complete(seed)`, `get_total_loss()`, `get_complete_count()`; persists as `seed_manifest.json`

### util.py

**Action:** Modify existing file

Changes needed:
- Add imports: `CheckpointManager`, `SeedManifest` from `checkpoint`; `NaNDetectionCallback`, `CheckpointCallback` from `callbacks`
- In `Trainer.train()`: after each epoch's callbacks fire, check if any `NaNDetectionCallback` has `nan_detected = True`; if so, set `val_loss = math.inf` and break
- In `run()`: create `SeedManifest(group_path)` before the seed loop
- In the seed loop: skip seeds where `manifest.is_complete(seed)` returns True
- In the seed loop: if `run_config.checkpoint_config.enabled`, create `CheckpointManager` from config fields and append `CheckpointCallback(ckpt_manager, config_hash)` to callbacks list
- After each seed completes: call `manifest.mark_complete(seed, val_loss)` and break the seed loop if `math.isinf(val_loss)`
- Return `manifest.get_total_loss()` divided by completed seed count

### configs/run_template.yaml

**Action:** Add section from `$TEMPLATE_DIR/configs/run_template.yaml`

Fields to add:
- `early_stopping_config:` — sub-fields `enabled`, `patience`, `mode`, `min_delta`
- `checkpoint_config:` — sub-fields `enabled`, `save_every_n_epochs`, `keep_last_k`, `save_best`, `monitor`, `mode`

---

## M3: Modular CLI (v3 → v4)

**Detect:** `test -f cli.py` fails (cli.py does not exist)

### cli.py

**Action:** Copy from `$TEMPLATE_DIR/cli.py`

Key components:
- `app` — `typer.Typer` instance
- `console` — `rich.console.Console` instance
- `train(run_config, device, optimize_config)` command — handles both normal training and HPO branch
- `validate(run_config)` command — checks structural and runtime config correctness
- `preview(run_config)` command — prints model, optimizer, scheduler, criterion summary
- `analyze(project, group, seed, device)` command — interactive (using `beaupy`) or non-interactive model analysis
- Entry point: `if __name__ == "__main__": app()`

Dependencies to add if not present: `typer`, `rich`, `beaupy`

### main.py

**Action:** No changes required

`main.py` remains as a legacy argparse entry point. Existing scripts calling `python main.py --run_config ...` continue to work unchanged.

---

## M4: Data Decoupling (v4 → v5)

**Detect:** `grep -c "data: str" config.py` returns 0

### config.py

**Action:** Modify existing file

Changes needed:
- Add `data: str` field to `RunConfig` (after `criterion_config`); default value is `"util.load_data"`
- In `RunConfig.__post_init__`: validate that `self.data` contains at least one `"."` (module.function format); raise `ValueError` if not
- In `validate_for_execution()`: add `("data", self.data)` to the import path check list
- Add `load_data()` method to `RunConfig` (after `create_criterion()`); copy from `$TEMPLATE_DIR/config.py`; uses `importlib` to split on the last `"."`, import the module, and call the function
- Add `validate_semantics()` method to `RunConfig` (after `validate_for_execution()`); copy from `$TEMPLATE_DIR/config.py`; returns a list of issue strings covering: non-positive lr, duplicate seeds, scheduler-specific constraint checks (HyperbolicLR `upper_bound >= total_steps`, CosineAnnealingLR `T_max == epochs`), and early stopping patience vs epochs check

### cli.py / main.py

**Action:** Modify existing file

Changes needed:
- Remove `from util import load_data` imports
- Replace all direct `load_data()` calls with `base_config.load_data()` (or `config.load_data()` in the `analyze` command)

### YAML configs

**Action:** Add `data:` field to all existing run configs that have `criterion_config` but no `data:` field

- Default value: `util.load_data`
- For recipe-specific loaders (e.g., `recipes/regression/data.py`), use the fully qualified path: `recipes.regression.data.load_data`

---

## M5: Diagnostics + Preflight + HPO Report (v5 → v6)

**Detect:** `grep -c "class GradientMonitorCallback" callbacks.py` returns 0

### callbacks.py

**Action:** Modify existing file

Changes needed:
- Add `GradientMonitorCallback` (priority=12) — copy from `$TEMPLATE_DIR/callbacks.py`; constructor params `warn_threshold: float`; tracks `_step_grad_norms` and `epoch_max_grad_norms`; hooks: `on_train_epoch_begin` (reset state), `on_train_step_end` (accumulate gradient norm across all params, warn if above threshold), `on_epoch_end` (append epoch max, write to `trainer._max_grad_norm`)
- Add `OverfitDetectionCallback` (priority=75) — copy from `$TEMPLATE_DIR/callbacks.py`; constructor params `warmup_epochs: int`, `window_size: int`; hooks: `on_epoch_end` — after warmup, checks if recent train losses are decreasing while recent val losses are increasing; if so, computes gap ratio and writes to `trainer._overfit_gap_ratio`
- Update `WandbLoggingCallback.on_epoch_end`: after `log_dict.update(metrics)`, check for `trainer._max_grad_norm` and `trainer._overfit_gap_ratio` attributes (using `hasattr`) and add them to `log_dict` if present

### util.py

**Action:** Modify existing file

Changes needed:
- Add `GradientMonitorCallback`, `OverfitDetectionCallback` to imports from `callbacks`
- In `Trainer.__init__`: add `self._max_grad_norm: float | None = None`, `self._overfit_gap_ratio: float | None = None`, and `self._loss_prediction: float | None = None` (the last is written by `LossPredictionCallback` from M1 but should be initialized here for safety)
- In `run()` callbacks list: insert `GradientMonitorCallback()` after `NaNDetectionCallback()` (from M2) and `OverfitDetectionCallback()` after `LossPredictionCallback` (from M1). The full callback order should now be: `OptimizerModeCallback` (M1), `NaNDetectionCallback` (M2), `GradientMonitorCallback` (M5), `LossPredictionCallback` (M1), `OverfitDetectionCallback` (M5), then logging/checkpoint callbacks

### cli.py

**Action:** Modify existing file

Changes needed:
- Add `preflight` command — copy from `$TEMPLATE_DIR/cli.py`; signature: `preflight(run_config: str, device: str, json_output: bool)`; runs tier-1 (structural), tier-2 (runtime), and tier-3 (semantic) validation, then prints a rich table or JSON report
- Add `hpo-report` command — copy from `$TEMPLATE_DIR/cli.py`; command name `"hpo-report"`, function name `hpo_report`; signature params: `db`, `study_name`, `opt_config`, `top_k`, `json_output`; loads Optuna study and prints top trials table

For both commands, copy the full implementations from `$TEMPLATE_DIR/cli.py` — they are ~120 lines each with non-trivial rich table formatting.

---

## M6: Dual Logging + TUI Monitor + Provenance (v6 → v7)

**Detect:** `grep -c "class CSVLoggingCallback" callbacks.py` returns 0

### config.py

**Action:** Modify existing file

Changes needed:
- Add `logging: str` field to `RunConfig` with default `"wandb"`
- Add `monitor: str` field to `RunConfig` with default `"val_loss"` — forward-compatible field for checkpoint monitoring configuration at the run level
- In `RunConfig.__post_init__`: validate that `self.logging` is one of `"wandb"` or `"tui"`; raise `ValueError` otherwise

### callbacks.py

**Action:** Modify existing file

Changes needed:
- Add `import csv` at the top if not already present
- Add `CSVLoggingCallback` (priority=81) — copy from `$TEMPLATE_DIR/callbacks.py`; constructor param `csv_path: str`; maintains `_fieldnames` list and `_rows` list; hook `on_epoch_end` collects epoch, train_loss, val_loss, lr, metrics, `_max_grad_norm`, `_overfit_gap_ratio`, `_loss_prediction` from trainer; handles dynamic column expansion by rewriting the entire CSV when new keys appear (`_flush_all()` method), otherwise appends a single row
- Add `TUILoggingCallback` (priority=80) — copy from `$TEMPLATE_DIR/callbacks.py`; no constructor params; hook `on_train_begin` prints a header separator; hook `on_epoch_end` prints a formatted line with epoch counter, train/val loss, lr, any float metrics, grad norm, and predicted final loss; hook `on_train_end` prints a footer separator
- Add `LatestModelCallback` (priority=96) — copy from `$TEMPLATE_DIR/callbacks.py`; constructor param `save_path: str`; hook `on_epoch_end` calls `torch.save(trainer.model.state_dict(), self.save_path)`

### provenance.py

**Action:** Create from `$TEMPLATE_DIR/provenance.py`

Key components:
- `capture_environment() -> dict` — captures Python version, platform, hostname, torch version, CUDA availability and version, GPU names and memory, cuDNN version, numpy version, git commit hash and dirty flag, relevant environment variables (`PYTHONHASHSEED`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `CUDA_VISIBLE_DEVICES`, `CUBLAS_WORKSPACE_CONFIG`)
- `compute_config_hash(config) -> str` — SHA-256 of canonical JSON serialization of the config dataclass (sorted keys, excludes `_frozen`)
- `capture_run_metadata(model, device, start_time, end_time) -> dict` — captures training time, device, total/trainable parameter counts, peak GPU memory if applicable
- `save_provenance(run_path, config, model, device, start_time, end_time)` — calls the above three functions; writes `env_snapshot.yaml` and `run_metadata.yaml` (with `config_hash` included) to `run_path`

### cli.py

**Action:** Modify existing file

Changes needed:
- Add `doctor` command — copy from `$TEMPLATE_DIR/cli.py`; no params; calls `capture_environment()` from `provenance` and renders a rich table with rows for Python version, PyTorch version, CUDA status (with per-GPU name/memory rows), wandb login status, and required package versions (`torch`, `numpy`, `optuna`, `wandb`, `tqdm`, `rich`, `beaupy`, `scienceplots`)
- Add `monitor` command — copy from `$TEMPLATE_DIR/cli.py`; params: `path` (optional path to `metrics.csv` or its parent dir), `interval` (refresh interval in ms); resolves the Rust binary at `tools/monitor/target/release/training-monitor` relative to `__file__`; if binary is missing, builds it via `cargo build --release` in `tools/monitor/`; if `path` is None, auto-detects the most recently modified `metrics.csv` under `runs/`; launches the binary as a subprocess

### util.py

**Action:** Modify existing file

Changes needed:
- Add `CSVLoggingCallback`, `TUILoggingCallback`, `LatestModelCallback` to imports from `callbacks`
- Add `from provenance import save_provenance, compute_config_hash` import
- In `run()`: read `use_wandb = run_config.logging == "wandb"` before building the callbacks list
- In `run()` callbacks list: conditionally append `WandbLoggingCallback()` if `use_wandb`, else `TUILoggingCallback()`
- In `run()` callbacks list: always append `CSVLoggingCallback(f"{run_path}/metrics.csv")` and `LatestModelCallback(f"{run_path}/latest_model.pt")`
- In `run()`: record `start_time = time.time()` before `trainer.train(...)` and `end_time = time.time()` after
- In `run()`: call `save_provenance(run_path, run_config, model, device, start_time, end_time)` after saving `model.pt`

### configs/run_template.yaml

**Action:** Add field from `$TEMPLATE_DIR/configs/run_template.yaml`

Fields to add:
- `logging:` — value is either `wandb` or `tui`; add after the `device:` field

### tools/monitor/ (Rust TUI)

**Action:** Copy directory from `$TEMPLATE_DIR/tools/monitor/`

The directory contains a Rust workspace with a binary crate `training-monitor`. It reads `metrics.csv` in real time and renders a live terminal dashboard with loss curves, learning rate, and gradient norm plots. The `monitor` CLI command builds it automatically on first use via `cargo build --release`. Requires a Rust toolchain (`rustc` / `cargo`).

### Legacy filename migration

If your project has scripts or code referencing `train_log.csv`, rename **all** occurrences to `metrics.csv`. The CSV logging callback writes to `metrics.csv` (not configurable). Affected locations:
- `util.py` — any hardcoded `train_log.csv` paths must become `metrics.csv`
- User scripts for plotting or analysis
- The Rust TUI monitor (`tools/monitor/`) expects `metrics.csv`

### CSV column consistency (important)

The CSV column names must be consistent across **three** locations:
1. `CSVLoggingCallback._collect_metrics()` in `callbacks.py` — defines which columns are written
2. `_list_runs()` in `cli.py` — defines the `known` set for detecting extra columns: `{"epoch", "train_loss", "val_loss", "lr", "max_grad_norm", "predicted_final_loss"}`
3. `KNOWN` const in `tools/monitor/src/main.rs` — the Rust TUI's known column set

If you add custom metrics to the CSV, you must update all three locations or the TUI monitor and `--list` command will misclassify them.

---

## M7: TUI Monitor Tabs + CLI Enhancements (v7 → current)

**Detect:** `grep -c "def update_skills" cli.py` returns 0

### tools/monitor/src/main.rs

**Action:** Replace from `$TEMPLATE_DIR/tools/monitor/src/main.rs`

Changes from previous version:
- `MetricRow` struct gains `extras: Vec<Option<f64>>` field for dynamic CSV columns
- `App` struct gains `active_tab: usize` and `extra_columns: Vec<String>` for tab navigation
- `try_reload()` now detects all CSV columns beyond the known set (`epoch`, `train_loss`, `val_loss`, `lr`, `max_grad_norm`, `predicted_final_loss`) and parses them into `extras`
- New rendering: `render_tab_bar()` shows tab names with `ratatui::widgets::Tabs`, `render_overview()` replaces the old monolithic render, `render_extra_tab()` draws a full-height chart for one extra column
- `render_generic_chart()` handles data with negative values (symlog transform)
- `handle_event()` adds `KeyCode::Right`/`KeyCode::Left`/`KeyCode::Tab`/`KeyCode::BackTab` for tab switching
- Status bar help text conditionally shows `←→: tabs` when extra columns are present
- After replacing the file, rebuild: `cd tools/monitor && cargo build --release`

### cli.py

**Action:** Modify existing file

Changes needed:

1. **Add `_list_runs()` helper** — copy from `$TEMPLATE_DIR/cli.py`, search for `def _list_runs`. Place it immediately before the existing `def monitor` function. This function scans `runs/**/metrics.csv`, parses path components (project/group/seed), reads epoch count and extra CSV columns, returns list sorted by modification time.

2. **Update existing `monitor` command** — find the existing `def monitor(` function and:
   - Add parameter: `list_runs: bool = typer.Option(False, "--list", help="List available runs")`
   - Add the `--list` branch at the top of the function body (before the monitor binary resolution). Copy the `if list_runs:` block from `$TEMPLATE_DIR/cli.py` — it calls `_list_runs()`, renders a Rich table, then uses `beaupy.select()` for interactive run selection and launches the monitor for the chosen run

3. **Add `update_skills` command** — copy from `$TEMPLATE_DIR/cli.py`, search for `def update_skills`. Place it before `if __name__ == "__main__":`. This command uses `import shutil` and `from pathlib import Path` (imported inside the function). It resolves template skills at `.claude/skills/` relative to `__file__`, installs as symlink by default (or copy with `--copy`) to `~/.claude/skills/`, and supports `--uninstall`

---

## How to Apply Migrations to a Real Project

For projects that forked or diverged significantly from the template (custom `util.py`, custom callbacks, etc.):

1. Do NOT overwrite `util.py` wholesale — the user likely has custom `load_data()`, analysis functions, and other logic.
2. Do apply targeted edits to `config.py` (add fields, add methods).
3. Do append new callbacks to `callbacks.py` — do not remove existing custom ones.
4. Do add new CLI commands to `cli.py` by appending, not replacing.
5. Do add the `data:` field to all YAML run configs.
6. Do update imports in `util.py` and wire new callbacks into `run()`.

### Divergence handling

If the user has heavily modified a file:
1. Read both the user's version and the corresponding file in `$TEMPLATE_DIR`.
2. Identify which template changes are absent from the user's version.
3. Apply only the missing changes, preserving the user's customizations.
4. Verify with `pytest` and the `preflight` command after migration.
