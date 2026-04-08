# HPO TUI Monitor Design Spec

## Goal

Add an HPO (hyperparameter optimization) monitoring mode to the existing Rust-based TUI monitor. Reads the Optuna SQLite DB in real-time to show trial progress, parameter analysis, and per-trial training curves — all from the terminal.

## Architecture

The existing `training-monitor` binary gains a `--hpo <db_path>` flag. When present, it enters HPO mode with a separate App struct (`HpoApp`) and dedicated rendering pipeline. Shared chart utilities (symlog, axis formatting, log scale) are extracted to a common module. CLI integration via `cli.py monitor --hpo`.

## Tech Stack

- **Rust** with ratatui 0.30, crossterm, rusqlite
- **Python** CLI: typer (existing cli.py)
- **Data sources**: Optuna SQLite DB + per-trial `metrics.csv` files

---

## Module Structure

```
tools/monitor/src/
├── main.rs          # CLI parsing (clap), mode dispatch
├── training.rs      # Existing training monitor (extracted from main.rs)
├── hpo.rs           # HpoApp struct, event loop, data polling
├── hpo_views.rs     # Tab rendering functions for HPO mode
└── charts.rs        # Shared: symlog, log scale, axis labels, chart helpers
```

### Dependency changes (Cargo.toml)

Add `rusqlite` with `bundled` feature (bundles SQLite, no system dependency needed).

---

## Data Flow

### SQLite Queries

Polling interval: **1 second**.

**Trial list (Overview, Parameters, Trials):**
```sql
SELECT t.trial_id, t.number, t.state,
       tv.value AS objective_value,
       t.datetime_start, t.datetime_complete
FROM trials t
LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id AND tv.objective_id = 0
WHERE t.study_id = ?
ORDER BY t.number
```

**Parameters (Parameters tab, Trials tab):**
```sql
SELECT trial_id, param_name, param_value, distribution_json
FROM trial_params WHERE study_id = ?
```

`distribution_json` provides parameter type (int/float/categorical) for axis formatting and log scale decisions.

**Best trial group_name (for CSV path resolution):**
```sql
SELECT value_json FROM trial_user_attributes
WHERE trial_id = ? AND key = 'group_name'
```

**Study ID resolution:**
```sql
SELECT study_id FROM studies WHERE study_name = ?
```

If only one study exists in the DB, auto-select it.

### metrics.csv Path Resolution

From DB: best trial's `group_name` user attribute → scan `runs/{project}/{group_name}/seed_*/metrics.csv` → select the seed with the lowest final val_loss. Same logic applies when viewing a specific trial from the Trials tab.

### Data Cache

```rust
struct HpoData {
    study_id: i64,
    trials: Vec<TrialInfo>,         // number, state, value, params, duration
    param_names: Vec<String>,       // ordered parameter names
    param_types: Vec<ParamType>,    // Int, Float, Categorical
    best_trial_idx: Option<usize>,  // index into trials vec
}

struct TrialInfo {
    number: i64,
    state: TrialState,              // Complete, Pruned, Fail, Running
    value: Option<f64>,
    params: Vec<Option<f64>>,       // parallel to param_names
    categorical_labels: Vec<Option<String>>,
    duration_secs: Option<f64>,
    group_name: Option<String>,
}
```

Polling strategy: check trial count on each tick. Full refresh only when count changes or a trial's state transitions.

---

## Tab Layout

### Tab 0: Overview

- **Top 30%**: Status summary — `Completed: N | Pruned: N | Failed: N | Running: N | Total: N/M`
- **Bottom 70%**: Best value convergence curve
  - X-axis: trial number
  - Y-axis: cumulative best objective value at each completed trial
  - Only completed trials plotted
  - `l` key toggles Y-axis log scale

### Tab 1: Parameters

- **Grid layout** based on parameter count:
  - 1-2 params: 1 row x 2 cols
  - 3-4 params: 2 rows x 2 cols
  - 5-6 params: 3 rows x 2 cols
  - 7+ params: 4 rows x 2 cols (remaining truncated with note)
- **Each cell**: scatter plot
  - X-axis: parameter value
  - Y-axis: objective value
  - Point colors by trial state: Complete=green, Pruned=yellow, Failed=red
  - Categorical params: discrete X positions with labels
- **Key bindings**:
  - `l`: toggle Y-axis log scale (all cells simultaneously)
  - `x`: toggle X-axis log scale (all cells simultaneously; ignored for categorical params)

### Tab 2: Best Trial

- Identical layout to existing training monitor's Overview
  - Loss chart (train/val), LR chart, gradient norm chart
  - Rendered from best trial's `metrics.csv`
- Header: `Best Trial #N — val_loss: X.XXe-X`
- Automatically switches when a new best trial appears
- `l` key toggles loss chart log scale

### Tab 3: Trials

**Table mode (default):**
- Columns: `#`, `State`, `Value`, all search_space parameters, `Duration`
- Sorted by objective value (best first); trials without a value (RUNNING, FAIL) sorted to the bottom by trial number
- State colors: COMPLETE=green, PRUNED=yellow, FAIL=red, RUNNING=cyan
- `↑↓` keys move row cursor
- `Enter` enters detail view for selected trial

**Detail mode (after Enter):**
- Same layout as Tab 2 (training monitor Overview), but for the selected trial
- Header: `Trial #N — val_loss: X.XXe-X`
- `Esc` returns to table mode
- `l` key toggles loss chart log scale
- If trial is PRUNED, curves show up to the pruning epoch

---

## State Management

```rust
struct HpoApp {
    // Data
    db_path: PathBuf,
    hpo_data: HpoData,
    best_trial_metrics: Vec<MetricRow>,   // CSV data for best trial
    detail_metrics: Vec<MetricRow>,       // CSV data for trial detail view

    // UI state
    active_tab: usize,                    // 0-3
    log_y: bool,                          // Y-axis log (Overview, Parameters, Best Trial)
    log_x: bool,                          // X-axis log (Parameters only)
    selected_trial: usize,               // Trials table cursor position
    trial_detail_mode: bool,             // true = detail view, false = table view

    // Timing
    last_db_poll: Instant,
    last_csv_modified: Option<SystemTime>,
}
```

### MetricRow (shared with training mode)

```rust
struct MetricRow {
    epoch: f64,
    train_loss: f64,
    val_loss: f64,
    lr: f64,
    max_grad_norm: Option<f64>,
    predicted_final_loss: Option<f64>,
    extras: Vec<Option<f64>>,
}
```

---

## Key Bindings

### Global (all tabs)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `←→` / `Tab` / `BackTab` | Switch tabs |

### Per-tab

| Tab | Key | Action |
|-----|-----|--------|
| Overview | `l` | Y-axis log toggle |
| Parameters | `l` | Y-axis (objective) log toggle |
| Parameters | `x` | X-axis (param value) log toggle |
| Best Trial | `l` | Loss chart log toggle |
| Trials (table) | `↑↓` | Row selection |
| Trials (table) | `Enter` | Enter detail view |
| Trials (detail) | `Esc` | Return to table |
| Trials (detail) | `l` | Loss chart log toggle |

### Footer Status Bar

```
Tab: Overview | Trials: 12/20 completed | Best: 3.45e-4 (trial #7) | q: quit  l: log  ←→: tabs
```

Parameters tab extends with `x: log-x`. Trials detail mode shows `Esc: back`.

---

## CLI Integration

### cli.py changes

Add `--hpo` flag to existing `monitor` command:

```python
@app.command()
def monitor(
    path: str = typer.Argument(None),
    interval: int = typer.Option(500, help="Refresh interval in ms"),
    hpo: bool = typer.Option(False, "--hpo", help="HPO monitor mode"),
    list_runs: bool = typer.Option(False, "--list", help="List available runs"),
):
```

When `--hpo`:
1. Search for `.db` files in the project directory
2. If one found, use it directly
3. If multiple, present beaupy selection (same pattern as `hpo-report` command's DB auto-detection)
4. Resolve the monitor binary (build if needed, same as existing)
5. Launch: `training-monitor --hpo <db_path>`

### training-monitor CLI (clap)

```
training-monitor [CSV_PATH]              # Training mode
training-monitor --hpo <DB_PATH>         # HPO mode
training-monitor --hpo <DB_PATH> --study <STUDY_NAME>  # Explicit study
```

`--study` is optional. If omitted and DB has exactly one study, auto-select. If multiple studies, print list and exit with error message.

---

## Shared Chart Utilities (charts.rs)

Extracted from current `main.rs`:

- `symlog(x, c)` / `symlog_inv(y, c)` — symmetric log transform for non-positive data
- `format_axis_label(value, log_mode)` — scientific notation for axis ticks
- `compute_chart_bounds(data, log_mode)` — min/max with padding
- `render_line_chart(frame, area, datasets, x_bounds, y_bounds, title, log_mode)` — generic line chart
- `render_scatter_chart(frame, area, points, x_bounds, y_bounds, title, log_x, log_y)` — scatter plot (new)
- `EXTRA_COLORS` palette — reused for parameter scatter colors

---

## Migration Doc Update

After implementation, add **M8** to the migration docs:

- Detection: `grep -c "hpo" tools/monitor/src/main.rs`
- Files: `tools/monitor/` (Rust source changes), `cli.py` (--hpo flag)
- Dependencies: `rusqlite` added to Cargo.toml
