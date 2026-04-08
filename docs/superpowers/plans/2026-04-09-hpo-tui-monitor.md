# HPO TUI Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--hpo` mode to the Rust TUI monitor that reads Optuna SQLite DB in real-time, showing trial convergence, parameter scatter plots, best trial curves, and a trials table with detail view.

**Architecture:** Split the existing monolithic `main.rs` (~840 lines) into modules: `charts.rs` (shared math/rendering), `training.rs` (existing training mode), `hpo.rs` (HPO data/events), `hpo_views.rs` (HPO rendering). Add `rusqlite` for DB access. CLI integration via `cli.py monitor --hpo`.

**Tech Stack:** Rust (ratatui 0.30, rusqlite, clap 4, csv), Python (typer)

**Spec:** `docs/superpowers/specs/2026-04-09-hpo-tui-monitor-design.md`

---

## File Structure

```
tools/monitor/
├── Cargo.toml              # Modify: add rusqlite dependency
└── src/
    ├── main.rs             # Modify: slim to CLI + mod declarations + dispatch
    ├── charts.rs           # Create: shared MetricRow, LogState, math, chart renderers
    ├── training.rs         # Create: App struct, training-specific rendering, run_training()
    ├── hpo.rs              # Create: HpoApp, HpoData, TrialInfo, DB polling, run_hpo()
    └── hpo_views.rs        # Create: HPO tab renderers (overview, params, best trial, trials)

cli.py                      # Modify: add --hpo flag to monitor command
```

---

### Task 1: Extract shared utilities to `charts.rs`

**Files:**
- Create: `tools/monitor/src/charts.rs`
- Modify: `tools/monitor/src/main.rs` (remove extracted code, add `mod charts; use charts::*;`)

This task extracts reusable code from `main.rs` into a shared module. Both training and HPO modes will import from here.

- [ ] **Step 1: Create `charts.rs` with extracted types and functions**

Create `tools/monitor/src/charts.rs`. Move these items from `main.rs`:

| Item | Source lines in main.rs | Notes |
|------|------------------------|-------|
| `MetricRow` struct | 35-44 | Add `pub` to struct and fields |
| `symlog()` | 254-256 | Add `pub` |
| `symlog_inv()` | 258-260 | Add `pub` |
| `bounds_xy()` | 744-762 | Add `pub` |
| `min_max_y()` | 764-771 | Add `pub` |
| `make_labels()` | 773-784 | Add `pub` |
| `make_inv_log10_labels()` | 787-794 | Add `pub` |
| `EXTRA_COLORS` | 369-376 | Add `pub` |
| symlog tests | 797-840 | Move to `charts.rs` `#[cfg(test)]` |

Then add these NEW items to `charts.rs`:

```rust
use std::path::Path;

/// Log-scale state — shared between training and HPO modes.
pub struct LogState {
    pub log_scale: bool,
    pub has_nonpositive: bool,
    pub symlog_c: f64,
}

impl LogState {
    /// Compute log state from a set of MetricRows.
    pub fn from_metrics(metrics: &[MetricRow]) -> Self {
        let has_nonpositive = metrics
            .iter()
            .any(|m| m.train_loss <= 0.0 || m.val_loss <= 0.0);

        let symlog_c = if has_nonpositive {
            let min_abs = metrics
                .iter()
                .flat_map(|m| [m.train_loss.abs(), m.val_loss.abs()])
                .filter(|&v| v > 1e-15)
                .fold(f64::INFINITY, f64::min);
            if min_abs.is_finite() { min_abs } else { 1.0 }
        } else {
            1.0
        };

        Self { log_scale: false, has_nonpositive, symlog_c }
    }

    /// Transform a loss value for the Y axis.
    pub fn loss_y(&self, v: f64) -> f64 {
        if !self.log_scale {
            v
        } else if self.has_nonpositive {
            symlog(v, self.symlog_c)
        } else {
            v.max(1e-20).log10()
        }
    }

    /// Inverse transform: transformed Y → original loss value.
    pub fn loss_y_inv(&self, y: f64) -> f64 {
        if !self.log_scale {
            y
        } else if self.has_nonpositive {
            symlog_inv(y, self.symlog_c)
        } else {
            10f64.powf(y)
        }
    }

    pub fn loss_title(&self) -> &'static str {
        match (self.log_scale, self.has_nonpositive) {
            (false, _) => " Loss Curves ",
            (true, false) => " Loss Curves (log\u{2081}\u{2080}) ",
            (true, true) => " Loss Curves (symlog\u{2081}\u{2080}) ",
        }
    }

    pub fn make_loss_labels(&self, lo: f64, hi: f64, n: usize) -> Vec<String> {
        (0..=n)
            .map(|i| {
                let y = lo + (hi - lo) * i as f64 / n as f64;
                if self.log_scale {
                    format!("{:.1e}", self.loss_y_inv(y))
                } else {
                    format!("{:.1e}", y)
                }
            })
            .collect()
    }
}

/// Load MetricRow data from a CSV file. Returns empty vec on error.
pub fn load_metrics_csv(path: &Path) -> (Vec<MetricRow>, Vec<String>) {
    let Ok(mut rdr) = csv::Reader::from_path(path) else {
        return (Vec::new(), Vec::new());
    };
    let Ok(headers) = rdr.headers().cloned() else {
        return (Vec::new(), Vec::new());
    };

    const KNOWN: &[&str] = &[
        "epoch", "train_loss", "val_loss", "lr",
        "max_grad_norm", "predicted_final_loss",
    ];
    let extra_columns: Vec<String> = headers
        .iter()
        .filter(|h| !KNOWN.contains(&h))
        .map(String::from)
        .collect();

    let mut rows = Vec::new();
    for result in rdr.records() {
        let Ok(record) = result else { continue };
        let get = |name: &str| -> Option<f64> {
            headers
                .iter()
                .position(|h| h == name)
                .and_then(|i| record.get(i))
                .and_then(|v| v.parse().ok())
        };
        let extras: Vec<Option<f64>> = extra_columns.iter().map(|col| get(col)).collect();
        rows.push(MetricRow {
            epoch: get("epoch").unwrap_or(0.0),
            train_loss: get("train_loss").unwrap_or(0.0),
            val_loss: get("val_loss").unwrap_or(0.0),
            lr: get("lr").unwrap_or(0.0),
            max_grad_norm: get("max_grad_norm"),
            predicted_final_loss: get("predicted_final_loss"),
            extras,
        });
    }
    (rows, extra_columns)
}
```

Also move these rendering functions from `main.rs` to `charts.rs`, generalizing them to take `&LogState` instead of `&App`:

- `render_loss_chart(frame, metrics: &[MetricRow], log_state: &LogState, area)` — adapted from lines 470-565
- `render_lr_chart(frame, metrics: &[MetricRow], log_scale: bool, area)` — adapted from lines 567-586
- `render_grad_chart(frame, metrics: &[MetricRow], log_scale: bool, area)` — adapted from lines 588-605
- `render_positive_chart(frame, log_scale: bool, area, data, name, y_title, color)` — adapted from lines 609-675
- `render_generic_chart(frame, log_state: &LogState, area, data, name, y_title, color)` — adapted from lines 414-468
- `render_training_overview(frame, metrics: &[MetricRow], log_state: &LogState, area)` — adapted from lines 343-367

Each function signature changes from `(frame, app: &App, ...)` to `(frame, metrics/log_state, ...)`. The body logic is identical — just replace `app.loss_y(x)` with `log_state.loss_y(x)`, `app.log_scale` with `log_state.log_scale`, etc.

- [ ] **Step 2: Update `main.rs` to use `charts` module**

At the top of `main.rs`, add:
```rust
mod charts;
use charts::*;
```

Remove all moved items from `main.rs`. The `App` struct and its methods, `render()`, `render_tab_bar()`, `render_extra_tab()`, `render_status()`, `run_app()`, and `main()` remain.

Update `App` to use `LogState`:
- Replace fields `log_scale`, `has_nonpositive`, `symlog_c` with `log_state: LogState`
- Update `App::loss_y()` → `self.log_state.loss_y()`
- Update `App::loss_y_inv()` → `self.log_state.loss_y_inv()`
- Update `App::loss_title()` → `self.log_state.loss_title()`
- Update `App::make_loss_labels()` → `self.log_state.make_loss_labels()`
- In `try_reload()`: use `load_metrics_csv()` then update `self.log_state` from result

Update rendering calls to pass `&self.metrics` and `&self.log_state` instead of `&self`.

- [ ] **Step 3: Verify build and tests**

Run: `cd tools/monitor && cargo test`
Expected: All 5 symlog tests pass. `cargo build` succeeds.

- [ ] **Step 4: Commit**

```bash
git add tools/monitor/src/charts.rs tools/monitor/src/main.rs
git commit -m "refactor(monitor): extract shared charts module from main.rs"
```

---

### Task 2: Extract training mode to `training.rs`

**Files:**
- Create: `tools/monitor/src/training.rs`
- Modify: `tools/monitor/src/main.rs` (slim down to CLI + dispatch)

- [ ] **Step 1: Create `training.rs`**

Move from `main.rs` to `training.rs`:
- `App` struct and all its `impl` methods (`new`, `try_reload`, `handle_event`, `elapsed_secs`, `total_tabs`)
- `render()`, `render_tab_bar()`, `render_extra_tab()`, `render_status()`
- `run_app()` renamed to `run_training()`

The public API is:
```rust
use crate::charts::*;
// ... ratatui imports ...

pub struct App { /* fields */ }

pub fn run_training(csv_path: PathBuf, interval: Duration) -> Result<()> {
    let mut app = App::new(csv_path, interval);
    let mut terminal = ratatui::init();
    let result = run_app(&mut terminal, &mut app);
    ratatui::restore();
    result
}

fn run_app(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> Result<()> {
    loop {
        app.try_reload();
        terminal.draw(|f| render(f, app))?;
        if event::poll(app.interval)? && app.handle_event(event::read()?) {
            return Ok(());
        }
    }
}
// ... remaining private rendering functions ...
```

- [ ] **Step 2: Slim `main.rs` to CLI + dispatch**

`main.rs` becomes:

```rust
mod charts;
mod training;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "training-monitor")]
#[command(about = "Real-time TUI training monitor for pytorch_template")]
struct Cli {
    /// Path to metrics.csv or its parent directory
    path: Option<PathBuf>,

    /// Refresh interval in milliseconds
    #[arg(short, long, default_value_t = 500)]
    interval: u64,

    /// HPO monitor mode: path to Optuna SQLite database
    #[arg(long)]
    hpo: Option<PathBuf>,

    /// Study name (required if DB contains multiple studies)
    #[arg(long)]
    study: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(_db_path) = cli.hpo {
        // HPO mode — will be implemented in Task 4
        anyhow::bail!("HPO mode not yet implemented");
    } else {
        // Training mode
        let path = cli.path.expect("CSV path required in training mode");
        let csv_path = if path.is_dir() {
            path.join("metrics.csv")
        } else {
            path
        };
        training::run_training(csv_path, Duration::from_millis(cli.interval))
    }
}
```

Note: `path` changes from required positional to `Option<PathBuf>` since HPO mode doesn't need it. Training mode validates it's present.

- [ ] **Step 3: Verify build and tests**

Run: `cd tools/monitor && cargo test && cargo build`
Expected: All tests pass. Binary works: `cargo run -- path/to/metrics.csv`

- [ ] **Step 4: Commit**

```bash
git add tools/monitor/src/training.rs tools/monitor/src/main.rs
git commit -m "refactor(monitor): extract training mode to training.rs, add --hpo CLI flag"
```

---

### Task 3: Add rusqlite + create `hpo.rs`

**Files:**
- Modify: `tools/monitor/Cargo.toml` (add rusqlite)
- Create: `tools/monitor/src/hpo.rs`
- Modify: `tools/monitor/src/main.rs` (add `mod hpo;`, wire up HPO dispatch)

- [ ] **Step 1: Add rusqlite dependency**

In `tools/monitor/Cargo.toml`, add to `[dependencies]`:
```toml
rusqlite = { version = "0.35", features = ["bundled"] }
```

- [ ] **Step 2: Create `hpo.rs` with data structures and DB polling**

```rust
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime},
};

use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use rusqlite::Connection;

use crate::charts::{self, LogState, MetricRow};

mod views;
pub use views::render_hpo;

// ── Trial Types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialState {
    Running,
    Complete,
    Pruned,
    Fail,
    Waiting,
}

impl TrialState {
    fn from_optuna(s: &str) -> Self {
        match s {
            "COMPLETE" => Self::Complete,
            "RUNNING" => Self::Running,
            "PRUNED" => Self::Pruned,
            "FAIL" => Self::Fail,
            "WAITING" => Self::Waiting,
            _ => Self::Fail,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Running => "RUNNING",
            Self::Complete => "COMPLETE",
            Self::Pruned => "PRUNED",
            Self::Fail => "FAIL",
            Self::Waiting => "WAITING",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrialInfo {
    pub number: i64,
    pub state: TrialState,
    pub value: Option<f64>,
    pub params: Vec<Option<f64>>,
    pub categorical_labels: Vec<Option<String>>,
    pub duration_secs: Option<f64>,
    pub group_name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    Float,
    Int,
    Categorical,
}

#[derive(Debug)]
pub struct HpoData {
    pub study_id: i64,
    pub study_name: String,
    pub trials: Vec<TrialInfo>,
    pub param_names: Vec<String>,
    pub param_types: Vec<ParamType>,
    pub best_trial_idx: Option<usize>,
    pub total_trials_expected: Option<usize>,
}

impl HpoData {
    fn empty(study_id: i64, study_name: String) -> Self {
        Self {
            study_id,
            study_name,
            trials: Vec::new(),
            param_names: Vec::new(),
            param_types: Vec::new(),
            best_trial_idx: None,
            total_trials_expected: None,
        }
    }
}

// ── App State ─────────────────────────────────────────────────────────────

pub struct HpoApp {
    // Data
    pub db_path: PathBuf,
    pub conn: Connection,
    pub hpo_data: HpoData,
    pub best_trial_metrics: Vec<MetricRow>,
    pub best_trial_log: LogState,
    pub detail_metrics: Vec<MetricRow>,
    pub detail_log: LogState,

    // UI state
    pub active_tab: usize,
    pub log_y: bool,
    pub log_x: bool,
    pub selected_trial: usize,
    pub trial_detail_mode: bool,

    // Timing
    pub interval: Duration,
    last_db_poll: Instant,
    last_trial_count: usize,
    last_best_csv_modified: Option<SystemTime>,
    last_detail_csv_modified: Option<SystemTime>,
}

impl HpoApp {
    pub fn total_tabs(&self) -> usize {
        4 // Overview, Parameters, Best Trial, Trials
    }

    pub fn tab_names(&self) -> Vec<&'static str> {
        vec!["Overview", "Parameters", "Best Trial", "Trials"]
    }

    fn poll_db(&mut self) {
        if self.last_db_poll.elapsed() < self.interval {
            return;
        }
        self.last_db_poll = Instant::now();

        // Check if trial count changed
        let count: i64 = self.conn
            .query_row(
                "SELECT COUNT(*) FROM trials WHERE study_id = ?1",
                [self.hpo_data.study_id],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if count as usize == self.last_trial_count && self.last_trial_count > 0 {
            // Check for state transitions (RUNNING → COMPLETE etc.)
            let any_running: bool = self.conn
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM trials WHERE study_id = ?1 AND state = 'RUNNING')",
                    [self.hpo_data.study_id],
                    |row| row.get(0),
                )
                .unwrap_or(false);
            if !any_running && self.hpo_data.trials.iter().all(|t| t.state != TrialState::Running) {
                return; // No changes
            }
        }
        self.last_trial_count = count as usize;

        self.reload_trials();
        self.reload_best_trial_csv();
    }

    fn reload_trials(&mut self) {
        // Load param names if first time
        if self.hpo_data.param_names.is_empty() {
            self.load_param_names();
        }

        // Load trials
        let mut stmt = match self.conn.prepare(
            "SELECT t.trial_id, t.number,
                    CASE t.state
                        WHEN 'COMPLETE' THEN 'COMPLETE'
                        WHEN 'RUNNING' THEN 'RUNNING'
                        WHEN 'PRUNED' THEN 'PRUNED'
                        WHEN 'FAIL' THEN 'FAIL'
                        WHEN 'WAITING' THEN 'WAITING'
                        ELSE t.state
                    END as state_str,
                    tv.value,
                    t.datetime_start, t.datetime_complete
             FROM trials t
             LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id AND tv.objective_id = 0
             WHERE t.study_id = ?1
             ORDER BY t.number"
        ) {
            Ok(s) => s,
            Err(_) => return,
        };

        let trial_rows: Vec<(i64, i64, String, Option<f64>, Option<String>, Option<String>)> =
            match stmt.query_map([self.hpo_data.study_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<f64>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                ))
            }) {
                Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
                Err(_) => return,
            };

        let mut trials = Vec::new();
        for (trial_id, number, state_str, value, dt_start, dt_complete) in &trial_rows {
            let duration_secs = match (dt_start, dt_complete) {
                (Some(s), Some(e)) => {
                    // Optuna stores ISO format
                    parse_duration_secs(s, e)
                }
                _ => None,
            };

            // Load params for this trial
            let params = self.load_trial_params(*trial_id);
            let categorical_labels = self.load_trial_categorical_labels(*trial_id);

            // Load group_name
            let group_name: Option<String> = self.conn
                .query_row(
                    "SELECT value_json FROM trial_user_attributes WHERE trial_id = ?1 AND key = 'group_name'",
                    [trial_id],
                    |row| row.get::<_, String>(0),
                )
                .ok()
                .map(|v| v.trim_matches('"').to_string());

            trials.push(TrialInfo {
                number: *number,
                state: TrialState::from_optuna(state_str),
                value: *value,
                params,
                categorical_labels,
                duration_secs,
                group_name,
            });
        }

        // Find best trial
        let best_idx = trials
            .iter()
            .enumerate()
            .filter(|(_, t)| t.state == TrialState::Complete && t.value.is_some())
            .min_by(|(_, a), (_, b)| {
                a.value.unwrap().partial_cmp(&b.value.unwrap()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        self.hpo_data.trials = trials;
        self.hpo_data.best_trial_idx = best_idx;

        // Clamp selected_trial
        if !self.hpo_data.trials.is_empty() && self.selected_trial >= self.hpo_data.trials.len() {
            self.selected_trial = self.hpo_data.trials.len() - 1;
        }
    }

    fn load_param_names(&mut self) {
        let mut stmt = match self.conn.prepare(
            "SELECT DISTINCT param_name, distribution_json
             FROM trial_params WHERE study_id = ?1
             ORDER BY param_name"
        ) {
            Ok(s) => s,
            Err(_) => return,
        };

        let rows: Vec<(String, String)> = match stmt.query_map(
            [self.hpo_data.study_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        ) {
            Ok(r) => r.filter_map(|r| r.ok()).collect(),
            Err(_) => return,
        };

        self.hpo_data.param_names.clear();
        self.hpo_data.param_types.clear();
        for (name, dist_json) in rows {
            let ptype = if dist_json.contains("Categorical") {
                ParamType::Categorical
            } else if dist_json.contains("Int") {
                ParamType::Int
            } else {
                ParamType::Float
            };
            if !self.hpo_data.param_names.contains(&name) {
                self.hpo_data.param_names.push(name);
                self.hpo_data.param_types.push(ptype);
            }
        }
    }

    fn load_trial_params(&self, trial_id: i64) -> Vec<Option<f64>> {
        self.hpo_data.param_names
            .iter()
            .map(|name| {
                self.conn
                    .query_row(
                        "SELECT param_value FROM trial_params WHERE trial_id = ?1 AND param_name = ?2",
                        rusqlite::params![trial_id, name],
                        |row| row.get::<_, f64>(0),
                    )
                    .ok()
            })
            .collect()
    }

    fn load_trial_categorical_labels(&self, trial_id: i64) -> Vec<Option<String>> {
        self.hpo_data.param_names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                if self.hpo_data.param_types.get(i) == Some(&ParamType::Categorical) {
                    self.conn
                        .query_row(
                            "SELECT param_value FROM trial_params WHERE trial_id = ?1 AND param_name = ?2",
                            rusqlite::params![trial_id, name],
                            |row| row.get::<_, String>(0),
                        )
                        .ok()
                } else {
                    None
                }
            })
            .collect()
    }

    fn reload_best_trial_csv(&mut self) {
        let Some(best_idx) = self.hpo_data.best_trial_idx else { return };
        let Some(group_name) = self.hpo_data.trials[best_idx].group_name.clone() else { return };
        load_trial_csv(&self.db_path, &group_name, self.log_y, &mut self.best_trial_metrics, &mut self.best_trial_log, &mut self.last_best_csv_modified);
    }

    pub fn reload_detail_trial_csv(&mut self) {
        if !self.trial_detail_mode { return }
        let sorted = self.sorted_trial_indices();
        if self.selected_trial >= sorted.len() { return }
        let group_name = match &self.hpo_data.trials[sorted[self.selected_trial]].group_name {
            Some(g) => g.clone(),
            None => return,
        };
        load_trial_csv(&self.db_path, &group_name, self.log_y, &mut self.detail_metrics, &mut self.detail_log, &mut self.last_detail_csv_modified);
    }
}

/// Free function to avoid borrow-checker conflicts (db_path + metrics are both fields of HpoApp).
fn load_trial_csv(
    db_path: &Path,
    group_name: &str,
    log_y: bool,
    metrics: &mut Vec<MetricRow>,
    log_state: &mut LogState,
    last_modified: &mut Option<SystemTime>,
) {
    // Find the project name from DB path
    let project = db_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

        // Scan for metrics.csv under runs/{project}/{group_name}/*/metrics.csv
        let pattern = format!("runs/{}/{}/**/metrics.csv", project, group_name);
        let candidates: Vec<PathBuf> = glob::glob(&pattern)
            .into_iter()
            .flatten()
            .filter_map(|r| r.ok())
            .collect();

    let csv_path = if candidates.len() == 1 {
        candidates[0].clone()
    } else if candidates.len() > 1 {
        // Pick the seed with the most recent modification
        candidates.into_iter()
            .max_by_key(|p| fs::metadata(p).and_then(|m| m.modified()).ok())
            .unwrap()
    } else {
        return;
    };

    // Check modification time
    if let Ok(meta) = fs::metadata(&csv_path) {
        if let Ok(modified) = meta.modified() {
            if *last_modified == Some(modified) {
                return;
            }
            *last_modified = Some(modified);
        }
    }

    let (rows, _extras) = charts::load_metrics_csv(&csv_path);
    *log_state = LogState::from_metrics(&rows);
    log_state.log_scale = log_y;
    *metrics = rows;
}

    /// Return trial indices sorted by objective value (best first), valueless at bottom.
    pub fn sorted_trial_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.hpo_data.trials.len()).collect();
        indices.sort_by(|&a, &b| {
            let ta = &self.hpo_data.trials[a];
            let tb = &self.hpo_data.trials[b];
            match (ta.value, tb.value) {
                (Some(va), Some(vb)) => va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => ta.number.cmp(&tb.number),
            }
        });
        indices
    }

    /// Compute best-so-far convergence curve: (trial_number, best_value_so_far)
    pub fn convergence_curve(&self) -> Vec<(f64, f64)> {
        let mut best_so_far = f64::INFINITY;
        let mut curve = Vec::new();
        for trial in &self.hpo_data.trials {
            if trial.state == TrialState::Complete {
                if let Some(v) = trial.value {
                    best_so_far = best_so_far.min(v);
                    curve.push((trial.number as f64, best_so_far));
                }
            }
        }
        curve
    }

    pub fn handle_event(&mut self, ev: Event) -> bool {
        if let Event::Key(key) = ev {
            if key.kind != KeyEventKind::Press {
                return false;
            }
            match key.code {
                KeyCode::Char('q') => return true,
                KeyCode::Char('l') => {
                    self.log_y = !self.log_y;
                    self.best_trial_log.log_scale = self.log_y;
                    self.detail_log.log_scale = self.log_y;
                }
                KeyCode::Char('x') => {
                    if self.active_tab == 1 {
                        self.log_x = !self.log_x;
                    }
                }
                KeyCode::Esc => {
                    if self.trial_detail_mode {
                        self.trial_detail_mode = false;
                    } else {
                        return true;
                    }
                }
                KeyCode::Tab | KeyCode::Right => {
                    if !self.trial_detail_mode {
                        let n = self.total_tabs();
                        self.active_tab = (self.active_tab + 1) % n;
                    }
                }
                KeyCode::BackTab | KeyCode::Left => {
                    if !self.trial_detail_mode {
                        let n = self.total_tabs();
                        self.active_tab = (self.active_tab + n - 1) % n;
                    }
                }
                KeyCode::Up => {
                    if self.active_tab == 3 && !self.trial_detail_mode && self.selected_trial > 0 {
                        self.selected_trial -= 1;
                    }
                }
                KeyCode::Down => {
                    if self.active_tab == 3 && !self.trial_detail_mode {
                        let max = self.sorted_trial_indices().len().saturating_sub(1);
                        if self.selected_trial < max {
                            self.selected_trial += 1;
                        }
                    }
                }
                KeyCode::Enter => {
                    if self.active_tab == 3 && !self.trial_detail_mode {
                        self.trial_detail_mode = true;
                        self.last_detail_csv_modified = None;
                        self.reload_detail_trial_csv();
                    }
                }
                _ => {}
            }
        }
        false
    }
}

fn parse_duration_secs(start: &str, end: &str) -> Option<f64> {
    let parse = |s: &str| -> Option<f64> {
        // Optuna stores "YYYY-MM-DD HH:MM:SS.ffffff" or ISO format
        // Simple approach: parse as seconds since we only need duration
        let s = s.replace('T', " ");
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() < 2 { return None; }
        let date_parts: Vec<&str> = parts[0].split('-').collect();
        let time_str = parts[1];
        let time_parts: Vec<&str> = time_str.split(':').collect();
        if date_parts.len() < 3 || time_parts.len() < 3 { return None; }

        let d: i64 = date_parts[0].parse::<i64>().ok()? * 365 * 24 * 3600
            + date_parts[1].parse::<i64>().ok()? * 30 * 24 * 3600
            + date_parts[2].parse::<i64>().ok()? * 24 * 3600;
        let sec_str = time_parts[2];
        let secs: f64 = sec_str.parse().ok()?;
        let h: f64 = time_parts[0].parse().ok()?;
        let m: f64 = time_parts[1].parse().ok()?;
        Some(d as f64 + h * 3600.0 + m * 60.0 + secs)
    };

    let s = parse(start)?;
    let e = parse(end)?;
    Some((e - s).max(0.0))
}

// ── Public entry point ────────────────────────────────────────────────────

pub fn run_hpo(db_path: PathBuf, study_name: Option<String>, interval: Duration) -> Result<()> {
    let conn = Connection::open(&db_path)?;

    // Resolve study
    let (study_id, resolved_name) = if let Some(name) = study_name {
        let id: i64 = conn.query_row(
            "SELECT study_id FROM studies WHERE study_name = ?1",
            [&name],
            |row| row.get(0),
        )?;
        (id, name)
    } else {
        // Auto-detect: require exactly one study
        let mut stmt = conn.prepare("SELECT study_id, study_name FROM studies")?;
        let studies: Vec<(i64, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        match studies.len() {
            0 => anyhow::bail!("No studies found in database"),
            1 => studies.into_iter().next().unwrap(),
            n => {
                let names: Vec<&str> = studies.iter().map(|(_, n)| n.as_str()).collect();
                anyhow::bail!(
                    "Multiple studies found ({}). Use --study to specify: {:?}",
                    n, names
                );
            }
        }
    };

    let hpo_data = HpoData::empty(study_id, resolved_name);
    let dummy_log = LogState { log_scale: false, has_nonpositive: false, symlog_c: 1.0 };

    let mut app = HpoApp {
        db_path,
        conn,
        hpo_data,
        best_trial_metrics: Vec::new(),
        best_trial_log: LogState { log_scale: false, has_nonpositive: false, symlog_c: 1.0 },
        detail_metrics: Vec::new(),
        detail_log: LogState { log_scale: false, has_nonpositive: false, symlog_c: 1.0 },
        active_tab: 0,
        log_y: false,
        log_x: false,
        selected_trial: 0,
        trial_detail_mode: false,
        interval,
        last_db_poll: Instant::now() - interval, // Force immediate poll
        last_trial_count: 0,
        last_best_csv_modified: None,
        last_detail_csv_modified: None,
    };

    let mut terminal = ratatui::init();
    let result = run_hpo_app(&mut terminal, &mut app);
    ratatui::restore();
    result
}

fn run_hpo_app(terminal: &mut ratatui::DefaultTerminal, app: &mut HpoApp) -> Result<()> {
    loop {
        app.poll_db();
        if app.trial_detail_mode {
            app.reload_detail_trial_csv();
        }
        terminal.draw(|f| render_hpo(f, app))?;
        if event::poll(app.interval)? && app.handle_event(event::read()?) {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trial_state_from_optuna() {
        assert_eq!(TrialState::from_optuna("COMPLETE"), TrialState::Complete);
        assert_eq!(TrialState::from_optuna("PRUNED"), TrialState::Pruned);
        assert_eq!(TrialState::from_optuna("RUNNING"), TrialState::Running);
        assert_eq!(TrialState::from_optuna("FAIL"), TrialState::Fail);
        assert_eq!(TrialState::from_optuna("UNKNOWN"), TrialState::Fail);
    }

    #[test]
    fn sorted_indices_best_first() {
        let data = HpoData {
            study_id: 1,
            study_name: "test".into(),
            trials: vec![
                TrialInfo { number: 0, state: TrialState::Complete, value: Some(0.5), params: vec![], categorical_labels: vec![], duration_secs: None, group_name: None },
                TrialInfo { number: 1, state: TrialState::Complete, value: Some(0.1), params: vec![], categorical_labels: vec![], duration_secs: None, group_name: None },
                TrialInfo { number: 2, state: TrialState::Running, value: None, params: vec![], categorical_labels: vec![], duration_secs: None, group_name: None },
                TrialInfo { number: 3, state: TrialState::Complete, value: Some(0.3), params: vec![], categorical_labels: vec![], duration_secs: None, group_name: None },
            ],
            param_names: vec![],
            param_types: vec![],
            best_trial_idx: Some(1),
            total_trials_expected: None,
        };

        let mut app = HpoApp {
            db_path: PathBuf::new(),
            conn: Connection::open_in_memory().unwrap(),
            hpo_data: data,
            best_trial_metrics: vec![],
            best_trial_log: LogState { log_scale: false, has_nonpositive: false, symlog_c: 1.0 },
            detail_metrics: vec![],
            detail_log: LogState { log_scale: false, has_nonpositive: false, symlog_c: 1.0 },
            active_tab: 0,
            log_y: false,
            log_x: false,
            selected_trial: 0,
            trial_detail_mode: false,
            interval: Duration::from_secs(1),
            last_db_poll: Instant::now(),
            last_trial_count: 0,
            last_best_csv_modified: None,
            last_detail_csv_modified: None,
        };

        let sorted = app.sorted_trial_indices();
        // Best first: trial 1 (0.1), trial 3 (0.3), trial 0 (0.5), trial 2 (None → bottom)
        assert_eq!(sorted, vec![1, 3, 0, 2]);
    }

    #[test]
    fn convergence_curve_tracks_best() {
        let data = HpoData {
            study_id: 1,
            study_name: "test".into(),
            trials: vec![
                TrialInfo { number: 0, state: TrialState::Complete, value: Some(0.5), params: vec![], categorical_labels: vec![], duration_secs: None, group_name: None },
                TrialInfo { number: 1, state: TrialState::Pruned, value: None, params: vec![], categorical_labels: vec![], duration_secs: None, group_name: None },
                TrialInfo { number: 2, state: TrialState::Complete, value: Some(0.3), params: vec![], categorical_labels: vec![], duration_secs: None, group_name: None },
                TrialInfo { number: 3, state: TrialState::Complete, value: Some(0.4), params: vec![], categorical_labels: vec![], duration_secs: None, group_name: None },
            ],
            param_names: vec![],
            param_types: vec![],
            best_trial_idx: Some(2),
            total_trials_expected: None,
        };

        let app = HpoApp {
            db_path: PathBuf::new(),
            conn: Connection::open_in_memory().unwrap(),
            hpo_data: data,
            best_trial_metrics: vec![],
            best_trial_log: LogState { log_scale: false, has_nonpositive: false, symlog_c: 1.0 },
            detail_metrics: vec![],
            detail_log: LogState { log_scale: false, has_nonpositive: false, symlog_c: 1.0 },
            active_tab: 0,
            log_y: false,
            log_x: false,
            selected_trial: 0,
            trial_detail_mode: false,
            interval: Duration::from_secs(1),
            last_db_poll: Instant::now(),
            last_trial_count: 0,
            last_best_csv_modified: None,
            last_detail_csv_modified: None,
        };

        let curve = app.convergence_curve();
        // trial 0: 0.5, trial 2: 0.3, trial 3: still 0.3
        assert_eq!(curve, vec![(0.0, 0.5), (2.0, 0.3), (3.0, 0.3)]);
    }
}
```

- [ ] **Step 3: Add glob dependency to Cargo.toml**

```toml
glob = "0.3"
```

- [ ] **Step 4: Wire up HPO mode in `main.rs`**

Update `main.rs`:
```rust
mod charts;
mod training;
mod hpo;

// ... existing CLI struct ...

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(db_path) = cli.hpo {
        hpo::run_hpo(db_path, cli.study, Duration::from_millis(cli.interval))
    } else {
        let path = cli.path.expect("CSV path required in training mode");
        let csv_path = if path.is_dir() { path.join("metrics.csv") } else { path };
        training::run_training(csv_path, Duration::from_millis(cli.interval))
    }
}
```

- [ ] **Step 5: Create stub `hpo/views.rs`**

Create `tools/monitor/src/hpo/views.rs` with a placeholder render function so hpo.rs compiles:

```rust
use ratatui::Frame;
use super::HpoApp;

pub fn render_hpo(frame: &mut Frame, app: &HpoApp) {
    // Stub — will be implemented in Task 4
    let area = frame.area();
    frame.render_widget(
        ratatui::widgets::Paragraph::new("HPO Monitor — loading..."),
        area,
    );
}
```

Note: `hpo.rs` uses `mod views;` so the views file must be at `src/hpo/views.rs`. Restructure: rename `hpo.rs` → `hpo/mod.rs` and place `views.rs` inside `hpo/`.

- [ ] **Step 6: Verify build and tests**

Run: `cd tools/monitor && cargo test && cargo build`
Expected: All tests pass (symlog + new hpo tests). Binary builds.

- [ ] **Step 7: Commit**

```bash
git add tools/monitor/
git commit -m "feat(monitor): add HPO data structures, DB polling, and event handling"
```

---

### Task 4: Create HPO tab renderers (`hpo/views.rs`)

**Files:**
- Modify: `tools/monitor/src/hpo/views.rs` (replace stub)

- [ ] **Step 1: Implement full `views.rs`**

Replace the stub with the complete implementation. This file contains all 4 tab renderers + status bar + scatter chart.

```rust
use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Style, Stylize},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Chart, Dataset, GraphType, Paragraph, Row, Table, Tabs},
    Frame,
};

use crate::charts::{
    self, render_training_overview, min_max_y, make_labels, make_inv_log10_labels,
    EXTRA_COLORS, symlog, symlog_inv,
};
use super::{HpoApp, TrialState, ParamType};

pub fn render_hpo(frame: &mut Frame, app: &HpoApp) {
    let [tab_area, content_area, status_area] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(0),
        Constraint::Length(3),
    ])
    .areas(frame.area());

    render_tab_bar(frame, app, tab_area);

    match app.active_tab {
        0 => render_overview(frame, app, content_area),
        1 => render_parameters(frame, app, content_area),
        2 => render_best_trial(frame, app, content_area),
        3 => {
            if app.trial_detail_mode {
                render_trial_detail(frame, app, content_area);
            } else {
                render_trials_table(frame, app, content_area);
            }
        }
        _ => {}
    }

    render_hpo_status(frame, app, status_area);
}

fn render_tab_bar(frame: &mut Frame, app: &HpoApp, area: Rect) {
    let titles: Vec<Line> = app.tab_names()
        .into_iter()
        .map(|n| Line::from(format!(" {} ", n)))
        .collect();

    let tabs = Tabs::new(titles)
        .select(app.active_tab)
        .style(Style::new().fg(Color::DarkGray))
        .highlight_style(Style::new().fg(Color::Cyan).bold())
        .divider(Span::raw("│"));

    frame.render_widget(tabs, area);
}

// ── Tab 0: Overview ───────────────────────────────────────────────────────

fn render_overview(frame: &mut Frame, app: &HpoApp, area: Rect) {
    let [summary_area, chart_area] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
    ])
    .areas(area);

    // Summary counts
    let trials = &app.hpo_data.trials;
    let completed = trials.iter().filter(|t| t.state == TrialState::Complete).count();
    let pruned = trials.iter().filter(|t| t.state == TrialState::Pruned).count();
    let failed = trials.iter().filter(|t| t.state == TrialState::Fail).count();
    let running = trials.iter().filter(|t| t.state == TrialState::Running).count();
    let total = trials.len();

    let summary = Line::from(vec![
        Span::styled(format!(" Completed: {} ", completed), Style::new().fg(Color::Green).bold()),
        Span::raw("│"),
        Span::styled(format!(" Pruned: {} ", pruned), Style::new().fg(Color::Yellow)),
        Span::raw("│"),
        Span::styled(format!(" Failed: {} ", failed), Style::new().fg(Color::Red)),
        Span::raw("│"),
        Span::styled(format!(" Running: {} ", running), Style::new().fg(Color::Cyan)),
        Span::raw("│"),
        Span::styled(format!(" Total: {} ", total), Style::new().fg(Color::White)),
    ]);

    frame.render_widget(
        Paragraph::new(summary).block(Block::bordered().title(" Study Status ")),
        summary_area,
    );

    // Convergence curve
    let curve = app.convergence_curve();
    if curve.is_empty() {
        frame.render_widget(
            Paragraph::new(" Waiting for completed trials...")
                .block(Block::bordered().title(" Best Value Convergence ")),
            chart_area,
        );
        return;
    }

    let data: Vec<(f64, f64)> = if app.log_y {
        curve.iter().map(|&(x, y)| (x, y.max(1e-20).log10())).collect()
    } else {
        curve.clone()
    };

    let x_max = data.last().map(|(x, _)| x + 1.0).unwrap_or(1.0);
    let (y_min, y_max) = min_max_y(&data);
    let y_pad = (y_max - y_min).max(1e-10) * 0.1;
    let y_lo = y_min - y_pad;
    let y_hi = y_max + y_pad;

    let datasets = vec![Dataset::default()
        .name("best")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(Color::Green))
        .data(&data)];

    let title = if app.log_y {
        " Best Value Convergence (log\u{2081}\u{2080}) "
    } else {
        " Best Value Convergence "
    };

    let x_labels = make_labels(0.0, x_max, 5, false);
    let y_labels = if app.log_y {
        make_inv_log10_labels(y_lo, y_hi, 5)
    } else {
        make_labels(y_lo, y_hi, 5, true)
    };

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(title))
        .x_axis(
            Axis::default()
                .title("trial")
                .style(Style::new().fg(Color::DarkGray))
                .bounds([0.0, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("objective")
                .style(Style::new().fg(Color::DarkGray))
                .bounds([y_lo, y_hi])
                .labels(y_labels),
        );

    frame.render_widget(chart, chart_area);
}

// ── Tab 1: Parameters ─────────────────────────────────────────────────────

fn render_parameters(frame: &mut Frame, app: &HpoApp, area: Rect) {
    let n_params = app.hpo_data.param_names.len();
    if n_params == 0 {
        frame.render_widget(
            Paragraph::new(" No parameters found").block(Block::bordered().title(" Parameters ")),
            area,
        );
        return;
    }

    // Grid layout
    let (rows_count, cols_count) = match n_params {
        1 => (1, 1),
        2 => (1, 2),
        3..=4 => (2, 2),
        5..=6 => (3, 2),
        _ => (4, 2),
    };

    let row_constraints: Vec<Constraint> = (0..rows_count)
        .map(|_| Constraint::Ratio(1, rows_count as u32))
        .collect();
    let col_constraints: Vec<Constraint> = (0..cols_count)
        .map(|_| Constraint::Ratio(1, cols_count as u32))
        .collect();

    let row_areas = Layout::vertical(row_constraints).split(area);

    for row_idx in 0..rows_count {
        let col_areas = Layout::horizontal(col_constraints.clone()).split(row_areas[row_idx]);
        for col_idx in 0..cols_count {
            let param_idx = row_idx * cols_count + col_idx;
            if param_idx < n_params {
                render_param_scatter(frame, app, col_areas[col_idx], param_idx);
            }
        }
    }
}

fn render_param_scatter(frame: &mut Frame, app: &HpoApp, area: Rect, param_idx: usize) {
    let param_name = &app.hpo_data.param_names[param_idx];
    let param_type = app.hpo_data.param_types[param_idx];

    // Collect points: (param_value, objective_value, state)
    let mut complete_points: Vec<(f64, f64)> = Vec::new();
    let mut pruned_points: Vec<(f64, f64)> = Vec::new();
    let mut failed_points: Vec<(f64, f64)> = Vec::new();

    for trial in &app.hpo_data.trials {
        let Some(param_val) = trial.params.get(param_idx).copied().flatten() else { continue };
        let Some(obj_val) = trial.value else { continue };

        let x = if app.log_x && param_type != ParamType::Categorical {
            param_val.abs().max(1e-20).log10()
        } else {
            param_val
        };
        let y = if app.log_y { obj_val.max(1e-20).log10() } else { obj_val };

        match trial.state {
            TrialState::Complete => complete_points.push((x, y)),
            TrialState::Pruned => pruned_points.push((x, y)),
            TrialState::Fail => failed_points.push((x, y)),
            _ => {}
        }
    }

    let all_points: Vec<&(f64, f64)> = complete_points.iter()
        .chain(pruned_points.iter())
        .chain(failed_points.iter())
        .collect();

    if all_points.is_empty() {
        frame.render_widget(
            Paragraph::new(" No data")
                .block(Block::bordered().title(format!(" {} ", param_name))),
            area,
        );
        return;
    }

    let x_min = all_points.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let x_max = all_points.iter().map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max);
    let y_min = all_points.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = all_points.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min).max(1e-10) * 0.15;
    let y_pad = (y_max - y_min).max(1e-10) * 0.1;
    let x_lo = x_min - x_pad;
    let x_hi = x_max + x_pad;
    let y_lo = y_min - y_pad;
    let y_hi = y_max + y_pad;

    let mut datasets = Vec::new();
    if !complete_points.is_empty() {
        datasets.push(Dataset::default()
            .name("ok")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Scatter)
            .style(Style::new().fg(Color::Green))
            .data(&complete_points));
    }
    if !pruned_points.is_empty() {
        datasets.push(Dataset::default()
            .name("pruned")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Scatter)
            .style(Style::new().fg(Color::Yellow))
            .data(&pruned_points));
    }
    if !failed_points.is_empty() {
        datasets.push(Dataset::default()
            .name("fail")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Scatter)
            .style(Style::new().fg(Color::Red))
            .data(&failed_points));
    }

    let title = format!(" {} ", param_name);
    let x_labels = if app.log_x && param_type != ParamType::Categorical {
        make_inv_log10_labels(x_lo, x_hi, 3)
    } else {
        make_labels(x_lo, x_hi, 3, true)
    };
    let y_labels = if app.log_y {
        make_inv_log10_labels(y_lo, y_hi, 3)
    } else {
        make_labels(y_lo, y_hi, 3, true)
    };

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(title))
        .x_axis(
            Axis::default()
                .title(param_name.as_str())
                .style(Style::new().fg(Color::DarkGray))
                .bounds([x_lo, x_hi])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("objective")
                .style(Style::new().fg(Color::DarkGray))
                .bounds([y_lo, y_hi])
                .labels(y_labels),
        );

    frame.render_widget(chart, area);
}

// ── Tab 2: Best Trial ─────────────────────────────────────────────────────

fn render_best_trial(frame: &mut Frame, app: &HpoApp, area: Rect) {
    if app.best_trial_metrics.is_empty() {
        let msg = if let Some(idx) = app.hpo_data.best_trial_idx {
            let t = &app.hpo_data.trials[idx];
            format!(" Best Trial #{} — waiting for metrics.csv...", t.number)
        } else {
            " No completed trials yet".to_string()
        };
        frame.render_widget(
            Paragraph::new(msg).block(Block::bordered().title(" Best Trial ")),
            area,
        );
        return;
    }

    let [header_area, chart_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(area);

    if let Some(idx) = app.hpo_data.best_trial_idx {
        let t = &app.hpo_data.trials[idx];
        let val_str = t.value.map(|v| format!("{:.4e}", v)).unwrap_or_default();
        frame.render_widget(
            Paragraph::new(format!(" Best Trial #{} — val_loss: {}", t.number, val_str))
                .style(Style::new().fg(Color::Green).bold()),
            header_area,
        );
    }

    render_training_overview(frame, &app.best_trial_metrics, &app.best_trial_log, chart_area);
}

// ── Tab 3: Trials ─────────────────────────────────────────────────────────

fn render_trials_table(frame: &mut Frame, app: &HpoApp, area: Rect) {
    let sorted = app.sorted_trial_indices();
    if sorted.is_empty() {
        frame.render_widget(
            Paragraph::new(" No trials yet").block(Block::bordered().title(" Trials ")),
            area,
        );
        return;
    }

    // Build header
    let mut header_cells = vec!["#", "State", "Value"];
    let param_names: Vec<&str> = app.hpo_data.param_names.iter().map(|s| s.as_str()).collect();
    header_cells.extend(param_names.iter());
    header_cells.push("Duration");

    let header = Row::new(header_cells.iter().map(|h| *h))
        .style(Style::new().bold().fg(Color::White))
        .bottom_margin(0);

    // Build rows
    let rows: Vec<Row> = sorted
        .iter()
        .enumerate()
        .map(|(display_idx, &trial_idx)| {
            let trial = &app.hpo_data.trials[trial_idx];
            let state_color = match trial.state {
                TrialState::Complete => Color::Green,
                TrialState::Pruned => Color::Yellow,
                TrialState::Fail => Color::Red,
                TrialState::Running => Color::Cyan,
                TrialState::Waiting => Color::DarkGray,
            };

            let mut cells: Vec<String> = vec![
                format!("{}", trial.number),
                trial.state.label().to_string(),
                trial.value.map(|v| format!("{:.4e}", v)).unwrap_or("-".into()),
            ];

            for (i, _name) in app.hpo_data.param_names.iter().enumerate() {
                let val = trial.params.get(i).copied().flatten();
                if app.hpo_data.param_types[i] == ParamType::Categorical {
                    let label = trial.categorical_labels.get(i).and_then(|l| l.as_deref());
                    cells.push(label.unwrap_or("-").to_string());
                } else {
                    cells.push(val.map(|v| format!("{:.4e}", v)).unwrap_or("-".into()));
                }
            }

            let dur = trial.duration_secs
                .map(|s| {
                    if s < 60.0 { format!("{:.0}s", s) }
                    else if s < 3600.0 { format!("{:.0}m{:.0}s", s / 60.0, s % 60.0) }
                    else { format!("{:.1}h", s / 3600.0) }
                })
                .unwrap_or("-".into());
            cells.push(dur);

            let style = if display_idx == app.selected_trial {
                Style::new().fg(state_color).reversed()
            } else {
                Style::new().fg(state_color)
            };

            Row::new(cells).style(style)
        })
        .collect();

    let widths: Vec<Constraint> = {
        let n = header_cells.len();
        let mut w = vec![Constraint::Length(4)]; // #
        w.push(Constraint::Length(9));            // State
        w.push(Constraint::Length(12));           // Value
        for _ in &app.hpo_data.param_names {
            w.push(Constraint::Length(12));        // Params
        }
        w.push(Constraint::Length(8));            // Duration
        w
    };

    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::bordered().title(" Trials "))
        .row_highlight_style(Style::new().reversed());

    frame.render_widget(table, area);
}

fn render_trial_detail(frame: &mut Frame, app: &HpoApp, area: Rect) {
    let sorted = app.sorted_trial_indices();
    if app.selected_trial >= sorted.len() { return }
    let trial = &app.hpo_data.trials[sorted[app.selected_trial]];

    let [header_area, chart_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(area);

    let val_str = trial.value.map(|v| format!("{:.4e}", v)).unwrap_or("-".into());
    let state_color = match trial.state {
        TrialState::Complete => Color::Green,
        TrialState::Pruned => Color::Yellow,
        TrialState::Fail => Color::Red,
        TrialState::Running => Color::Cyan,
        _ => Color::DarkGray,
    };
    frame.render_widget(
        Paragraph::new(format!(
            " Trial #{} — {} — val_loss: {} │ Esc: back",
            trial.number, trial.state.label(), val_str
        ))
        .style(Style::new().fg(state_color).bold()),
        header_area,
    );

    if app.detail_metrics.is_empty() {
        frame.render_widget(
            Paragraph::new(" Waiting for metrics.csv data...")
                .block(Block::bordered()),
            chart_area,
        );
    } else {
        render_training_overview(frame, &app.detail_metrics, &app.detail_log, chart_area);
    }
}

// ── Status Bar ────────────────────────────────────────────────────────────

fn render_hpo_status(frame: &mut Frame, app: &HpoApp, area: Rect) {
    let completed = app.hpo_data.trials.iter().filter(|t| t.state == TrialState::Complete).count();
    let total = app.hpo_data.trials.len();

    let best_str = app.hpo_data.best_trial_idx
        .and_then(|i| {
            let t = &app.hpo_data.trials[i];
            t.value.map(|v| format!("Best: {:.4e} (trial #{})", v, t.number))
        })
        .unwrap_or("Best: -".into());

    let tab_name = app.tab_names()[app.active_tab];

    let log_tags = match (app.active_tab, app.log_y, app.log_x) {
        (1, true, true) => " [LOG-Y] [LOG-X]",
        (1, true, false) => " [LOG-Y]",
        (1, false, true) => " [LOG-X]",
        (_, true, _) => " [LOG]",
        _ => "",
    };

    let extra_keys = match app.active_tab {
        1 => " │ x: log-x",
        3 if app.trial_detail_mode => " │ Esc: back",
        3 => " │ ↑↓: select │ Enter: detail",
        _ => "",
    };

    let text = format!(
        " Tab: {} │ Trials: {}/{} completed │ {} │ Study: {}\n q: quit │ l: log │ ←→: tabs{}{}",
        tab_name, completed, total, best_str,
        app.hpo_data.study_name,
        extra_keys, log_tags,
    );

    frame.render_widget(
        Paragraph::new(text)
            .block(Block::bordered().title(" Status ").fg(Color::DarkGray)),
        area,
    );
}
```

- [ ] **Step 2: Verify build**

Run: `cd tools/monitor && cargo build`
Expected: Compiles successfully.

- [ ] **Step 3: Commit**

```bash
git add tools/monitor/src/hpo/
git commit -m "feat(monitor): implement HPO tab renderers (overview, params, best trial, trials)"
```

---

### Task 5: Update `cli.py` — `--hpo` flag

**Files:**
- Modify: `cli.py` (monitor command)

- [ ] **Step 1: Add `--hpo` flag to monitor command**

Update the `monitor` function signature and add the HPO branch. In `cli.py`, modify the `monitor` command:

```python
@app.command()
def monitor(
    path: str = typer.Argument(
        None, help="Path to metrics.csv or its parent directory"
    ),
    interval: int = typer.Option(500, help="Refresh interval in milliseconds"),
    list_runs: bool = typer.Option(False, "--list", help="List available runs"),
    hpo: bool = typer.Option(False, "--hpo", help="HPO monitor mode"),
):
```

Add the HPO branch **after** the `if list_runs:` block and **before** the existing monitor binary resolution:

```python
    if hpo:
        import glob as glob_mod

        # Auto-detect DB
        db_files = glob_mod.glob("*.db")
        if len(db_files) == 1:
            db_path = db_files[0]
        elif not db_files:
            console.print("[red]No .db files found. Run HPO first.[/red]")
            raise typer.Exit(code=1)
        else:
            from beaupy import select
            selected = select(db_files, cursor_index=0, return_index=True)
            if selected is None:
                return
            db_path = db_files[selected]

        console.print(f"[dim]Using DB: {db_path}[/dim]")

        # Build monitor if needed (same as training mode)
        monitor_bin = os.path.join(os.path.dirname(__file__), "tools", "monitor", "target", "release", "training-monitor")
        if not os.path.exists(monitor_bin):
            console.print("[yellow]Monitor binary not found. Building...[/yellow]")
            cargo_dir = os.path.join(os.path.dirname(__file__), "tools", "monitor")
            result = subprocess.run(["cargo", "build", "--release"], cwd=cargo_dir)
            if result.returncode != 0:
                console.print("[red]Failed to build monitor. Install Rust: https://rustup.rs[/red]")
                raise typer.Exit(code=1)

        try:
            subprocess.run([monitor_bin, "--hpo", db_path, "--interval", str(interval)])
        except KeyboardInterrupt:
            pass
        return
```

- [ ] **Step 2: Verify preflight still passes**

Run: `uv run python -m cli preflight configs/run_template.yaml --device cpu`
Expected: All checks pass (no functional changes to training).

- [ ] **Step 3: Commit**

```bash
git add cli.py
git commit -m "feat(cli): add --hpo flag to monitor command for HPO monitoring"
```

---

### Task 6: Build, integration test, rebuild binary

**Files:** No new files — verification task.

- [ ] **Step 1: Release build**

```bash
cd tools/monitor && cargo build --release
```

Expected: Build succeeds. Binary at `tools/monitor/target/release/training-monitor`.

- [ ] **Step 2: Verify training mode still works**

```bash
# If there's a metrics.csv available:
./tools/monitor/target/release/training-monitor runs/*/metrics.csv --interval 1000
# Or just verify --help works:
./tools/monitor/target/release/training-monitor --help
```

Expected: Help shows both `[PATH]` positional arg and `--hpo <HPO>` option.

- [ ] **Step 3: Verify HPO mode launches**

```bash
# If an Optuna DB exists:
./tools/monitor/target/release/training-monitor --hpo *.db
# Or via CLI:
uv run python -m cli monitor --hpo
```

Expected: HPO TUI launches showing the 4 tabs with real trial data from the DB.

- [ ] **Step 4: Run all tests**

```bash
cd tools/monitor && cargo test
uv run pytest tests/ -x -q
```

Expected: All Rust and Python tests pass.

- [ ] **Step 5: Commit**

```bash
git add tools/monitor/
git commit -m "build(monitor): release build with HPO mode"
```

---

### Task 7: Update migration docs (M8)

**Files:**
- Modify: `.claude/skills/pytorch-migrate/SKILL.md`
- Modify: `.claude/skills/pytorch-migrate/references/migrations.md`

- [ ] **Step 1: Add M8 detection to SKILL.md**

Add to the detection table (after the `update_skills` row):

```markdown
| `tools/monitor/src/hpo/mod.rs` exists | File exists | v7 (pre-HPO-monitor) |
```

Add to the detection script:

```bash
test -f tools/monitor/src/hpo/mod.rs && echo "1" || echo "0"
```

Add to the migration summary table:

```markdown
| M8: HPO TUI Monitor | v7→current | Add HPO mode to Rust monitor, `--hpo` CLI flag, `rusqlite` dependency |
```

- [ ] **Step 2: Add M8 section to migrations.md**

Add after the M7 section:

```markdown
---

## M8: HPO TUI Monitor (v7 → current)

**Detect:** `test -f tools/monitor/src/hpo/mod.rs` fails

### tools/monitor/

**Action:** Replace from `$TEMPLATE_DIR/tools/monitor/`

Changes from previous version:
- `Cargo.toml` gains `rusqlite = { version = "0.35", features = ["bundled"] }` and `glob = "0.3"` dependencies
- `src/main.rs` slimmed to CLI parsing + mode dispatch; adds `--hpo <DB_PATH>` and `--study <NAME>` flags
- `src/charts.rs` — new module: shared `MetricRow`, `LogState`, symlog, label formatting, chart rendering functions extracted from old `main.rs`
- `src/training.rs` — new module: training mode `App` struct and rendering, extracted from old `main.rs`
- `src/hpo/mod.rs` — new module: `HpoApp`, `HpoData`, `TrialInfo` structs, SQLite DB polling via `rusqlite`, event handling
- `src/hpo/views.rs` — new module: 4-tab HPO rendering (Overview convergence curve, Parameters scatter grid, Best Trial training curves, Trials table with detail view)
- After replacing, rebuild: `cd tools/monitor && cargo build --release`

### cli.py

**Action:** Modify existing file

Changes needed:
- Add `hpo: bool = typer.Option(False, "--hpo", help="HPO monitor mode")` parameter to existing `monitor` command
- Add HPO branch: auto-detect `.db` files (beaupy selection if multiple), launch `training-monitor --hpo <db_path>`
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/pytorch-migrate/
git commit -m "docs(migrate): add M8 HPO TUI monitor migration"
```
