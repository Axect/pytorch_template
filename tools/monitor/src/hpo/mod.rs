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
#[allow(dead_code)]
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
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM trials WHERE study_id = ?1",
                [self.hpo_data.study_id],
                |row| row.get(0),
            )
            .unwrap_or(0);

        if count as usize == self.last_trial_count && self.last_trial_count > 0 {
            // Check for state transitions (RUNNING → COMPLETE etc.)
            let any_running: bool = self
                .conn
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM trials WHERE study_id = ?1 AND state = 'RUNNING')",
                    [self.hpo_data.study_id],
                    |row| row.get(0),
                )
                .unwrap_or(false);

            if !any_running
                && self
                    .hpo_data
                    .trials
                    .iter()
                    .all(|t| t.state != TrialState::Running)
            {
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
             ORDER BY t.number",
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
                (Some(s), Some(e)) => parse_duration_secs(s, e),
                _ => None,
            };

            // Load params for this trial
            let params = self.load_trial_params(*trial_id);
            let categorical_labels = self.load_trial_categorical_labels(*trial_id);

            // Load group_name
            let group_name: Option<String> = self
                .conn
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
                a.value
                    .unwrap()
                    .partial_cmp(&b.value.unwrap())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        self.hpo_data.trials = trials;
        self.hpo_data.best_trial_idx = best_idx;

        // Clamp selected_trial
        if !self.hpo_data.trials.is_empty()
            && self.selected_trial >= self.hpo_data.trials.len()
        {
            self.selected_trial = self.hpo_data.trials.len() - 1;
        }
    }

    fn load_param_names(&mut self) {
        let mut stmt = match self.conn.prepare(
            "SELECT DISTINCT param_name, distribution_json
             FROM trial_params WHERE study_id = ?1
             ORDER BY param_name",
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
        self.hpo_data
            .param_names
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
        self.hpo_data
            .param_names
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
        let Some(best_idx) = self.hpo_data.best_trial_idx else {
            return;
        };
        let Some(group_name) = self.hpo_data.trials[best_idx].group_name.clone() else {
            return;
        };
        load_trial_csv(
            &self.db_path,
            &group_name,
            self.log_y,
            &mut self.best_trial_metrics,
            &mut self.best_trial_log,
            &mut self.last_best_csv_modified,
        );
    }

    pub fn reload_detail_trial_csv(&mut self) {
        if !self.trial_detail_mode {
            return;
        }
        let sorted = self.sorted_trial_indices();
        if self.selected_trial >= sorted.len() {
            return;
        }
        let group_name = match &self.hpo_data.trials[sorted[self.selected_trial]].group_name {
            Some(g) => g.clone(),
            None => return,
        };
        load_trial_csv(
            &self.db_path,
            &group_name,
            self.log_y,
            &mut self.detail_metrics,
            &mut self.detail_log,
            &mut self.last_detail_csv_modified,
        );
    }

    /// Return trial indices sorted by objective value (best first), valueless at bottom.
    pub fn sorted_trial_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.hpo_data.trials.len()).collect();
        indices.sort_by(|&a, &b| {
            let ta = &self.hpo_data.trials[a];
            let tb = &self.hpo_data.trials[b];
            match (ta.value, tb.value) {
                (Some(va), Some(vb)) => va
                    .partial_cmp(&vb)
                    .unwrap_or(std::cmp::Ordering::Equal),
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
                    if self.active_tab == 3
                        && !self.trial_detail_mode
                        && self.selected_trial > 0
                    {
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
        candidates
            .into_iter()
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

fn parse_duration_secs(start: &str, end: &str) -> Option<f64> {
    let parse = |s: &str| -> Option<f64> {
        // Optuna stores "YYYY-MM-DD HH:MM:SS.ffffff" or ISO format
        let s = s.replace('T', " ");
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() < 2 {
            return None;
        }
        let date_parts: Vec<&str> = parts[0].split('-').collect();
        let time_str = parts[1];
        let time_parts: Vec<&str> = time_str.split(':').collect();
        if date_parts.len() < 3 || time_parts.len() < 3 {
            return None;
        }

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
                    n,
                    names
                );
            }
        }
    };

    let hpo_data = HpoData::empty(study_id, resolved_name);

    let mut app = HpoApp {
        db_path,
        conn,
        hpo_data,
        best_trial_metrics: Vec::new(),
        best_trial_log: LogState {
            log_scale: false,
            has_nonpositive: false,
            symlog_c: 1.0,
        },
        detail_metrics: Vec::new(),
        detail_log: LogState {
            log_scale: false,
            has_nonpositive: false,
            symlog_c: 1.0,
        },
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
                TrialInfo {
                    number: 0,
                    state: TrialState::Complete,
                    value: Some(0.5),
                    params: vec![],
                    categorical_labels: vec![],
                    duration_secs: None,
                    group_name: None,
                },
                TrialInfo {
                    number: 1,
                    state: TrialState::Complete,
                    value: Some(0.1),
                    params: vec![],
                    categorical_labels: vec![],
                    duration_secs: None,
                    group_name: None,
                },
                TrialInfo {
                    number: 2,
                    state: TrialState::Running,
                    value: None,
                    params: vec![],
                    categorical_labels: vec![],
                    duration_secs: None,
                    group_name: None,
                },
                TrialInfo {
                    number: 3,
                    state: TrialState::Complete,
                    value: Some(0.3),
                    params: vec![],
                    categorical_labels: vec![],
                    duration_secs: None,
                    group_name: None,
                },
            ],
            param_names: vec![],
            param_types: vec![],
            best_trial_idx: Some(1),
            total_trials_expected: None,
        };

        let app = HpoApp {
            db_path: PathBuf::new(),
            conn: Connection::open_in_memory().unwrap(),
            hpo_data: data,
            best_trial_metrics: vec![],
            best_trial_log: LogState {
                log_scale: false,
                has_nonpositive: false,
                symlog_c: 1.0,
            },
            detail_metrics: vec![],
            detail_log: LogState {
                log_scale: false,
                has_nonpositive: false,
                symlog_c: 1.0,
            },
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
                TrialInfo {
                    number: 0,
                    state: TrialState::Complete,
                    value: Some(0.5),
                    params: vec![],
                    categorical_labels: vec![],
                    duration_secs: None,
                    group_name: None,
                },
                TrialInfo {
                    number: 1,
                    state: TrialState::Pruned,
                    value: None,
                    params: vec![],
                    categorical_labels: vec![],
                    duration_secs: None,
                    group_name: None,
                },
                TrialInfo {
                    number: 2,
                    state: TrialState::Complete,
                    value: Some(0.3),
                    params: vec![],
                    categorical_labels: vec![],
                    duration_secs: None,
                    group_name: None,
                },
                TrialInfo {
                    number: 3,
                    state: TrialState::Complete,
                    value: Some(0.4),
                    params: vec![],
                    categorical_labels: vec![],
                    duration_secs: None,
                    group_name: None,
                },
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
            best_trial_log: LogState {
                log_scale: false,
                has_nonpositive: false,
                symlog_c: 1.0,
            },
            detail_metrics: vec![],
            detail_log: LogState {
                log_scale: false,
                has_nonpositive: false,
                symlog_c: 1.0,
            },
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
