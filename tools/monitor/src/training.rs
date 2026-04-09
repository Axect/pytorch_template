use std::{
    fs,
    path::PathBuf,
    time::{Duration, SystemTime},
};

use anyhow::Result;
use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    layout::{Constraint, Layout, Rect},
    style::{Color, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Paragraph, Tabs},
    Frame,
};

use crate::charts::{
    load_metrics_csv, render_generic_chart, render_positive_chart, render_training_overview,
    LogState, MetricRow, EXTRA_COLORS,
};

// ── App State ──────────────────────────────────────────────────────────────

pub struct App {
    pub csv_path: PathBuf,
    pub interval: Duration,
    pub metrics: Vec<MetricRow>,
    pub last_modified: Option<SystemTime>,
    /// Shared log/symlog state (log_scale toggle + sign characteristics)
    pub log_state: LogState,
    /// Currently active tab: 0 = Overview, 1+ = extra columns
    pub active_tab: usize,
    /// Names of dynamically detected extra CSV columns
    pub extra_columns: Vec<String>,
    /// Currently focused panel within the active tab (None = no focus)
    pub focused_panel: Option<usize>,
    /// Per-panel Y-axis bounds for Overview tab: [Loss, LR, Grad]
    pub overview_bounds: [Option<(f64, f64)>; 3],
    /// Per-extra-tab Y-axis bounds (one per extra column)
    pub extra_bounds: Vec<Option<(f64, f64)>>,
}

impl App {
    pub fn new(csv_path: PathBuf, interval: Duration) -> Self {
        Self {
            csv_path,
            interval,
            metrics: Vec::new(),
            last_modified: None,
            log_state: LogState {
                log_scale: false,
                has_nonpositive: false,
                symlog_c: 1.0,
            },
            active_tab: 0,
            extra_columns: Vec::new(),
            focused_panel: None,
            overview_bounds: [None; 3],
            extra_bounds: Vec::new(),
        }
    }

    pub fn try_reload(&mut self) {
        let Ok(meta) = fs::metadata(&self.csv_path) else {
            return;
        };
        let Ok(modified) = meta.modified() else {
            return;
        };
        if self.last_modified == Some(modified) {
            return;
        }
        self.last_modified = Some(modified);

        let (rows, new_extras) = load_metrics_csv(&self.csv_path);

        // Merge with existing extra_columns (append new ones, preserve order)
        for col in &new_extras {
            if !self.extra_columns.contains(col) {
                self.extra_columns.push(col.clone());
            }
        }

        // Grow extra_bounds to match extra_columns length
        self.extra_bounds.resize(self.extra_columns.len(), None);

        // Clamp active_tab if columns were removed
        let total_tabs = 1 + self.extra_columns.len();
        if self.active_tab >= total_tabs {
            self.active_tab = 0;
        }

        // Preserve the user's log_scale toggle, recompute sign characteristics
        let log_scale_was = self.log_state.log_scale;
        self.log_state = LogState::from_metrics(&rows);
        self.log_state.log_scale = log_scale_was;

        self.metrics = rows;
    }

    pub fn elapsed_secs(&self) -> Option<u64> {
        self.last_modified
            .and_then(|t| SystemTime::now().duration_since(t).ok())
            .map(|d| d.as_secs())
    }

    pub fn total_tabs(&self) -> usize {
        1 + self.extra_columns.len()
    }

    /// Number of panels in the current tab
    pub fn panel_count(&self) -> usize {
        if self.active_tab == 0 {
            let has_grad = self.metrics.iter().any(|m| m.max_grad_norm.is_some());
            if has_grad { 3 } else { 2 }
        } else {
            1 // Extra tabs have 1 panel
        }
    }

    /// Get the y_bounds for a specific overview panel
    pub fn overview_panel_bounds(&self, panel: usize) -> Option<(f64, f64)> {
        self.overview_bounds[panel]
    }

    /// Get the y_bounds for an extra tab
    pub fn extra_tab_bounds(&self, col_idx: usize) -> Option<(f64, f64)> {
        self.extra_bounds.get(col_idx).copied().flatten()
    }

    /// Auto-compute Y bounds for the currently focused panel
    fn auto_y_bounds_for_focused(&self) -> (f64, f64) {
        if self.active_tab == 0 {
            match self.focused_panel {
                Some(0) => {
                    // Loss panel
                    let train: Vec<(f64, f64)> = self.metrics.iter()
                        .map(|m| (m.epoch, self.log_state.loss_y(m.train_loss)))
                        .collect();
                    let val: Vec<(f64, f64)> = self.metrics.iter()
                        .map(|m| (m.epoch, self.log_state.loss_y(m.val_loss)))
                        .collect();
                    let (_, y_min, y_max) = crate::charts::bounds_xy(&train, &val);
                    let pad = (y_max - y_min).max(1e-10) * 0.1;
                    (y_min - pad, y_max + pad)
                }
                Some(1) => {
                    // LR panel
                    let data: Vec<(f64, f64)> = self.metrics.iter().map(|m| {
                        let y = if self.log_state.log_scale { m.lr.max(1e-20).log10() } else { m.lr };
                        (m.epoch, y)
                    }).collect();
                    let (y_min, y_max) = crate::charts::min_max_y(&data);
                    let pad = (y_max - y_min).max(1e-15) * 0.1;
                    (y_min - pad, y_max + pad)
                }
                Some(2) => {
                    // Grad panel
                    let data: Vec<(f64, f64)> = self.metrics.iter()
                        .filter_map(|m| m.max_grad_norm.map(|g| {
                            let y = if self.log_state.log_scale { g.max(1e-20).log10() } else { g };
                            (m.epoch, y)
                        })).collect();
                    let (y_min, y_max) = crate::charts::min_max_y(&data);
                    let pad = (y_max - y_min).max(1e-15) * 0.1;
                    (y_min - pad, y_max + pad)
                }
                _ => (0.0, 1.0),
            }
        } else {
            // Extra tab
            let col_idx = self.active_tab - 1;
            let data: Vec<(f64, f64)> = self.metrics.iter()
                .filter_map(|m| m.extras.get(col_idx).copied().flatten().map(|v| {
                    let y = if self.log_state.log_scale {
                        if self.log_state.has_nonpositive {
                            crate::charts::symlog(v, self.log_state.symlog_c)
                        } else {
                            v.max(1e-20).log10()
                        }
                    } else { v };
                    (m.epoch, y)
                })).collect();
            let (y_min, y_max) = crate::charts::min_max_y(&data);
            let pad = (y_max - y_min).max(1e-10) * 0.1;
            (y_min - pad, y_max + pad)
        }
    }

    pub fn handle_event(&mut self, ev: Event) -> bool {
        if let Event::Key(key) = ev {
            if key.kind != KeyEventKind::Press {
                return false;
            }
            let n = self.total_tabs();
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return true,
                KeyCode::Char('l') => self.log_state.log_scale = !self.log_state.log_scale,
                KeyCode::Right | KeyCode::Tab => {
                    if n > 1 {
                        self.active_tab = (self.active_tab + 1) % n;
                        self.focused_panel = None;
                    }
                }
                KeyCode::Left | KeyCode::BackTab => {
                    if n > 1 {
                        self.active_tab = (self.active_tab + n - 1) % n;
                        self.focused_panel = None;
                    }
                }
                // Number keys 1-9: toggle panel focus
                KeyCode::Char(c @ '1'..='9') => {
                    let idx = (c as usize) - ('1' as usize);
                    let count = self.panel_count();
                    if idx < count {
                        if self.focused_panel == Some(idx) {
                            self.focused_panel = None;
                        } else {
                            self.focused_panel = Some(idx);
                        }
                    }
                }
                // Zoom in: narrow Y range by 20%
                KeyCode::Char('+') | KeyCode::Char('=') => {
                    if self.can_zoom() {
                        let (lo, hi) = self.get_active_bounds();
                        let center = (lo + hi) / 2.0;
                        let half = (hi - lo) / 2.0 * 0.8;
                        self.set_active_bounds(Some((center - half, center + half)));
                    }
                }
                // Zoom out: widen Y range by 25%
                KeyCode::Char('-') => {
                    if self.can_zoom() {
                        let (lo, hi) = self.get_active_bounds();
                        let center = (lo + hi) / 2.0;
                        let half = (hi - lo) / 2.0 * 1.25;
                        self.set_active_bounds(Some((center - half, center + half)));
                    }
                }
                // Pan up
                KeyCode::Up => {
                    if self.can_zoom() {
                        let (lo, hi) = self.get_active_bounds();
                        let shift = (hi - lo) * 0.1;
                        self.set_active_bounds(Some((lo + shift, hi + shift)));
                    }
                }
                // Pan down
                KeyCode::Down => {
                    if self.can_zoom() {
                        let (lo, hi) = self.get_active_bounds();
                        let shift = (hi - lo) * 0.1;
                        self.set_active_bounds(Some((lo - shift, hi - shift)));
                    }
                }
                // Reset bounds
                KeyCode::Char('r') => {
                    if self.can_zoom() {
                        self.set_active_bounds(None);
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Whether zoom/pan keys should be active
    fn can_zoom(&self) -> bool {
        if self.active_tab == 0 {
            self.focused_panel.is_some() // Overview: need panel selected
        } else {
            true // Extra tabs: always zoomable (single panel)
        }
    }

    /// Get current Y bounds for the active zoom target
    fn get_active_bounds(&self) -> (f64, f64) {
        if self.active_tab == 0 {
            if let Some(p) = self.focused_panel {
                self.overview_bounds[p].unwrap_or_else(|| self.auto_y_bounds_for_focused())
            } else {
                (0.0, 1.0)
            }
        } else {
            let idx = self.active_tab - 1;
            self.extra_bounds.get(idx).copied().flatten()
                .unwrap_or_else(|| self.auto_y_bounds_for_focused())
        }
    }

    /// Set Y bounds for the active zoom target
    fn set_active_bounds(&mut self, bounds: Option<(f64, f64)>) {
        if self.active_tab == 0 {
            if let Some(p) = self.focused_panel {
                self.overview_bounds[p] = bounds;
            }
        } else {
            let idx = self.active_tab - 1;
            if let Some(b) = self.extra_bounds.get_mut(idx) {
                *b = bounds;
            }
        }
    }
}

// ── Rendering ──────────────────────────────────────────────────────────────

pub fn render(frame: &mut Frame, app: &App) {
    let has_tabs = !app.extra_columns.is_empty();

    if has_tabs {
        // Tab bar + content + status
        let [tab_area, content_area, status_area] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .areas(frame.area());

        render_tab_bar(frame, app, tab_area);

        if app.active_tab == 0 {
            render_training_overview(frame, &app.metrics, &app.log_state, content_area, &app.overview_bounds, app.focused_panel);
        } else {
            render_extra_tab(frame, app, content_area, app.active_tab - 1);
        }

        render_status(frame, app, status_area);
    } else {
        // No extra columns — original layout without tab bar
        let [content_area, status_area] = Layout::vertical([
            Constraint::Min(0),
            Constraint::Length(3),
        ])
        .areas(frame.area());

        render_training_overview(frame, &app.metrics, &app.log_state, content_area, &app.overview_bounds, app.focused_panel);
        render_status(frame, app, status_area);
    }
}

pub fn render_tab_bar(frame: &mut Frame, app: &App, area: Rect) {
    let mut titles: Vec<Line> = vec![Line::from(" Overview ")];
    for col in &app.extra_columns {
        titles.push(Line::from(format!(" {} ", col)));
    }

    let tabs = Tabs::new(titles)
        .select(app.active_tab)
        .style(Style::new().fg(Color::DarkGray))
        .highlight_style(Style::new().fg(Color::Cyan).bold())
        .divider(Span::raw("│"));

    frame.render_widget(tabs, area);
}

pub fn render_extra_tab(frame: &mut Frame, app: &App, area: Rect, col_idx: usize) {
    let col_name = &app.extra_columns[col_idx];
    let data: Vec<(f64, f64)> = app
        .metrics
        .iter()
        .filter_map(|m| {
            m.extras
                .get(col_idx)
                .copied()
                .flatten()
                .map(|v| (m.epoch, v))
        })
        .collect();

    if data.is_empty() {
        frame.render_widget(
            Paragraph::new(format!(" No data for column '{}'", col_name))
                .block(Block::bordered().title(format!(" {} ", col_name))),
            area,
        );
        return;
    }

    let has_negative = data.iter().any(|(_, y)| *y < 0.0);
    let color = EXTRA_COLORS[col_idx % EXTRA_COLORS.len()];
    let bounds = app.extra_tab_bounds(col_idx);

    if has_negative {
        // Use loss-style rendering (supports symlog for negative values)
        render_generic_chart(frame, &app.log_state, area, &data, col_name, col_name, color, bounds, false);
    } else {
        render_positive_chart(frame, app.log_state.log_scale, area, &data, col_name, col_name, color, bounds, false);
    }
}

pub fn render_status(frame: &mut Frame, app: &App, area: Rect) {
    let text = if let Some(last) = app.metrics.last() {
        let mut parts = vec![
            format!("Epoch {}", last.epoch as u64 + 1),
            format!("train: {:.4e}", last.train_loss),
            format!("val: {:.4e}", last.val_loss),
            format!("lr: {:.4e}", last.lr),
        ];
        if let Some(g) = last.max_grad_norm {
            parts.push(format!("grad: {:.2e}", g));
        }
        if let Some(p) = last.predicted_final_loss {
            parts.push(format!("pred: {:.2e}", p));
        }

        let elapsed = app
            .elapsed_secs()
            .map(|s| {
                if s < 60 {
                    format!("{s}s ago")
                } else {
                    format!("{}m{}s ago", s / 60, s % 60)
                }
            })
            .unwrap_or_default();

        let log_tag = match (app.log_state.log_scale, app.log_state.has_nonpositive) {
            (false, _) => "",
            (true, false) => " [LOG]",
            (true, true) => " [SYMLOG]",
        };

        let tab_hint = if app.extra_columns.is_empty() {
            ""
        } else {
            " │ ←→: tabs"
        };

        format!(
            " {} │ updated {}\n q: quit │ l: log scale{}{}",
            parts.join(" │ "),
            elapsed,
            log_tag,
            tab_hint,
        )
    } else {
        let tab_hint = if app.extra_columns.is_empty() {
            ""
        } else {
            " │ ←→: tabs"
        };
        format!(
            " Waiting: {} │ q: quit │ l: log scale{}",
            app.csv_path.display(),
            tab_hint,
        )
    };

    frame.render_widget(
        Paragraph::new(text)
            .block(Block::bordered().title(" Status ").fg(Color::DarkGray)),
        area,
    );
}

// ── Entry Point ────────────────────────────────────────────────────────────

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
