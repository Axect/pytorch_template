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
                    }
                }
                KeyCode::Left | KeyCode::BackTab => {
                    if n > 1 {
                        self.active_tab = (self.active_tab + n - 1) % n;
                    }
                }
                _ => {}
            }
        }
        false
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
            render_training_overview(frame, &app.metrics, &app.log_state, content_area);
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

        render_training_overview(frame, &app.metrics, &app.log_state, content_area);
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

    if has_negative {
        // Use loss-style rendering (supports symlog for negative values)
        render_generic_chart(frame, &app.log_state, area, &data, col_name, col_name, color);
    } else {
        render_positive_chart(frame, app.log_state.log_scale, area, &data, col_name, col_name, color);
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
