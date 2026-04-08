use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Style, Stylize},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Chart, Dataset, GraphType, Paragraph, Row, Table, Tabs},
    Frame,
};

use crate::charts::{
    make_inv_log10_labels, make_labels, min_max_y, render_training_overview,
};
use super::{HpoApp, ParamType, TrialState};

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
    let titles: Vec<Line> = app
        .tab_names()
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
    let completed = trials
        .iter()
        .filter(|t| t.state == TrialState::Complete)
        .count();
    let pruned = trials
        .iter()
        .filter(|t| t.state == TrialState::Pruned)
        .count();
    let failed = trials
        .iter()
        .filter(|t| t.state == TrialState::Fail)
        .count();
    let running = trials
        .iter()
        .filter(|t| t.state == TrialState::Running)
        .count();
    let total = trials.len();

    let summary = Line::from(vec![
        Span::styled(
            format!(" Completed: {} ", completed),
            Style::new().fg(Color::Green).bold(),
        ),
        Span::raw("│"),
        Span::styled(
            format!(" Pruned: {} ", pruned),
            Style::new().fg(Color::Yellow),
        ),
        Span::raw("│"),
        Span::styled(
            format!(" Failed: {} ", failed),
            Style::new().fg(Color::Red),
        ),
        Span::raw("│"),
        Span::styled(
            format!(" Running: {} ", running),
            Style::new().fg(Color::Cyan),
        ),
        Span::raw("│"),
        Span::styled(
            format!(" Total: {} ", total),
            Style::new().fg(Color::White),
        ),
    ]);

    frame.render_widget(
        Paragraph::new(summary).block(Block::bordered().title(" Study Status ")),
        summary_area,
    );

    // All trial values scatter + best convergence line
    let curve = app.convergence_curve();

    // Collect all trial objective values as scatter points
    let all_scatter: Vec<(f64, f64)> = app
        .hpo_data
        .trials
        .iter()
        .filter_map(|t| {
            if t.state == TrialState::Complete {
                t.value.map(|v| (t.number as f64, v))
            } else {
                None
            }
        })
        .collect();

    if curve.is_empty() && all_scatter.is_empty() {
        frame.render_widget(
            Paragraph::new(" Waiting for completed trials...")
                .block(Block::bordered().title(" Objective Values ")),
            chart_area,
        );
        return;
    }

    // Transform for log scale
    let scatter_data: Vec<(f64, f64)> = if app.log_y {
        all_scatter.iter().map(|&(x, y)| (x, y.max(1e-20).log10())).collect()
    } else {
        all_scatter.clone()
    };

    let best_data: Vec<(f64, f64)> = if app.log_y {
        curve.iter().map(|&(x, y)| (x, y.max(1e-20).log10())).collect()
    } else {
        curve.clone()
    };

    // Compute bounds from both datasets
    let all_points: Vec<&(f64, f64)> = scatter_data.iter().chain(best_data.iter()).collect();
    let x_max = all_points.iter().map(|(x, _)| *x).fold(0.0_f64, f64::max) + 1.0;
    let y_min = all_points.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = all_points.iter().map(|(_, y)| *y).fold(f64::NEG_INFINITY, f64::max);
    let y_pad = (y_max - y_min).max(1e-10) * 0.1;
    let y_lo = y_min - y_pad;
    let y_hi = y_max + y_pad;

    let mut datasets = vec![];

    // Scatter first (behind the line)
    if !scatter_data.is_empty() {
        datasets.push(
            Dataset::default()
                .name("trials")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Scatter)
                .style(Style::new().fg(Color::DarkGray))
                .data(&scatter_data),
        );
    }

    // Best convergence line on top
    if !best_data.is_empty() {
        datasets.push(
            Dataset::default()
                .name("best")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::new().fg(Color::Green))
                .data(&best_data),
        );
    }

    let title = Line::from(vec![
        Span::raw(if app.log_y {
            " Objective Values (log\u{2081}\u{2080}) "
        } else {
            " Objective Values "
        }),
        Span::styled("· trials ", Style::new().fg(Color::DarkGray)),
        Span::styled("━ best ", Style::new().fg(Color::Green)),
    ]);

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
            Paragraph::new(" No parameters found")
                .block(Block::bordered().title(" Parameters ")),
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
        let Some(param_val) = trial.params.get(param_idx).copied().flatten() else {
            continue;
        };
        let Some(obj_val) = trial.value else { continue };

        let x = if app.log_x && param_type != ParamType::Categorical {
            param_val.abs().max(1e-20).log10()
        } else {
            param_val
        };
        let y = if app.log_y {
            obj_val.max(1e-20).log10()
        } else {
            obj_val
        };

        match trial.state {
            TrialState::Complete => complete_points.push((x, y)),
            TrialState::Pruned => pruned_points.push((x, y)),
            TrialState::Fail => failed_points.push((x, y)),
            _ => {}
        }
    }

    let all_points: Vec<&(f64, f64)> = complete_points
        .iter()
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

    let x_min = all_points
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::INFINITY, f64::min);
    let x_max = all_points
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = all_points
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);
    let y_max = all_points
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);

    let x_pad = (x_max - x_min).max(1e-10) * 0.15;
    let y_pad = (y_max - y_min).max(1e-10) * 0.1;
    let x_lo = x_min - x_pad;
    let x_hi = x_max + x_pad;
    let y_lo = y_min - y_pad;
    let y_hi = y_max + y_pad;

    let mut datasets = Vec::new();
    if !complete_points.is_empty() {
        datasets.push(
            Dataset::default()
                .name("ok")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Scatter)
                .style(Style::new().fg(Color::Green))
                .data(&complete_points),
        );
    }
    if !pruned_points.is_empty() {
        datasets.push(
            Dataset::default()
                .name("pruned")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Scatter)
                .style(Style::new().fg(Color::Yellow))
                .data(&pruned_points),
        );
    }
    if !failed_points.is_empty() {
        datasets.push(
            Dataset::default()
                .name("fail")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Scatter)
                .style(Style::new().fg(Color::Red))
                .data(&failed_points),
        );
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
            Paragraph::new(format!(
                " Best Trial #{} — val_loss: {}",
                t.number, val_str
            ))
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
    let param_names: Vec<&str> = app
        .hpo_data
        .param_names
        .iter()
        .map(|s| s.as_str())
        .collect();
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
                trial
                    .value
                    .map(|v| format!("{:.4e}", v))
                    .unwrap_or("-".into()),
            ];

            for (i, _name) in app.hpo_data.param_names.iter().enumerate() {
                let val = trial.params.get(i).copied().flatten();
                if app.hpo_data.param_types[i] == ParamType::Categorical {
                    let label = trial
                        .categorical_labels
                        .get(i)
                        .and_then(|l| l.as_deref());
                    cells.push(label.unwrap_or("-").to_string());
                } else {
                    cells.push(val.map(|v| format!("{:.4e}", v)).unwrap_or("-".into()));
                }
            }

            let dur = trial
                .duration_secs
                .map(|s| {
                    if s < 60.0 {
                        format!("{:.0}s", s)
                    } else if s < 3600.0 {
                        format!("{:.0}m{:.0}s", s / 60.0, s % 60.0)
                    } else {
                        format!("{:.1}h", s / 3600.0)
                    }
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
        let mut w = vec![Constraint::Length(4)]; // #
        w.push(Constraint::Length(9)); // State
        w.push(Constraint::Length(12)); // Value
        for _ in &app.hpo_data.param_names {
            w.push(Constraint::Length(12)); // Params
        }
        w.push(Constraint::Length(8)); // Duration
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
    if app.selected_trial >= sorted.len() {
        return;
    }
    let trial = &app.hpo_data.trials[sorted[app.selected_trial]];

    let [header_area, chart_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(0),
    ])
    .areas(area);

    let val_str = trial
        .value
        .map(|v| format!("{:.4e}", v))
        .unwrap_or("-".into());
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
            trial.number,
            trial.state.label(),
            val_str
        ))
        .style(Style::new().fg(state_color).bold()),
        header_area,
    );

    if app.detail_metrics.is_empty() {
        frame.render_widget(
            Paragraph::new(" Waiting for metrics.csv data...").block(Block::bordered()),
            chart_area,
        );
    } else {
        render_training_overview(frame, &app.detail_metrics, &app.detail_log, chart_area);
    }
}

// ── Status Bar ────────────────────────────────────────────────────────────

fn render_hpo_status(frame: &mut Frame, app: &HpoApp, area: Rect) {
    let completed = app
        .hpo_data
        .trials
        .iter()
        .filter(|t| t.state == TrialState::Complete)
        .count();
    let total = app.hpo_data.trials.len();

    let best_str = app
        .hpo_data
        .best_trial_idx
        .and_then(|i| {
            let t = &app.hpo_data.trials[i];
            t.value
                .map(|v| format!("Best: {:.4e} (trial #{})", v, t.number))
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
        tab_name,
        completed,
        total,
        best_str,
        app.hpo_data.study_name,
        extra_keys,
        log_tags,
    );

    frame.render_widget(
        Paragraph::new(text)
            .block(Block::bordered().title(" Status ").fg(Color::DarkGray)),
        area,
    );
}
