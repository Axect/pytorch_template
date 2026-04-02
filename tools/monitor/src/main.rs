use std::{
    fs,
    path::PathBuf,
    time::{Duration, SystemTime},
};

use anyhow::Result;
use clap::Parser;
use ratatui::{
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    layout::{Constraint, Layout, Rect},
    style::{Color, Style, Stylize},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Chart, Dataset, GraphType, Paragraph},
    Frame,
};

// ── CLI ────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "training-monitor")]
#[command(about = "Real-time TUI training monitor for pytorch_template")]
struct Cli {
    /// Path to metrics.csv or its parent directory
    path: PathBuf,

    /// Refresh interval in milliseconds
    #[arg(short, long, default_value_t = 500)]
    interval: u64,
}

// ── Data ───────────────────────────────────────────────────────────────────

struct MetricRow {
    epoch: f64,
    train_loss: f64,
    val_loss: f64,
    lr: f64,
    max_grad_norm: Option<f64>,
    predicted_final_loss: Option<f64>,
}

// ── App State ──────────────────────────────────────────────────────────────

struct App {
    csv_path: PathBuf,
    interval: Duration,
    metrics: Vec<MetricRow>,
    last_modified: Option<SystemTime>,
    log_scale: bool,
    /// true when any train/val loss ≤ 0 — triggers symlog instead of log₁₀
    has_nonpositive: bool,
    /// symlog threshold: smallest |loss| > 0 in the data
    symlog_c: f64,
}

impl App {
    fn new(csv_path: PathBuf, interval: Duration) -> Self {
        Self {
            csv_path,
            interval,
            metrics: Vec::new(),
            last_modified: None,
            log_scale: false,
            has_nonpositive: false,
            symlog_c: 1.0,
        }
    }

    fn try_reload(&mut self) {
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

        let Ok(mut rdr) = csv::Reader::from_path(&self.csv_path) else {
            return;
        };
        let Ok(headers) = rdr.headers().cloned() else {
            return;
        };

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

            rows.push(MetricRow {
                epoch: get("epoch").unwrap_or(0.0),
                train_loss: get("train_loss").unwrap_or(0.0),
                val_loss: get("val_loss").unwrap_or(0.0),
                lr: get("lr").unwrap_or(0.0),
                max_grad_norm: get("max_grad_norm"),
                predicted_final_loss: get("predicted_final_loss"),
            });
        }

        // Detect sign characteristics for log-scale mode selection
        self.has_nonpositive = rows
            .iter()
            .any(|m| m.train_loss <= 0.0 || m.val_loss <= 0.0);

        if self.has_nonpositive {
            let min_abs = rows
                .iter()
                .flat_map(|m| [m.train_loss.abs(), m.val_loss.abs()])
                .filter(|&v| v > 1e-15)
                .fold(f64::INFINITY, f64::min);
            self.symlog_c = if min_abs.is_finite() { min_abs } else { 1.0 };
        }

        self.metrics = rows;
    }

    /// Transform a loss value for the Y axis
    fn loss_y(&self, v: f64) -> f64 {
        if !self.log_scale {
            v
        } else if self.has_nonpositive {
            symlog(v, self.symlog_c)
        } else {
            v.max(1e-20).log10()
        }
    }

    /// Inverse transform: transformed Y → original loss value (for axis labels)
    fn loss_y_inv(&self, y: f64) -> f64 {
        if !self.log_scale {
            y
        } else if self.has_nonpositive {
            symlog_inv(y, self.symlog_c)
        } else {
            10f64.powf(y)
        }
    }

    fn loss_title(&self) -> &'static str {
        match (self.log_scale, self.has_nonpositive) {
            (false, _) => " Loss Curves ",
            (true, false) => " Loss Curves (log\u{2081}\u{2080}) ",
            (true, true) => " Loss Curves (symlog\u{2081}\u{2080}) ",
        }
    }

    /// Generate Y-axis labels showing original loss values
    fn make_loss_labels(&self, lo: f64, hi: f64, n: usize) -> Vec<String> {
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

    fn elapsed_secs(&self) -> Option<u64> {
        self.last_modified
            .and_then(|t| SystemTime::now().duration_since(t).ok())
            .map(|d| d.as_secs())
    }

    fn handle_event(&mut self, ev: Event) -> bool {
        if let Event::Key(key) = ev {
            if key.kind != KeyEventKind::Press {
                return false;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return true,
                KeyCode::Char('l') => self.log_scale = !self.log_scale,
                _ => {}
            }
        }
        false
    }
}

// ── Symmetric log ─────────────────────────────────────────────────────────
// symlog(x, C) = sign(x) · log₁₀(1 + |x|/C)
//   - all positive & x >> C  →  behaves like log₁₀(x/C)
//   - near zero              →  approximately linear (x / (C·ln10))
//   - negative values        →  mirrors the positive side

fn symlog(x: f64, c: f64) -> f64 {
    x.signum() * (1.0 + x.abs() / c).log10()
}

fn symlog_inv(y: f64, c: f64) -> f64 {
    y.signum() * c * (10f64.powf(y.abs()) - 1.0)
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    let csv_path = if cli.path.is_dir() {
        cli.path.join("metrics.csv")
    } else {
        cli.path
    };

    let mut app = App::new(csv_path, Duration::from_millis(cli.interval));
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

// ── Rendering ──────────────────────────────────────────────────────────────

fn render(frame: &mut Frame, app: &App) {
    let has_grad = app.metrics.iter().any(|m| m.max_grad_norm.is_some());

    if has_grad {
        let [loss_area, lr_area, grad_area, status_area] = Layout::vertical([
            Constraint::Percentage(45),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Length(3),
        ])
        .areas(frame.area());

        render_loss_chart(frame, app, loss_area);
        render_lr_chart(frame, app, lr_area);
        render_grad_chart(frame, app, grad_area);
        render_status(frame, app, status_area);
    } else {
        let [loss_area, lr_area, status_area] = Layout::vertical([
            Constraint::Percentage(65),
            Constraint::Percentage(22),
            Constraint::Length(3),
        ])
        .areas(frame.area());

        render_loss_chart(frame, app, loss_area);
        render_lr_chart(frame, app, lr_area);
        render_status(frame, app, status_area);
    }
}

fn render_loss_chart(frame: &mut Frame, app: &App, area: Rect) {
    if app.metrics.is_empty() {
        frame.render_widget(
            Paragraph::new(" Waiting for metrics.csv data...")
                .block(Block::bordered().title(" Loss Curves ")),
            area,
        );
        return;
    }

    let train: Vec<(f64, f64)> = app
        .metrics
        .iter()
        .map(|m| (m.epoch, app.loss_y(m.train_loss)))
        .collect();

    let val: Vec<(f64, f64)> = app
        .metrics
        .iter()
        .map(|m| (m.epoch, app.loss_y(m.val_loss)))
        .collect();

    let (x_max, y_min, y_max) = bounds_xy(&train, &val);
    let y_pad = (y_max - y_min).max(1e-10) * 0.1;
    let y_lo = y_min - y_pad;
    let y_hi = y_max + y_pad;

    let mut datasets = vec![
        Dataset::default()
            .name("train")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::new().fg(Color::Cyan))
            .data(&train),
        Dataset::default()
            .name("val")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::new().fg(Color::Yellow))
            .data(&val),
    ];

    // Overlay predicted final loss as a horizontal target line
    let pred_data: Vec<(f64, f64)>;
    if let Some(pred) = app.metrics.last().and_then(|m| m.predicted_final_loss) {
        let pred_y = app.loss_y(pred);
        if pred_y >= y_lo && pred_y <= y_hi {
            // Sparse dotted line — sample every few x-units
            let step = (x_max / 40.0).max(1.0);
            let mut pts = Vec::new();
            let mut x = 0.0;
            while x <= x_max {
                pts.push((x, pred_y));
                x += step;
            }
            pred_data = pts;
            datasets.push(
                Dataset::default()
                    .name("pred")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Scatter)
                    .style(Style::new().fg(Color::DarkGray))
                    .data(&pred_data),
            );
        }
    }

    let x_labels = make_labels(0.0, x_max, 5, false);
    let y_labels = app.make_loss_labels(y_lo, y_hi, 5);

    // Inline legend in title bar — always visible
    let title = Line::from(vec![
        Span::raw(app.loss_title()),
        Span::styled("━ train ", Style::new().fg(Color::Cyan)),
        Span::styled("━ val ", Style::new().fg(Color::Yellow)),
    ]);

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(title))
        .x_axis(
            Axis::default()
                .title("epoch")
                .style(Style::new().fg(Color::DarkGray))
                .bounds([0.0, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title("loss")
                .style(Style::new().fg(Color::DarkGray))
                .bounds([y_lo, y_hi])
                .labels(y_labels),
        );

    frame.render_widget(chart, area);
}

fn render_lr_chart(frame: &mut Frame, app: &App, area: Rect) {
    if app.metrics.is_empty() {
        frame.render_widget(
            Paragraph::new(" Waiting...")
                .block(Block::bordered().title(" Learning Rate ")),
            area,
        );
        return;
    }

    render_positive_chart(
        frame,
        app,
        area,
        &app.metrics.iter().map(|m| (m.epoch, m.lr)).collect::<Vec<_>>(),
        "Learning Rate",
        "lr",
        Color::Green,
    );
}

fn render_grad_chart(frame: &mut Frame, app: &App, area: Rect) {
    let grad: Vec<(f64, f64)> = app
        .metrics
        .iter()
        .filter_map(|m| m.max_grad_norm.map(|g| (m.epoch, g)))
        .collect();

    if grad.is_empty() {
        frame.render_widget(
            Paragraph::new(" No gradient data")
                .block(Block::bordered().title(" Gradient Norm ")),
            area,
        );
        return;
    }

    render_positive_chart(frame, app, area, &grad, "Gradient Norm", "‖∇‖", Color::Yellow);
}

/// Shared renderer for always-positive metric charts (LR, grad norm).
/// Applies log₁₀ when the global log_scale toggle is on.
fn render_positive_chart(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    raw_data: &[(f64, f64)],
    name: &str,
    y_title: &str,
    color: Color,
) {
    let data: Vec<(f64, f64)> = raw_data
        .iter()
        .map(|&(x, y)| {
            let y_t = if app.log_scale {
                y.max(1e-20).log10()
            } else {
                y
            };
            (x, y_t)
        })
        .collect();

    let x_max = data.last().map(|(x, _)| x + 1.0).unwrap_or(1.0);
    let (y_min, y_max) = min_max_y(&data);
    let y_pad = (y_max - y_min).max(1e-15) * 0.1;
    let y_lo = y_min - y_pad;
    let y_hi = y_max + y_pad;

    let datasets = vec![Dataset::default()
        .name(name)
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(color))
        .data(&data)];

    let title = if app.log_scale {
        format!(" {} (log\u{2081}\u{2080}) ", name)
    } else {
        format!(" {} ", name)
    };

    let x_labels = make_labels(0.0, x_max, 5, false);
    let y_labels = if app.log_scale {
        // Show original values via 10^y
        make_inv_log10_labels(y_lo, y_hi, 3)
    } else {
        make_labels(y_lo, y_hi, 3, true)
    };

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(title))
        .x_axis(
            Axis::default()
                .title("epoch")
                .style(Style::new().fg(Color::DarkGray))
                .bounds([0.0, x_max])
                .labels(x_labels),
        )
        .y_axis(
            Axis::default()
                .title(y_title)
                .style(Style::new().fg(Color::DarkGray))
                .bounds([y_lo, y_hi])
                .labels(y_labels),
        );

    frame.render_widget(chart, area);
}

fn render_status(frame: &mut Frame, app: &App, area: Rect) {
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

        let log_tag = match (app.log_scale, app.has_nonpositive) {
            (false, _) => "",
            (true, false) => " [LOG]",
            (true, true) => " [SYMLOG]",
        };

        format!(
            " {} │ updated {}\n q: quit │ l: log scale{}",
            parts.join(" │ "),
            elapsed,
            log_tag,
        )
    } else {
        format!(
            " Waiting: {} │ q: quit │ l: log scale",
            app.csv_path.display()
        )
    };

    frame.render_widget(
        Paragraph::new(text)
            .block(Block::bordered().title(" Status ").fg(Color::DarkGray)),
        area,
    );
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn bounds_xy(d1: &[(f64, f64)], d2: &[(f64, f64)]) -> (f64, f64, f64) {
    let x_max = d1
        .iter()
        .chain(d2.iter())
        .map(|(x, _)| *x)
        .fold(0.0_f64, f64::max)
        + 1.0;
    let y_min = d1
        .iter()
        .chain(d2.iter())
        .map(|(_, y)| *y)
        .fold(f64::INFINITY, f64::min);
    let y_max = d1
        .iter()
        .chain(d2.iter())
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    (x_max, y_min, y_max)
}

fn min_max_y(data: &[(f64, f64)]) -> (f64, f64) {
    let y_min = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = data
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    (y_min, y_max)
}

fn make_labels(lo: f64, hi: f64, n: usize, scientific: bool) -> Vec<String> {
    (0..=n)
        .map(|i| {
            let val = lo + (hi - lo) * i as f64 / n as f64;
            if scientific {
                format!("{:.1e}", val)
            } else {
                format!("{:.0}", val)
            }
        })
        .collect()
}

/// Labels for log₁₀-transformed positive data — shows original values (10^y).
fn make_inv_log10_labels(lo: f64, hi: f64, n: usize) -> Vec<String> {
    (0..=n)
        .map(|i| {
            let y = lo + (hi - lo) * i as f64 / n as f64;
            format!("{:.1e}", 10f64.powf(y))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symlog_zero() {
        assert_eq!(symlog(0.0, 1.0), 0.0);
    }

    #[test]
    fn symlog_positive_large() {
        // For x >> c: symlog(x, c) ≈ log10(x/c)
        let val = symlog(1000.0, 0.01);
        let expected = (1.0 + 1000.0 / 0.01_f64).log10();
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn symlog_negative_symmetry() {
        let c = 0.1;
        assert!((symlog(5.0, c) + symlog(-5.0, c)).abs() < 1e-10);
    }

    #[test]
    fn symlog_roundtrip() {
        let c = 0.001;
        for &x in &[-10.0, -0.5, -0.001, 0.0, 0.001, 0.5, 10.0] {
            let y = symlog(x, c);
            let recovered = symlog_inv(y, c);
            assert!(
                (x - recovered).abs() < 1e-10,
                "roundtrip failed: x={x}, y={y}, recovered={recovered}"
            );
        }
    }

    #[test]
    fn symlog_monotonic() {
        let c = 0.01;
        let xs: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        for w in xs.windows(2) {
            assert!(symlog(w[0], c) < symlog(w[1], c), "not monotonic at {}", w[0]);
        }
    }
}
