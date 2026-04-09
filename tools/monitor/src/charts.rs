use std::path::Path;

use ratatui::{
    layout::Rect,
    style::{Color, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Chart, Dataset, GraphType, Paragraph},
    Frame,
};

// ── Data ───────────────────────────────────────────────────────────────────

pub struct MetricRow {
    pub epoch: f64,
    pub train_loss: f64,
    pub val_loss: f64,
    pub lr: f64,
    pub max_grad_norm: Option<f64>,
    pub predicted_final_loss: Option<f64>,
    /// Extra columns (parallel to App::extra_columns)
    pub extras: Vec<Option<f64>>,
}

// ── Extra Colors ──────────────────────────────────────────────────────────

pub const EXTRA_COLORS: &[Color] = &[
    Color::Cyan,
    Color::Magenta,
    Color::Green,
    Color::Yellow,
    Color::Red,
    Color::Blue,
];

// ── Symmetric log ─────────────────────────────────────────────────────────
// symlog(x, C) = sign(x) · log₁₀(1 + |x|/C)
//   - all positive & x >> C  →  behaves like log₁₀(x/C)
//   - near zero              →  approximately linear (x / (C·ln10))
//   - negative values        →  mirrors the positive side

pub fn symlog(x: f64, c: f64) -> f64 {
    x.signum() * (1.0 + x.abs() / c).log10()
}

pub fn symlog_inv(y: f64, c: f64) -> f64 {
    y.signum() * c * (10f64.powf(y.abs()) - 1.0)
}

// ── LogState ──────────────────────────────────────────────────────────────

/// Shared log-scale state, computed from the current metrics.
/// Used by both training and HPO modes.
pub struct LogState {
    pub log_scale: bool,
    /// true when any train/val loss ≤ 0 — triggers symlog instead of log₁₀
    pub has_nonpositive: bool,
    /// symlog threshold: smallest |loss| > 0 in the data
    pub symlog_c: f64,
}

impl LogState {
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

        Self {
            log_scale: false,
            has_nonpositive,
            symlog_c,
        }
    }

    /// Transform a loss value for the Y axis
    pub fn loss_y(&self, v: f64) -> f64 {
        if !self.log_scale {
            v
        } else if self.has_nonpositive {
            symlog(v, self.symlog_c)
        } else {
            v.max(1e-20).log10()
        }
    }

    /// Inverse transform: transformed Y → original loss value (for axis labels)
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

    /// Generate Y-axis labels showing original loss values
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

// ── CSV Loading ──────────────────────────────────────────────────────────

const KNOWN_COLS: &[&str] = &[
    "epoch",
    "train_loss",
    "val_loss",
    "lr",
    "max_grad_norm",
    "predicted_final_loss",
];

/// Load a metrics CSV file.
/// Returns `(rows, extra_column_names)`.
pub fn load_metrics_csv(path: &Path) -> (Vec<MetricRow>, Vec<String>) {
    let Ok(mut rdr) = csv::Reader::from_path(path) else {
        return (Vec::new(), Vec::new());
    };
    let Ok(headers) = rdr.headers().cloned() else {
        return (Vec::new(), Vec::new());
    };

    let extra_columns: Vec<String> = headers
        .iter()
        .filter(|h| !KNOWN_COLS.contains(&h))
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

        let extras: Vec<Option<f64>> =
            extra_columns.iter().map(|col| get(col)).collect();

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

// ── Helpers ────────────────────────────────────────────────────────────────

pub fn bounds_xy(d1: &[(f64, f64)], d2: &[(f64, f64)]) -> (f64, f64, f64) {
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

pub fn min_max_y(data: &[(f64, f64)]) -> (f64, f64) {
    let y_min = data.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let y_max = data
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);
    (y_min, y_max)
}

pub fn make_labels(lo: f64, hi: f64, n: usize, scientific: bool) -> Vec<String> {
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
pub fn make_inv_log10_labels(lo: f64, hi: f64, n: usize) -> Vec<String> {
    (0..=n)
        .map(|i| {
            let y = lo + (hi - lo) * i as f64 / n as f64;
            format!("{:.1e}", 10f64.powf(y))
        })
        .collect()
}

// ── Generalized Rendering Functions ───────────────────────────────────────

/// Render the training overview layout: loss/lr/grad charts.
pub fn render_training_overview(
    frame: &mut Frame,
    metrics: &[MetricRow],
    log_state: &LogState,
    area: Rect,
    overview_bounds: &[Option<(f64, f64)>; 3],
    focused_panel: Option<usize>,
) {
    use ratatui::{layout::Constraint, layout::Layout};

    let has_grad = metrics.iter().any(|m| m.max_grad_norm.is_some());

    if has_grad {
        let [loss_area, lr_area, grad_area] = Layout::vertical([
            Constraint::Percentage(50),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .areas(area);

        render_loss_chart(frame, metrics, log_state, loss_area, overview_bounds[0], focused_panel == Some(0));
        render_lr_chart(frame, metrics, log_state.log_scale, lr_area, overview_bounds[1], focused_panel == Some(1));
        render_grad_chart(frame, metrics, log_state.log_scale, grad_area, overview_bounds[2], focused_panel == Some(2));
    } else {
        let [loss_area, lr_area] = Layout::vertical([
            Constraint::Percentage(65),
            Constraint::Percentage(35),
        ])
        .areas(area);

        render_loss_chart(frame, metrics, log_state, loss_area, overview_bounds[0], focused_panel == Some(0));
        render_lr_chart(frame, metrics, log_state.log_scale, lr_area, overview_bounds[1], focused_panel == Some(1));
    }
}

/// Render the loss chart (train/val with optional predicted final loss line).
pub fn render_loss_chart(
    frame: &mut Frame,
    metrics: &[MetricRow],
    log_state: &LogState,
    area: Rect,
    y_bounds: Option<(f64, f64)>,
    focused: bool,
) {
    if metrics.is_empty() {
        frame.render_widget(
            Paragraph::new(" Waiting for metrics.csv data...")
                .block(Block::bordered().title(" Loss Curves ")),
            area,
        );
        return;
    }

    let train: Vec<(f64, f64)> = metrics
        .iter()
        .map(|m| (m.epoch, log_state.loss_y(m.train_loss)))
        .collect();

    let val: Vec<(f64, f64)> = metrics
        .iter()
        .map(|m| (m.epoch, log_state.loss_y(m.val_loss)))
        .collect();

    let (x_max, y_min, y_max) = bounds_xy(&train, &val);
    let (y_lo, y_hi) = if let Some((lo, hi)) = y_bounds {
        (lo, hi)
    } else {
        let y_pad = (y_max - y_min).max(1e-10) * 0.1;
        (y_min - y_pad, y_max + y_pad)
    };

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
    if let Some(pred) = metrics.last().and_then(|m| m.predicted_final_loss) {
        let pred_y = log_state.loss_y(pred);
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
    let y_labels = log_state.make_loss_labels(y_lo, y_hi, 5);

    // Inline legend in title bar — always visible
    let title = Line::from(vec![
        Span::raw(log_state.loss_title()),
        Span::styled("━ train ", Style::new().fg(Color::Cyan)),
        Span::styled("━ val ", Style::new().fg(Color::Yellow)),
    ]);

    let border_style = if focused {
        Style::new().fg(Color::Yellow)
    } else {
        Style::default()
    };

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(title).border_style(border_style))
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

/// Render the learning rate chart.
pub fn render_lr_chart(
    frame: &mut Frame,
    metrics: &[MetricRow],
    log_scale: bool,
    area: Rect,
    y_bounds: Option<(f64, f64)>,
    focused: bool,
) {
    if metrics.is_empty() {
        frame.render_widget(
            Paragraph::new(" Waiting...")
                .block(Block::bordered().title(" Learning Rate ")),
            area,
        );
        return;
    }

    let data: Vec<(f64, f64)> = metrics.iter().map(|m| (m.epoch, m.lr)).collect();
    render_positive_chart(frame, log_scale, area, &data, "Learning Rate", "lr", Color::Green, y_bounds, focused);
}

/// Render the gradient norm chart.
pub fn render_grad_chart(
    frame: &mut Frame,
    metrics: &[MetricRow],
    log_scale: bool,
    area: Rect,
    y_bounds: Option<(f64, f64)>,
    focused: bool,
) {
    let grad: Vec<(f64, f64)> = metrics
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

    render_positive_chart(frame, log_scale, area, &grad, "Gradient Norm", "\u{2016}\u{2207}\u{2016}", Color::Yellow, y_bounds, focused);
}

/// Shared renderer for always-positive metric charts (LR, grad norm).
/// Applies log₁₀ when `log_scale` is on.
pub fn render_positive_chart(
    frame: &mut Frame,
    log_scale: bool,
    area: Rect,
    raw_data: &[(f64, f64)],
    name: &str,
    y_title: &str,
    color: Color,
    y_bounds: Option<(f64, f64)>,
    focused: bool,
) {
    let data: Vec<(f64, f64)> = raw_data
        .iter()
        .map(|&(x, y)| {
            let y_t = if log_scale {
                y.max(1e-20).log10()
            } else {
                y
            };
            (x, y_t)
        })
        .collect();

    let x_max = data.last().map(|(x, _)| x + 1.0).unwrap_or(1.0);
    let (y_lo, y_hi) = if let Some((lo, hi)) = y_bounds {
        (lo, hi)
    } else {
        let (y_min, y_max) = min_max_y(&data);
        let y_pad = (y_max - y_min).max(1e-15) * 0.1;
        (y_min - y_pad, y_max + y_pad)
    };

    let datasets = vec![Dataset::default()
        .name(name)
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(color))
        .data(&data)];

    let title = if log_scale {
        format!(" {} (log\u{2081}\u{2080}) ", name)
    } else {
        format!(" {} ", name)
    };

    let x_labels = make_labels(0.0, x_max, 5, false);
    let y_labels = if log_scale {
        // Show original values via 10^y
        make_inv_log10_labels(y_lo, y_hi, 3)
    } else {
        make_labels(y_lo, y_hi, 3, true)
    };

    let border_style = if focused {
        Style::new().fg(Color::Yellow)
    } else {
        Style::default()
    };

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(title).border_style(border_style))
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

/// Chart renderer for data that may contain negative values.
/// Applies the same log/symlog transform as the loss chart.
pub fn render_generic_chart(
    frame: &mut Frame,
    log_state: &LogState,
    area: Rect,
    raw_data: &[(f64, f64)],
    name: &str,
    y_title: &str,
    color: Color,
    y_bounds: Option<(f64, f64)>,
    focused: bool,
) {
    let data: Vec<(f64, f64)> = raw_data
        .iter()
        .map(|&(x, y)| (x, log_state.loss_y(y)))
        .collect();

    let x_max = data.last().map(|(x, _)| x + 1.0).unwrap_or(1.0);
    let (y_lo, y_hi) = if let Some((lo, hi)) = y_bounds {
        (lo, hi)
    } else {
        let (y_min, y_max) = min_max_y(&data);
        let y_pad = (y_max - y_min).max(1e-10) * 0.1;
        (y_min - y_pad, y_max + y_pad)
    };

    let datasets = vec![Dataset::default()
        .name(name)
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::new().fg(color))
        .data(&data)];

    let title = if log_state.log_scale {
        format!(" {} (log\u{2081}\u{2080}) ", name)
    } else {
        format!(" {} ", name)
    };

    let x_labels = make_labels(0.0, x_max, 5, false);
    let y_labels = log_state.make_loss_labels(y_lo, y_hi, 5);

    let border_style = if focused {
        Style::new().fg(Color::Yellow)
    } else {
        Style::default()
    };

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(title).border_style(border_style))
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

// ── Tests ──────────────────────────────────────────────────────────────────

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
