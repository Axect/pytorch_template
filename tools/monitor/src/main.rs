mod charts;
mod training;
mod hpo;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;

// ── CLI ────────────────────────────────────────────────────────────────────

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

    /// Y-axis minimum for objective value charts
    #[arg(long)]
    y_min: Option<f64>,

    /// Y-axis maximum for objective value charts
    #[arg(long)]
    y_max: Option<f64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(db_path) = cli.hpo {
        let y_bounds = match (cli.y_min, cli.y_max) {
            (Some(lo), Some(hi)) => Some((lo, hi)),
            (Some(lo), None) => Some((lo, f64::INFINITY)),
            (None, Some(hi)) => Some((f64::NEG_INFINITY, hi)),
            (None, None) => None,
        };
        hpo::run_hpo(db_path, cli.study, Duration::from_millis(cli.interval), y_bounds)
    } else {
        let path = cli.path.expect("CSV path required in training mode");
        let csv_path = if path.is_dir() { path.join("metrics.csv") } else { path };
        training::run_training(csv_path, Duration::from_millis(cli.interval))
    }
}
