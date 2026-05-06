# 2026-05-06

- Promote `LatestModelCallback` to a full training-state checkpoint (`latest_model.pt` now stores model + optimizer + scheduler + RNG + early-stopping state + best_value + config_hash)
- Add `Trainer.train(start_epoch=...)` so a run can pick up partway through the epoch loop
- Wire `--resume` flag into `cli.py train` and `main.py`; `run()` gains a `resume` parameter that loads `latest_model.pt` per seed and rehydrates optimizer/scheduler/RNG with config-hash verification
- `CheckpointManager` no longer writes a redundant `latest.pt`; `latest_model.pt` is the single resume entry point. `best.pt` and periodic `checkpoint_epoch_*.pt` remain opt-in via `checkpoint_config.enabled`

# 2026-04-08

- Add dual logging: `logging: wandb` (default) or `logging: tui` for agent-friendly terminal output
- Add CSVLoggingCallback (always active): writes `metrics.csv` every epoch with dynamic column expansion
- Add TUILoggingCallback: structured per-epoch terminal output replacing W&B
- Add LatestModelCallback (always active): saves `latest_model.pt` every epoch
- Add provenance tracking: `env_snapshot.yaml` + `run_metadata.yaml` per run
- Add Rust TUI monitor (`tools/monitor/`): real-time loss curve visualization from `metrics.csv`
- Add `doctor` CLI command: system environment health check
- Add `monitor` CLI command: launch TUI monitor with auto-detection
- Improve loss prediction: shifted exponential decay `L(t) = a·exp(-b·t) + c`
- Complete migration skill documentation (M1–M6) with detailed code-level migration steps
- Add migration enforcement: pre-push hook blocks push when source files change without migration doc updates

# 2025-04-14

- Change metric of pruner: `val_loss` -> `train_loss`

# 2024-12-16

- Add PFL (Predicted Final Loss) Pruner and its documentation
  - Add PFL Pruner to optimize hyperparameter search
  - Update default optimization template with PFL Pruner configuration
  - Add Pruner documentation in README.md under Appendix section

# 2024-09-22

- Early stop when loss becomes NaN
- Fix a bug in load_best_model

# 2024-09-20

- Add `utils.select_project`

# 2024-09-10

- Support `TPE.GridSampler`
  - Add `OptimizeConfig.grid_search_space`
  - Modify `OptimizeConfig._create_sampler`
  - **Caution**: For GridSampler, all variables in search_space should be categorical
