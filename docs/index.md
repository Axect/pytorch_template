---
title: Home
nav_order: 0
---

# PyTorch Template — The Human Skill Guide

Your AI agent reads `.claude/skills/pytorch-train/SKILL.md` to know how to run experiments.
These docs are the human equivalent — they teach you the same pipeline, with the **why** that machines don't need.

## Two Skills, One Pipeline

| | AI Agent Skill | Human Skill (these docs) |
|---|---|---|
| **Location** | `.claude/skills/pytorch-train/` | `docs/` |
| **Reads** | Config rules, param ranges, CLI commands | Workflow intuition, design decisions, trade-offs |
| **Learns** | What to do | Why to do it |
| **Format** | Imperative instructions | Tutorial with examples |

## The Pipeline

Both skills follow the same 7-phase pipeline:

```
Phase 1: Config Creation      → Chapter 2
Phase 2: Pre-flight Check     → Chapter 1
Phase 3: Training              → Chapter 1, 3
Phase 4: HPO with Optuna      → Chapter 4
Phase 5: HPO Analysis          → Chapter 4
Phase 6: Final Training        → Chapter 1
Phase 7: Analysis              → Chapter 1
```

## Chapters

1. **[The Full Pipeline](01_pipeline.html)** — End-to-end walkthrough from config to analysis
2. **[Configuration Deep Dive](02_config.html)** — RunConfig, OptimizeConfig, data loading, validation tiers
3. **[Callback System](03_callbacks.html)** — 9 built-in callbacks, priority ordering, writing your own
4. **[Hyperparameter Optimization](04_hpo.html)** — Search spaces, PFL pruner, hpo-report, extracting best params
5. **[Customization Guide](05_customization.html)** — Custom models, data loaders, loss functions, metrics

## Quick Reference

```bash
python -m cli doctor              # Check environment
python -m cli preflight <config>  # Pre-flight check (1 batch forward+backward)
python -m cli preview <config>    # Show model architecture
python -m cli train <config>      # Train
python -m cli hpo-report          # Analyze HPO results
python -m cli analyze             # Analyze trained model
```
