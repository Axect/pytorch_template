# HPO Best Parameter Extraction

Procedure for extracting the best hyperparameters from a completed Optuna study and creating `best.yaml`.

---

## Step 1: Locate the Optuna Database

After HPO, Optuna creates a SQLite database:

```
<PROJECT>_Opt.db
```

The `_Opt` suffix is automatically appended by the CLI during HPO runs. Check for it:

```bash
ls *.db
```

---

## Step 2: Load Study and Extract Best Trial

```python
import optuna
import yaml

# ── Load study ──
PROJECT = "<PROJECT>_v<VERSION>_<MODEL>"
STUDY_NAME = "<MODEL>_TPE"  # Must match opt.yaml study_name
DB_PATH = f"{PROJECT}_Opt.db"

study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=f"sqlite:///{DB_PATH}"
)

# ── Best trial info ──
trial = study.best_trial
print(f"Best trial: #{trial.number}")
print(f"Best value: {trial.value}")
print(f"Group name: {trial.user_attrs.get('group_name', 'N/A')}")
print(f"\nParameters:")
for key, value in trial.params.items():
    print(f"  {key}: {value}")
```

---

## Step 3: Map Parameters to YAML Structure

Optuna flattens parameter names as `<category>_<param>`. Map them back:

```python
def map_params_to_yaml(params: dict) -> dict:
    """Map Optuna flat params to nested YAML structure.

    Example:
        {'net_config_layers': 4, 'optimizer_config_lr': 0.342}
        ->
        {'net_config': {'layers': 4}, 'optimizer_config': {'lr': 0.342}}
    """
    yaml_overrides = {}

    # Known categories (from OptimizeConfig.search_space keys)
    categories = ['net_config', 'optimizer_config', 'scheduler_config']

    for key, value in params.items():
        matched = False
        for category in categories:
            prefix = f"{category}_"
            if key.startswith(prefix):
                param_name = key[len(prefix):]
                if category not in yaml_overrides:
                    yaml_overrides[category] = {}
                yaml_overrides[category][param_name] = value
                matched = True
                break

        if not matched:
            print(f"Warning: Could not map parameter '{key}' to any category")

    return yaml_overrides
```

---

## Step 4: Create best.yaml

```python
import copy

# ── Load base run config ──
RUN_CONFIG_PATH = "configs/<DIR>/<model>_run.yaml"
BEST_CONFIG_PATH = "configs/<DIR>/<model>_best.yaml"
FULL_EPOCHS = 150  # Task-dependent: typically 100-500

with open(RUN_CONFIG_PATH, 'r') as f:
    base_config = yaml.safe_load(f)

# ── Apply HPO-found overrides ──
best_config = copy.deepcopy(base_config)
overrides = map_params_to_yaml(trial.params)

for category, params in overrides.items():
    if category in best_config:
        best_config[category].update(params)
    else:
        best_config[category] = params

# ── Update for full training ──
best_config['epochs'] = FULL_EPOCHS
best_config['seeds'] = [58, 89, 231, 928, 814]

# Sync scheduler total_steps with epochs
if 'scheduler_config' in best_config:
    best_config['scheduler_config']['total_steps'] = FULL_EPOCHS

# Remove _Opt from project name if present
best_config['project'] = best_config['project'].replace('_Opt', '')

# Enable early stopping for long training
best_config['early_stopping_config'] = {
    'enabled': True,
    'patience': 30,
    'mode': 'min',
    'min_delta': 0.0001,
}

# Enable checkpointing
best_config['checkpoint_config'] = {
    'enabled': True,
    'save_every_n_epochs': 25,
    'keep_last_k': 3,
    'save_best': True,
    'monitor': 'val_loss',
    'mode': 'min',
}

# ── Write best.yaml ──
with open(BEST_CONFIG_PATH, 'w') as f:
    yaml.dump(best_config, f, sort_keys=False, default_flow_style=False)

print(f"Created {BEST_CONFIG_PATH}")
print(f"  epochs: {FULL_EPOCHS}")
print(f"  seeds: {best_config['seeds']}")
for category, params in overrides.items():
    for k, v in params.items():
        print(f"  {category}.{k}: {v}")
```

---

## Complete One-Shot Script

Save as `extract_best.py` and run:

```python
#!/usr/bin/env python
"""Extract best HPO params and create best.yaml."""

import optuna
import yaml
import copy
import sys

# ── Configuration (edit these) ──
PROJECT_OPT = "<PROJECT>_v<VERSION>_<MODEL>_Opt"  # DB project name
STUDY_NAME = "<MODEL>_TPE"
RUN_CONFIG = "configs/<DIR>/<model>_run.yaml"
BEST_CONFIG = "configs/<DIR>/<model>_best.yaml"
FULL_EPOCHS = 150
SEEDS = [58, 89, 231, 928, 814]

CATEGORIES = ['net_config', 'optimizer_config', 'scheduler_config']


def map_params(params):
    result = {}
    for key, value in params.items():
        for cat in CATEGORIES:
            if key.startswith(f"{cat}_"):
                param = key[len(f"{cat}_"):]
                result.setdefault(cat, {})[param] = value
                break
    return result


def main():
    # Load study
    db_path = f"{PROJECT_OPT}.db"
    try:
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=f"sqlite:///{db_path}"
        )
    except Exception as e:
        print(f"Error loading study: {e}")
        print(f"Available studies in {db_path}:")
        summaries = optuna.study.get_all_study_summaries(f"sqlite:///{db_path}")
        for s in summaries:
            print(f"  - {s.study_name} ({s.n_trials} trials)")
        sys.exit(1)

    trial = study.best_trial
    print(f"Best trial #{trial.number}: {trial.value}")

    # Load base config
    with open(RUN_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides
    overrides = map_params(trial.params)
    for cat, params in overrides.items():
        if cat in config:
            config[cat].update(params)

    # Update for full training
    config['epochs'] = FULL_EPOCHS
    config['seeds'] = SEEDS
    config['project'] = config['project'].replace('_Opt', '')

    if 'scheduler_config' in config:
        config['scheduler_config']['total_steps'] = FULL_EPOCHS

    config['early_stopping_config'] = {
        'enabled': True, 'patience': 30, 'mode': 'min', 'min_delta': 0.0001
    }
    config['checkpoint_config'] = {
        'enabled': True, 'save_every_n_epochs': 25, 'keep_last_k': 3,
        'save_best': True, 'monitor': 'val_loss', 'mode': 'min'
    }

    # Write
    with open(BEST_CONFIG, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"\nCreated: {BEST_CONFIG}")
    print(f"  epochs: {FULL_EPOCHS}, seeds: {SEEDS}")
    for cat, params in overrides.items():
        for k, v in params.items():
            print(f"  {cat}.{k}: {v}")


if __name__ == "__main__":
    main()
```

---

## Edge Cases

### No Completed Trials

If all trials were pruned:
```python
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if not completed:
    print("No completed trials. Consider:")
    print("  - Reducing PFLPruner aggressiveness (increase n_startup_trials)")
    print("  - Increasing HPO epochs")
    print("  - Widening search space")
```

### Multiple Studies in Same DB

List all studies:
```python
summaries = optuna.study.get_all_study_summaries(f"sqlite:///{db_path}")
for s in summaries:
    print(f"{s.study_name}: {s.n_trials} trials, best={s.best_trial.value if s.best_trial else 'N/A'}")
```

### Verifying Best Config

After creating best.yaml, validate:
```bash
python -m cli validate configs/<DIR>/<model>_best.yaml
python -m cli preview configs/<DIR>/<model>_best.yaml
```
