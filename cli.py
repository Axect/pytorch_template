"""
Typer-based CLI entrypoint for the PyTorch Template project.

Usage:
    python cli.py train config.yaml
    python cli.py validate config.yaml
    python cli.py preview config.yaml
    python cli.py doctor
    python cli.py analyze --project MyProject --group exp1 --seed 42
"""

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="PyTorch Template CLI")
console = Console()


@app.command()
def train(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
    device: str = typer.Option(None, help="Device override (e.g. 'cuda:0' or 'cpu')"),
    optimize_config: str = typer.Option(None, help="Path to optimization config for HPO"),
):
    """Train a model using the given run configuration."""
    try:
        from torch.utils.data import DataLoader

        from config import RunConfig, OptimizeConfig
        from util import load_data, run

        base_config = RunConfig.from_yaml(run_config)

        if device:
            base_config = base_config.with_overrides(device=device)

        ds_train, ds_val = load_data()
        dl_train = DataLoader(ds_train, batch_size=base_config.batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=base_config.batch_size, shuffle=False)

        if optimize_config:
            opt_config = OptimizeConfig.from_yaml(optimize_config)
            pruner = opt_config.create_pruner()

            def objective(trial, base_config, opt_config, dl_train, dl_val):
                params = opt_config.suggest_params(trial)

                overrides = {"project": f"{base_config.project}_Opt"}
                for category, category_params in params.items():
                    overrides[category] = category_params

                trial_config = base_config.with_overrides(**overrides)
                group_name = trial_config.gen_group_name()
                group_name += f"[{trial.number}]"

                trial.set_user_attr("group_name", group_name)

                return run(
                    trial_config, dl_train, dl_val, group_name, trial=trial, pruner=pruner
                )

            study = opt_config.create_study(project=f"{base_config.project}_Opt")
            study.optimize(
                lambda trial: objective(
                    trial, base_config, opt_config, dl_train, dl_val
                ),
                n_trials=opt_config.trials,
            )

            console.print("\n[bold green]Best trial:[/bold green]")
            trial = study.best_trial
            console.print(f"  Value: {trial.value}")
            console.print("  Params:")
            for key, value in trial.params.items():
                console.print(f"    {key}: {value}")
            console.print(
                f"  Path: runs/{base_config.project}_Opt/{trial.user_attrs['group_name']}"
            )
        else:
            run(base_config, dl_train, dl_val)
            console.print("[bold green]Training complete.[/bold green]")

    except Exception:
        console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def validate(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
):
    """Validate a run configuration for structural and runtime correctness."""
    try:
        from config import RunConfig

        config = RunConfig.from_yaml(run_config)
        config.validate_for_execution()
        console.print(f"[bold green]PASS[/bold green] — config '{run_config}' is valid.")

    except Exception as e:
        console.print(f"[bold red]FAIL[/bold red] — {e}")
        raise typer.Exit(code=1)


@app.command()
def preview(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
):
    """Preview the model, optimizer, scheduler, and criterion from a config."""
    try:
        from config import RunConfig

        config = RunConfig.from_yaml(run_config)

        model = config.create_model()
        optimizer = config.create_optimizer(model)
        scheduler = config.create_scheduler(optimizer)
        criterion = config.create_criterion()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        table = Table(title="Run Preview", show_lines=True)
        table.add_column("Property", style="bold cyan")
        table.add_column("Value")

        table.add_row("Model", repr(model))
        table.add_row("Total parameters", f"{total_params:,}")
        table.add_row("Trainable parameters", f"{trainable_params:,}")
        table.add_row("Optimizer", f"{type(optimizer).__name__} {optimizer.defaults}")
        table.add_row("Scheduler", f"{type(scheduler).__name__} {scheduler.state_dict()}")
        table.add_row("Criterion", f"{type(criterion).__name__}")
        table.add_row("Device", config.device)

        console.print(table)

    except Exception:
        console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def doctor():
    """Check system environment: Python, PyTorch, CUDA, packages, and wandb."""
    from provenance import capture_environment

    env = capture_environment()

    table = Table(title="System Doctor", show_lines=True)
    table.add_column("Check", style="bold cyan")
    table.add_column("Status")

    # Python version
    table.add_row("Python version", f"[green]{env['python_version']}[/green]")

    # PyTorch version
    torch_ver = env.get("torch_version", "not installed")
    if torch_ver == "not installed":
        table.add_row("PyTorch", "[red]not installed[/red]")
    else:
        table.add_row("PyTorch", f"[green]{torch_ver}[/green]")

    # CUDA
    cuda_available = env.get("cuda_available", False)
    if cuda_available:
        cuda_ver = env.get("cuda_version", "N/A")
        table.add_row("CUDA", f"[green]available (v{cuda_ver})[/green]")
        gpu_devices = env.get("gpu_devices", [])
        for i, gpu in enumerate(gpu_devices):
            table.add_row(
                f"  GPU {i}",
                f"[green]{gpu['name']} ({gpu['memory_total_mb']} MB)[/green]",
            )
    else:
        table.add_row("CUDA", "[yellow]not available[/yellow]")

    # wandb login status
    try:
        import wandb
        api_key = wandb.api.api_key
        if api_key:
            table.add_row("wandb", "[green]logged in[/green]")
        else:
            table.add_row("wandb", "[red]not logged in[/red]")
    except Exception:
        table.add_row("wandb", "[red]not installed or error[/red]")

    # Required packages
    required_packages = [
        "torch", "numpy", "optuna", "wandb", "tqdm", "rich", "beaupy", "scienceplots",
    ]
    for pkg in required_packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "ok")
            table.add_row(f"  {pkg}", f"[green]{version}[/green]")
        except ImportError:
            table.add_row(f"  {pkg}", "[red]missing[/red]")

    console.print(table)


@app.command()
def analyze(
    project: str = typer.Option(None, help="Project name (non-interactive)"),
    group: str = typer.Option(None, help="Group name (non-interactive)"),
    seed: str = typer.Option(None, help="Seed (non-interactive)"),
    device: str = typer.Option("cpu", help="Device for analysis"),
):
    """Analyze a trained model — interactive or non-interactive."""
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        from util import (
            select_project,
            select_group,
            select_seed,
            load_model,
            load_data,
        )

        if project is None:
            console.print("Select a project:")
            project = select_project()
        if group is None:
            console.print("Select a group:")
            group = select_group(project)
        if seed is None:
            console.print("Select a seed:")
            seed = select_seed(project, group)

        console.print(
            f"[bold]Analyzing[/bold] {project}/{group}/{seed} on [cyan]{device}[/cyan]"
        )

        model, config = load_model(project, group, seed)
        model = model.to(device)
        model.eval()

        _, ds_val = load_data()
        dl_val = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False)

        total_loss = 0.0
        count = 0
        with torch.inference_mode():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = F.mse_loss(y_pred, y)
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count if count > 0 else float("inf")
        console.print(f"Validation MSE Loss: [bold]{avg_loss:.6f}[/bold]")

    except Exception:
        console.print_exception()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
