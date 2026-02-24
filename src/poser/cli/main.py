"""
cli/main.py
~~~~~~~~~~~
Typer-based command-line interface for PoseR.

Entry-point:  ``poser``  (declared in ``setup.cfg``)

Commands
--------
``poser train``          — train a behaviour decoder from labelled pose data
``poser predict``        — run behaviour inference on a pose file
``poser batch``          — run pose estimation + behaviour decoding on many files
``poser finetune``       — fine-tune a YOLO pose model on exported frames
``poser skeleton list``  — list built-in skeleton definitions
``poser skeleton info``  — show details of a skeleton
``poser model list``     — list registered / discovered model architectures
``poser init``           — scaffold a new PoseR project directory
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="poser",
    help="PoseR — multi-species behaviour decoding from pose estimation data.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

skeleton_app = typer.Typer(help="Manage skeleton definitions.", no_args_is_help=True)
model_app = typer.Typer(help="Manage model architectures.", no_args_is_help=True)
app.add_typer(skeleton_app, name="skeleton")
app.add_typer(model_app, name="model")

console = Console()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    files: List[Path] = typer.Argument(..., help="HDF5 classification files to train on."),
    config: Path = typer.Option("decoder_config.yml", "--config", "-c", help="TrainingConfig YAML."),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Override output directory."),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Override max_epochs."),
    run_name: Optional[str] = typer.Option(None, "--name", "-n", help="Run name / sub-directory."),
    auto_lr: bool = typer.Option(False, "--auto-lr", help="Run LR-finder before training."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train a PoseR behaviour decoder from labelled pose files."""
    _setup_logging(verbose)

    from poser.training.config import TrainingConfig
    from poser.training.trainer import PoseRTrainer
    from poser.training.data_prep import prepare_dataset

    # Load / create config
    if config.exists():
        cfg = TrainingConfig.from_yaml(config)
        console.print(f"[green]Loaded config:[/green] {config}")
    else:
        cfg = TrainingConfig()
        console.print("[yellow]Config file not found — using defaults.[/yellow]")

    # Apply CLI overrides
    if output_dir:
        cfg.output_dir = output_dir
    if epochs:
        cfg.trainer.max_epochs = epochs
    if run_name:
        cfg.run_name = run_name
    if auto_lr:
        cfg.optimiser.auto_lr = True

    # Prepare data
    console.print(f"Loading {len(files)} classification file(s)...")
    try:
        train_dl, val_dl = prepare_dataset(files, cfg)
    except ValueError as exc:
        console.print(f"[red]Data error:[/red] {exc}")
        raise typer.Exit(1)

    # Build model
    from poser.models.registry import load_model
    model = load_model(
        cfg.model.architecture,
        num_class=cfg.model.num_class,
        num_nodes=cfg.model.num_nodes,
        in_channels=cfg.model.in_channels,
        layout=cfg.model.layout,
    )

    # Train
    trainer = PoseRTrainer(cfg)
    console.print(
        f"[bold]Training[/bold] {cfg.model.architecture} "
        f"| classes={cfg.model.num_class} | epochs={cfg.trainer.max_epochs}"
    )
    trainer.fit(model, train_dl, val_dl)

    best = trainer.best_checkpoint
    console.print(f"\n[green]Training complete.[/green]  Best checkpoint: {best}")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

@app.command()
def predict(
    pose_file: Path = typer.Argument(..., help="Input pose HDF5 or CSV file."),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-w", help="Model .ckpt / .pt file."),
    config: Path = typer.Option("decoder_config.yml", "--config", "-c"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output HDF5 path."),
    individual: str = typer.Option("ind1", "--individual", "-i"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run behaviour inference on a single pose file."""
    _setup_logging(verbose)

    from poser.api import PoseR

    cfg_path = config if config.exists() else None
    poser = PoseR(config=cfg_path)
    result = poser.predict(
        pose_file=pose_file,
        checkpoint=checkpoint,
        individual=individual,
    )

    out_path = output or pose_file.with_suffix(".predictions.json")
    with open(out_path, "w") as fh:
        json.dump({str(k): int(v) for k, v in result.items()}, fh, indent=2)
    console.print(f"[green]Predictions saved to:[/green] {out_path}")


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

@app.command()
def batch(
    pose_files: List[Path] = typer.Argument(..., help="Pose files to process."),
    mode: str = typer.Option("behaviour", "--mode", "-m",
                             help="'behaviour' or 'pose_estimation'."),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-w"),
    config: Path = typer.Option("decoder_config.yml", "--config", "-c"),
    output_dir: Path = typer.Option(Path("batch_output"), "--output", "-o"),
    n_individuals: int = typer.Option(1, "--individuals", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Batch process multiple pose files."""
    _setup_logging(verbose)

    from poser.core.batch import BatchJob
    from poser.training.config import TrainingConfig

    cfg = TrainingConfig.from_yaml(config) if config.exists() else TrainingConfig()

    def _progress(done: int, total: int, path: str) -> None:
        console.print(f"  [{done}/{total}] {path}")

    job = BatchJob(
        pose_files=list(pose_files),
        mode=mode,
        checkpoint=checkpoint,
        config=cfg,
        output_dir=output_dir,
        n_individuals=n_individuals,
        progress_callback=_progress,
    )
    results = job.run()

    n_ok = sum(1 for r in results if r.success)
    console.print(f"\n[green]{n_ok}/{len(results)} files processed successfully.[/green]")
    console.print(f"Manifest: {output_dir / 'batch_manifest.csv'}")


# ---------------------------------------------------------------------------
# finetune
# ---------------------------------------------------------------------------

@app.command()
def finetune(
    images_dir: Path = typer.Argument(..., help="YOLO-format training images directory."),
    base_weights: Path = typer.Option(Path("yolov8m-pose.pt"), "--weights", "-w"),
    val_dir: Optional[Path] = typer.Option(None, "--val-dir"),
    num_keypoints: int = typer.Option(9, "--keypoints", "-k"),
    epochs: int = typer.Option(50, "--epochs", "-e"),
    batch_size: int = typer.Option(16, "--batch", "-b"),
    project: str = typer.Option("poser_yolo_finetune", "--project"),
    name: str = typer.Option("run", "--name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fine-tune a YOLOv8 pose model on exported frames."""
    _setup_logging(verbose)

    from poser.training.finetune_yolo import finetune_yolo

    run_dir = finetune_yolo(
        images_dir=images_dir,
        val_dir=val_dir,
        base_weights=base_weights,
        num_keypoints=num_keypoints,
        epochs=epochs,
        batch=batch_size,
        project=project,
        name=name,
    )
    console.print(f"[green]Fine-tuning complete.[/green]  Outputs: {run_dir}")


# ---------------------------------------------------------------------------
# skeleton sub-commands
# ---------------------------------------------------------------------------

@skeleton_app.command("list")
def skeleton_list() -> None:
    """List all registered skeleton definitions."""
    from poser.skeletons.registry import list_skeletons

    skeletons = list_skeletons()
    table = Table(title="Registered Skeletons", show_lines=True)
    table.add_column("Name", style="cyan")
    table.add_column("Nodes", justify="right")
    table.add_column("Center")
    table.add_column("Head")

    for name in skeletons:
        from poser.skeletons.registry import get_skeleton
        s = get_skeleton(name)
        table.add_row(
            s.name,
            str(s.num_nodes),
            str(s.center_node),
            str(s.head_node) if s.head_node is not None else "—",
        )
    console.print(table)


@skeleton_app.command("info")
def skeleton_info(
    name: str = typer.Argument(..., help="Skeleton name (e.g. 'zebrafish').")
) -> None:
    """Show detailed information about a skeleton."""
    from poser.skeletons.registry import get_skeleton

    try:
        s = get_skeleton(name)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{s.name}[/bold]  ({s.num_nodes} nodes)")
    console.print(f"  Center node : {s.center_node}")
    console.print(f"  Head node   : {s.head_node}")
    console.print(f"  Strategy    : {s.partition_strategy}")
    console.print(f"\n  Nodes:")
    for i, n in enumerate(s.node_names or []):
        console.print(f"    {i:2d}  {n}")
    console.print(f"\n  Edges: {s.edges}")


@skeleton_app.command("validate")
def skeleton_validate(
    yaml_path: Path = typer.Argument(..., help="Path to a skeleton YAML file.")
) -> None:
    """Validate a custom skeleton YAML file."""
    from poser.skeletons.registry import _registry

    try:
        spec = _registry.validate_yaml(yaml_path)
        console.print(f"[green]Valid skeleton:[/green] {spec.name}  ({spec.num_nodes} nodes)")
    except Exception as exc:
        console.print(f"[red]Validation failed:[/red] {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# model sub-commands
# ---------------------------------------------------------------------------

@model_app.command("list")
def model_list() -> None:
    """List registered model architectures and discovered checkpoints."""
    from poser.models.registry import _registry, list_checkpoints

    archs = _registry.list_archs()
    table = Table(title="Registered Architectures")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Description")

    for arch in archs:
        info = _registry.arch_info(arch)
        table.add_row(info["name"], info["model_version"], info["description"])
    console.print(table)

    ckpts = list_checkpoints()
    if ckpts:
        console.print(f"\n[bold]Discovered checkpoints ({len(ckpts)}):[/bold]")
        for p in ckpts:
            console.print(f"  {p}")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init(
    project_dir: Path = typer.Argument(Path("."), help="Project directory to initialise."),
    species: str = typer.Option("zebrafish", "--species", "-s"),
    num_classes: int = typer.Option(2, "--classes", "-c"),
) -> None:
    """Scaffold a new PoseR project directory with a starter config."""
    from poser.training.config import TrainingConfig, ModelConfig, BehaviourSchema

    project_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainingConfig(
        model=ModelConfig(layout=species, num_class=num_classes),
        behaviour_schema=BehaviourSchema(
            labels={i: f"behaviour_{i}" for i in range(num_classes)}
        ),
        species=species,
        run_name="run_01",
    )
    config_path = project_dir / "decoder_config.yml"
    cfg.to_yaml(config_path)

    console.print(f"[green]Initialised PoseR project in:[/green] {project_dir}")
    console.print(f"  Config: {config_path}")
    console.print(f"  Edit [cyan]decoder_config.yml[/cyan] to customise your training settings.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:  # pragma: no cover
    app()
