"""
cli/main.py
~~~~~~~~~~~
Typer-based command-line interface for PoseR.

Entry-point:  ``poser``  (declared in ``setup.cfg``)

Commands
--------
``poser train``           - train a behaviour decoder from labelled pose data
``poser predict``         - run behaviour inference on a pose file
``poser batch``           - run pose estimation + behaviour decoding on many files
``poser finetune``        - fine-tune a YOLO pose model on exported frames
``poser install-torch``   - auto-detect CUDA and install matching PyTorch build
``poser skeleton list``   - list built-in skeleton definitions
``poser skeleton info``   - show details of a skeleton
``poser model list``      - list registered / discovered model architectures
``poser init``            - scaffold a new PoseR project directory
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="poser",
    help="PoseR - multi-species behaviour decoding from pose estimation data.",
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
# install-torch
# ---------------------------------------------------------------------------

# CUDA version → PyTorch wheel index mapping (extend as new builds are released)
_CUDA_INDEX: list[tuple[tuple[int, int], str]] = [
    ((12, 6), "https://download.pytorch.org/whl/cu126"),
    ((12, 4), "https://download.pytorch.org/whl/cu124"),
    ((12, 1), "https://download.pytorch.org/whl/cu121"),
    ((11, 8), "https://download.pytorch.org/whl/cu118"),
]


def _detect_cuda() -> tuple[str, str]:
    """Return (cuda_tag, index_url) by querying nvidia-smi, or CPU fallback."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "cpu", "https://download.pytorch.org/whl/cpu"

    if result.returncode != 0:
        return "cpu", "https://download.pytorch.org/whl/cpu"

    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
    if not m:
        return "cpu", "https://download.pytorch.org/whl/cpu"

    major, minor = int(m.group(1)), int(m.group(2))
    for (req_major, req_minor), url in _CUDA_INDEX:
        if (major, minor) >= (req_major, req_minor):
            tag = url.rsplit("/", 1)[-1]  # e.g. "cu126"
            return tag, url

    return "cpu", "https://download.pytorch.org/whl/cpu"


@app.command(name="install-torch")
def install_torch(
    force_cpu: bool = typer.Option(False, "--cpu", help="Force CPU-only install even if GPU detected."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the install command without running it."),
) -> None:
    """Auto-detect CUDA and install the matching PyTorch build.

    Queries [bold]nvidia-smi[/bold] to determine the installed CUDA version, then
    installs [cyan]torch[/cyan] and [cyan]torchvision[/cyan] from the appropriate
    PyTorch wheel index.  Falls back to CPU if no GPU is found.
    """
    if force_cpu:
        cuda_tag, index_url = "cpu", "https://download.pytorch.org/whl/cpu"
    else:
        cuda_tag, index_url = _detect_cuda()

    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch>=2.5", "torchvision>=0.20",
        "--extra-index-url", index_url,
    ]

    console.print(f"[bold]Detected:[/bold] {cuda_tag}")
    console.print(f"[bold]Command :[/bold] {' '.join(cmd)}\n")

    if dry_run:
        console.print("[yellow]Dry run - not installing.[/yellow]")
        return

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Install failed (exit code {exc.returncode}).[/red]")
        raise typer.Exit(exc.returncode)

    console.print("\n[green]OK PyTorch installed successfully.[/green]")
    console.print("  Verify with:  python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@app.command()
def train(
    files: List[Path] = typer.Argument(..., help="HDF5 or *_pose.npy files to train on."),
    config: Path = typer.Option("decoder_config.yml", "--config", "-c", help="TrainingConfig YAML."),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Override output directory."),
    epochs: Optional[int] = typer.Option(None, "--epochs", "-e", help="Override max_epochs."),
    run_name: Optional[str] = typer.Option(None, "--name", "-n", help="Run name / sub-directory."),
    num_class: Optional[int] = typer.Option(None, "--num-class", help="Override model num_class."),
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
        console.print("[yellow]Config file not found - using defaults.[/yellow]")

    # Apply CLI overrides
    if output_dir:
        cfg.output_dir = output_dir
    else:
        # Default: place runs alongside the training data, not in CWD
        cfg.output_dir = files[0].resolve().parent / "poser_runs"
    if epochs:
        cfg.trainer.max_epochs = epochs
    if run_name:
        cfg.run_name = run_name
    if num_class:
        cfg.model.num_class = num_class
    if auto_lr:
        cfg.optimiser.auto_lr = True

    # Prepare data
    console.print(f"Loading {len(files)} classification file(s)...")
    try:
        train_dl, val_dl = prepare_dataset(files, cfg)
    except Exception as exc:
        import traceback
        console.print(f"[red]Data error:[/red] {exc}")
        console.print(traceback.format_exc())
        raise typer.Exit(1)

    # Build model - prefer from_training_config so OptimiserConfig is wired in
    from poser.models.registry import list_models
    arch = cfg.model.architecture.lower()
    console.print(f"Building model: [bold]{arch}[/bold]")

    # Ensure built-in models are imported so @register_model decorators run
    try:
        import poser.models.st_gcn_aaai18_pylightning_3block  # noqa: F401
    except ImportError:
        pass

    from poser.models.registry import load_model as _load_model
    arch_cls = _load_model.__func__.__self__ if hasattr(_load_model, "__func__") else None  # type: ignore[attr-defined]

    # Try from_training_config first (preferred - wires OptimiserConfig)
    try:
        from poser.models.registry import _registry  # type: ignore[attr-defined]
        model_cls = _registry.get_class(arch)
        if hasattr(model_cls, "from_training_config"):
            model = model_cls.from_training_config(cfg)
            console.print("  [green]OK[/green] OptimiserConfig wired via from_training_config")
        else:
            # Fallback: plain instantiation with flat kwargs
            model = _load_model(
                arch,
                num_class=cfg.model.num_class,
                num_nodes=cfg.model.num_nodes,
                in_channels=cfg.model.in_channels,
                layout=cfg.model.layout,
            )
    except Exception as exc:
        import traceback
        console.print(f"[red]Model build error:[/red] {exc}")
        console.print(traceback.format_exc())
        raise typer.Exit(1)

    # Train
    trainer = PoseRTrainer(cfg)
    console.print(
        f"[bold]Training[/bold] {arch} "
        f"| classes={cfg.model.num_class} | epochs={cfg.trainer.max_epochs}"
    )
    try:
        trainer.fit(model, train_dl, val_dl)
    except Exception as exc:
        import traceback
        console.print(f"[red]Training error:[/red] {exc}")
        console.print(traceback.format_exc())
        raise typer.Exit(1)

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
# predict-npy  (frame-by-frame inference on a *_pose.npy file)
# ---------------------------------------------------------------------------

@app.command(name="predict-npy")
def predict_npy(
    pose_file: Path = typer.Argument(
        ..., help="Input *_pose.npy file, shape (C, T, V, M)."
    ),
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", "-w", help="Trained .ckpt file."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output path. Defaults to <pose_file>_predictions.npy and .csv."
    ),
    batch_size: int = typer.Option(64, "--batch", "-b", help="Inference batch size."),
    device: str = typer.Option(
        "auto", "--device", "-d", help="'auto', 'cpu', or 'cuda'."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run frame-by-frame behaviour inference on a *_pose.npy file.

    Loads the model architecture and hyperparameters directly from the
    checkpoint (no extra config needed).  Produces a .npy array of
    per-frame predicted class indices and a .csv with frame, label columns.
    """
    _setup_logging(verbose)

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    # ── Resolve device ──────────────────────────────────────────────────
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    console.print(f"Device: [cyan]{dev}[/cyan]")

    # ── Load pose data ───────────────────────────────────────────────────
    if not pose_file.exists():
        console.print(f"[red]File not found:[/red] {pose_file}")
        raise typer.Exit(1)
    pose = np.load(pose_file)
    if pose.ndim == 3:          # (C, T, V) — no M axis
        pose = pose[..., np.newaxis]
    if pose.ndim != 4:
        console.print(f"[red]Expected 4-D array (C, T, V, M), got shape {pose.shape}[/red]")
        raise typer.Exit(1)
    C, T_total, V, M = pose.shape
    console.print(f"Pose data: C={C}  T={T_total}  V={V}  M={M}")

    # ── Load model from checkpoint ────────────────────────────────────
    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found:[/red] {checkpoint}")
        raise typer.Exit(1)

    console.print(f"Loading model from [cyan]{checkpoint.name}[/cyan] ...")
    try:
        # Read hparams from checkpoint without fully loading weights yet
        raw_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hp = raw_ckpt.get("hyper_parameters", {})
        data_cfg = hp.get("data_cfg", {})
        T2        = int(data_cfg.get("T2", 100))
        transform = data_cfg.get("transform", ["center", "align", "pad"])
        center_node = int(hp.get("graph_cfg", {}).get("center", 0))
        head_node = int(data_cfg.get("head", 0))
        num_class = int(hp.get("num_class", 2))
    except Exception as exc:
        console.print(f"[red]Could not read checkpoint hparams:[/red] {exc}")
        raise typer.Exit(1)

    console.print(
        f"Checkpoint hparams: num_class={num_class}  T2={T2}  "
        f"center_node={center_node}  transforms={transform}"
    )

    try:
        from poser.models.st_gcn_aaai18_pylightning_3block import ST_GCN_18
        import types

        # 'hparams' was a required __init__ arg that got removed from the
        # checkpoint because it couldn't be pickled.  Provide a minimal stub
        # so load_from_checkpoint can reconstruct the model.
        _infer_hparams = types.SimpleNamespace(
            learning_rate=1e-4,          # unused in eval mode
            batch_size=batch_size,
            dropout=float(hp.get("dropout", 0.5)),
        )

        model = ST_GCN_18.load_from_checkpoint(
            str(checkpoint),
            map_location=dev,
            hparams=_infer_hparams,
        )
        model.eval()
        model.to(dev)
    except Exception as exc:
        import traceback as _tb
        console.print(f"[red]Failed to load model:[/red] {exc}")
        console.print(_tb.format_exc())
        raise typer.Exit(1)

    console.print("Model loaded.")

    # ── Build dataset (unlabelled sliding-window) ─────────────────────
    from poser.core.dataset import PoseDataset

    dummy_labels = np.zeros(T_total, dtype=np.int64)
    ds = PoseDataset(
        data=pose,
        labels=dummy_labels,
        preprocess_frame=True,
        window_size=T2,
        transform=transform,
        augmentation=None,
        center_node=center_node,
        head_node=head_node,
        T=T2,
        num_class=num_class,
        C=C,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Inference ─────────────────────────────────────────────────────
    all_preds: list = []
    n_batches = (T_total + batch_size - 1) // batch_size
    console.print(f"Running inference over {T_total} frames ({n_batches} batches)...")

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dl):
            x = x.to(dev)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            if batch_idx % max(1, n_batches // 10) == 0:
                console.print(
                    f"  {min((batch_idx+1)*batch_size, T_total)}/{T_total} frames done"
                )

    predictions = np.array(all_preds, dtype=np.int64)  # (T_total,)

    # ── Save outputs ──────────────────────────────────────────────────
    base = output or pose_file.parent / (pose_file.stem + "_predictions")
    npy_path = base.with_suffix(".npy")
    csv_path = base.with_suffix(".csv")

    np.save(npy_path, predictions)

    import csv
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "predicted_label"])
        for frame_idx, label in enumerate(predictions):
            writer.writerow([frame_idx, int(label)])

    console.print(f"\n[green]Done.[/green]  {T_total} frames predicted.")
    console.print(f"  .npy -> {npy_path}")
    console.print(f"  .csv -> {csv_path}")

    # Quick class distribution summary
    unique, counts = np.unique(predictions, return_counts=True)
    console.print("\nPredicted class distribution:")
    for cls_id, cnt in zip(unique, counts):
        pct = 100.0 * cnt / T_total
        console.print(f"  class {cls_id:3d}: {cnt:6d} frames  ({pct:.1f}%)")


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
    base_weights: Path = typer.Option(Path("yolo11m-pose.pt"), "--weights", "-w"),
    val_dir: Optional[Path] = typer.Option(None, "--val-dir"),
    num_keypoints: int = typer.Option(9, "--keypoints", "-k"),
    epochs: int = typer.Option(50, "--epochs", "-e"),
    batch_size: int = typer.Option(16, "--batch", "-b"),
    project: str = typer.Option("poser_yolo_finetune", "--project"),
    name: str = typer.Option("run", "--name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Fine-tune a YOLO11 pose model on exported frames."""
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
            str(s.head_node) if s.head_node is not None else "N/A",
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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    if not verbose:
        # Silence chatty third-party libraries but keep Lightning/torch visible
        for _noisy in ("urllib3", "PIL", "matplotlib", "numba", "h5py",
                       "fsspec", "botocore", "boto3"):
            logging.getLogger(_noisy).setLevel(logging.WARNING)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":
    main()
