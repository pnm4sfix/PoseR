"""Score per-frame predictions against hand-labelled bouts."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .metrics import benchmark_model_performance
from .schemas.scoring import ClassNames, DecoderScore, LabelledBout


def labelled_bouts(classification_data: dict) -> list[LabelledBout]:
    """Flatten read_classification_h5's output into one bout per labelled span.

    Args:
        classification_data: {individual: {behaviour: bout}}, as
            core.io.read_classification_h5 returns it.
    """
    return [
        LabelledBout(
            start=int(bout["start"]),
            stop=int(bout["stop"]),
            name=str(bout["classification"]),
        )
        for bouts in classification_data.values()
        for bout in bouts.values()
    ]


def class_names_from_config(path: Path) -> ClassNames:
    """Read a decoder's class names and ignored classes from decoder_config.yml.

    classification_dict keys may be written as numbers or strings, and
    labels_to_ignore may name a class or give its number.
    """
    data_cfg = yaml.safe_load(Path(path).read_text())["data_cfg"]
    names = {
        int(index): str(name)
        for index, name in data_cfg["classification_dict"].items()
    }
    ignored = set()
    for label in data_cfg.get("labels_to_ignore") or []:
        if str(label) in names.values():
            ignored.add(str(label))
        elif str(label).isdigit() and int(label) in names:
            ignored.add(names[int(label)])
    return ClassNames(names=names, ignored=frozenset(ignored))


def class_names_from_checkpoint(checkpoint: dict) -> ClassNames | None:
    """Read class names stored in a decoder checkpoint, if it stores any.

    `poser model repair` writes them to data_cfg["label_dict"]. Older
    checkpoints store nothing, or a label_dict that maps numbers to numbers,
    and give None.
    """
    hp = checkpoint.get("hyper_parameters", {}) or {}
    data_cfg = hp.get("data_cfg", {}) or {}
    label_dict = data_cfg.get("label_dict") or {}
    if not label_dict or not all(isinstance(n, str) for n in label_dict.values()):
        return None
    names = {int(index): name for index, name in label_dict.items()}
    ignored = {
        str(label)
        for label in data_cfg.get("labels_to_ignore") or []
        if str(label) in names.values()
    }
    return ClassNames(names=names, ignored=frozenset(ignored))


def score_bouts(
    predictions: np.ndarray, bouts: list[LabelledBout], classes: ClassNames
) -> DecoderScore:
    """Score per-frame predictions against hand-labelled bouts.

    Each bout gets the label predicted most often across its frames. Bouts of
    an ignored class are left out, as they were in training, and so are bouts
    of a class the decoder does not know or with no predicted frames.

    Args:
        predictions: One predicted class index per frame.
        bouts: The hand-labelled bouts, from labelled_bouts.
        classes: The decoder's class names and ignored classes.

    Raises:
        ValueError: If no bout could be scored.
    """
    index_of = {name: index for index, name in classes.names.items()}
    truth, voted = [], []
    n_ignored = n_skipped = 0
    for bout in bouts:
        if bout.name in classes.ignored:
            n_ignored += 1
            continue
        # stop is one past the last frame of the bout.
        frames = predictions[max(0, bout.start) : min(len(predictions), bout.stop)]
        if bout.name not in index_of or len(frames) == 0:
            n_skipped += 1
            continue
        truth.append(index_of[bout.name])
        voted.append(Counter(frames.tolist()).most_common(1)[0][0])
    if not truth:
        raise ValueError("None of the hand-labelled bouts could be scored.")

    scored_classes = {
        index: name
        for index, name in classes.names.items()
        if name not in classes.ignored
    }
    result = benchmark_model_performance(
        np.array(voted), np.array(truth), scored_classes, report_as_dict=True
    )
    class_names = tuple(scored_classes[index] for index in sorted(scored_classes))
    return DecoderScore(
        accuracy=float(result["accuracy"]),
        balanced_accuracy=float(result["balanced_accuracy"]),
        confusion_matrix=result["confusion_matrix"],
        class_names=class_names,
        per_class={name: result["classification_report"][name] for name in class_names},
        n_scored=len(truth),
        n_ignored=n_ignored,
        n_skipped=n_skipped,
    )


def save_score_csv(score: DecoderScore, base: Path) -> list[Path]:
    """Write a score as <base>_metrics.csv and <base>_matrix.csv.

    The metrics file has one row per class, then accuracy and balanced
    accuracy in the f1 column, as sklearn's text report lays them out.

    Args:
        score: The score to write.
        base: Path whose stem names both files, e.g. rec_confusion_matrix.png.

    Returns:
        The two written paths.
    """
    base = Path(base)
    metrics_path = base.with_name(f"{base.stem}_metrics.csv")
    matrix_path = base.with_name(f"{base.stem}_matrix.csv")

    with open(metrics_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name in score.class_names:
            row = score.per_class[name]
            writer.writerow(
                [name, row["precision"], row["recall"], row["f1-score"],
                 int(row["support"])]
            )
        writer.writerow(["accuracy", "", "", score.accuracy, score.n_scored])
        writer.writerow(
            ["balanced accuracy", "", "", score.balanced_accuracy, score.n_scored]
        )

    with open(matrix_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true \\ predicted", *score.class_names])
        for name, counts in zip(score.class_names, score.confusion_matrix):
            writer.writerow([name, *counts.tolist()])
    return [metrics_path, matrix_path]
