"""Tests for poser._panels.ethogram_panel."""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace

import numpy as np
import torch

from poser._panels import ethogram_panel
from poser._panels.ethogram_panel import EthogramPanel


def _messages(caplog, level: int) -> list[str]:
    """What the ethogram panel logged at level, in order."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "poser._panels.ethogram_panel" and record.levelno == level
    ]


def test_received_annotations_are_logged_with_frames_per_label(caplog):
    # The Annotation panel sends -1 for the frames of a bout not labelled yet,
    # which the ethogram draws grey, and 0 for the frames outside any bout.
    caplog.set_level(logging.DEBUG, logger="poser")
    sources = []
    panel = SimpleNamespace(
        _annotation_names={},
        _ann_label=SimpleNamespace(setText=sources.append),
        _refresh=lambda: None,
    )

    EthogramPanel.load_annotations(panel, np.array([0, -1, -1, -1, 1, 1, 0]))

    [message] = _messages(caplog, logging.INFO)
    assert "received annotations of shape (7,)" in message
    assert "frames per label {-1: 3, 0: 2, 1: 2}" in message
    assert "0 also covers frames outside any bout" in message
    assert sources == ["Annotation panel"]  # it stayed None


def test_none_clears_only_the_labels_the_annotation_panel_sent(caplog):
    # The panel sends None when it lists no bouts; the row kept the labels of the
    # recording shown before.
    caplog.set_level(logging.DEBUG, logger="poser")
    sources, redrawn = [], []
    panel = SimpleNamespace(
        _annotations=np.array([0, 1, 1]),
        _annotations_from_panel=False,  # a browsed labels file
        _ann_label=SimpleNamespace(setText=sources.append),
        _refresh=lambda: redrawn.append(True),
    )

    EthogramPanel.load_annotations(panel, None)
    browsed = panel._annotations
    panel._annotations_from_panel = True
    EthogramPanel.load_annotations(panel, None)

    assert browsed.tolist() == [0, 1, 1]
    assert panel._annotations is None
    assert sources == ["None"]
    assert redrawn == [True]
    assert _messages(caplog, logging.INFO) == [
        "Ethogram annotation row cleared: the Annotation panel lists no bouts"
    ]


def test_browsed_annotations_and_their_class_names_are_logged(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="poser")
    labels = tmp_path / "fish1_labels.npy"
    np.save(labels, np.array([0, 0, 1]))
    schema = tmp_path / "fish1_schema.json"
    schema.write_text(json.dumps({"0": "other", "1": "swim"}))
    monkeypatch.setattr(
        ethogram_panel,
        "QFileDialog",
        SimpleNamespace(getOpenFileName=lambda *args: (str(labels), "")),
    )
    panel = SimpleNamespace(
        _annotation_names={},
        _annotations_from_panel=True,
        _prediction_names={0: "forward"},
        _ann_label=SimpleNamespace(setText=lambda text: None),
        _ann_legend=SimpleNamespace(update_names=lambda names: None),
        _refresh=lambda: None,
    )
    panel._try_load_schema = lambda path: EthogramPanel._try_load_schema(panel, path)

    EthogramPanel._on_browse_annotations(panel)

    assert not panel._annotations_from_panel  # so the Annotation panel cannot clear it
    loaded, names = _messages(caplog, logging.INFO)
    assert f"Loaded annotations from {labels}: shape (3,)" in loaded
    assert "frames per label {0: 2, 1: 1}" in loaded
    assert names == f"Loaded class names from {schema}: {{0: 'other', 1: 'swim'}}"
    assert panel._annotation_names == {0: "other", 1: "swim"}
    assert panel._prediction_names == {0: "forward"}


def test_a_browsed_predictions_schema_names_only_the_prediction_row(
    tmp_path, monkeypatch
):
    # Both rows shared one name map, but label indices and decoder classes differ.
    predictions = tmp_path / "fish1_predictions.npy"
    np.save(predictions, np.array([0, 1, 1]))
    schema = {"0": "forward", "1": "left"}
    (tmp_path / "fish1_schema.json").write_text(json.dumps(schema))
    monkeypatch.setattr(
        ethogram_panel,
        "QFileDialog",
        SimpleNamespace(getOpenFileName=lambda *args: (str(predictions), "")),
    )
    panel = SimpleNamespace(
        _annotation_names={0: "other", 1: "swim"},
        _prediction_names={},
        _pred_label=SimpleNamespace(setText=lambda text: None),
        _pred_legend=SimpleNamespace(update_names=lambda names: None),
        _refresh=lambda: None,
    )
    panel._try_load_schema = lambda path: EthogramPanel._try_load_schema(panel, path)

    EthogramPanel._on_browse_predictions(panel)

    assert panel._prediction_names == {0: "forward", 1: "left"}
    assert panel._annotation_names == {0: "other", 1: "swim"}


def test_prediction_names_come_from_the_label_dict_of_the_checkpoint(tmp_path):
    # It looked for a 'class_names' key, which PoseR never writes.
    checkpoint = tmp_path / "decoder.ckpt"
    data_cfg = {"label_dict": {0: "forward", 1: "left"}}  # poser model repair writes it
    torch.save({"hyper_parameters": {"data_cfg": data_cfg}}, checkpoint)
    panel = SimpleNamespace(
        _annotation_names={0: "other", 1: "swim"},
        _prediction_names={},
        _pred_label=SimpleNamespace(setText=lambda text: None),
        _refresh=lambda: None,
    )

    EthogramPanel.load_predictions(panel, np.array([0, 1, 1]), checkpoint)

    assert panel._prediction_names == {0: "forward", 1: "left"}
    assert panel._annotation_names == {0: "other", 1: "swim"}

    older = tmp_path / "older.ckpt"  # its label_dict maps numbers to numbers
    torch.save({"hyper_parameters": {"data_cfg": {"label_dict": {0: 0, 1: 1}}}}, older)
    EthogramPanel.load_predictions(panel, np.array([1, 0, 0]), older)

    assert panel._prediction_names == {}  # the names before were another decoder's


def test_annotation_names_name_the_annotation_row_only():
    legends, redrawn, statuses = [], [], []
    panel = SimpleNamespace(
        _annotations=None,
        _annotation_names={},
        _prediction_names={0: "forward"},
        _ann_legend=SimpleNamespace(update_names=legends.append),
        _pred_legend=SimpleNamespace(update_names=legends.append),
        _refresh=lambda: redrawn.append(True),
        _update_frame_status=lambda: statuses.append(True),
    )

    EthogramPanel.set_annotation_names(panel, {0: "other", 1: "swim"})
    EthogramPanel.set_annotation_names(panel, {0: "other", 1: "swim"})  # sent again

    assert panel._annotation_names == {0: "other", 1: "swim"}
    assert panel._prediction_names == {0: "forward"}
    assert legends == [{0: "other", 1: "swim"}]  # once: unchanged names change nothing
    assert statuses == [True]
    assert redrawn == []  # no annotation row yet: a redraw would draw nothing

    panel._annotations = np.array([0, 1, 1])
    EthogramPanel.set_annotation_names(panel, {0: "other", 1: "dart"})

    # Names come before the labels of the entry shown: a redraw here drew the labels
    # shown before under them, and warned of each label the new names lack.
    assert redrawn == []
    assert statuses == [True, True]

    EthogramPanel.set_label_names(panel, {0: "rest"})  # names both rows, as it did

    assert panel._annotation_names == panel._prediction_names == {0: "rest"}


def test_the_status_strip_names_each_value_with_its_own_rows_names():
    shown = []
    panel = SimpleNamespace(
        _viewer=SimpleNamespace(dims=SimpleNamespace(current_step=(2,))),
        _annotations=np.array([0, 1, 1, -1]),
        _predictions=np.array([0, 0, 1, 1]),
        _annotation_names={0: "other", 1: "swim"},
        _prediction_names={0: "forward", 1: "left"},
        _frame_status=SimpleNamespace(setText=shown.append),
    )
    panel._label_name = lambda *args: EthogramPanel._label_name(panel, *args)

    EthogramPanel._update_frame_status(panel)
    panel._viewer.dims.current_step = (3,)
    EthogramPanel._update_frame_status(panel)

    texts = [re.sub(r"<[^>]+>|&nbsp;", " ", text).split() for text in shown]
    assert texts == [
        ["Frame", "2", "|", "Ann:", "swim", "|", "Pred:", "left"],
        ["Frame", "3", "|", "Ann:", "unlabelled", "|", "Pred:", "left"],
    ]


def test_a_redraw_gives_each_legend_its_own_rows_names():
    # There was one legend, for the names both rows shared.
    ann_legend, pred_legend = [], []
    axes = SimpleNamespace(
        cla=lambda: None,
        set_xlim=lambda *args: None,
        axvline=lambda *args, **kwargs: object(),
    )
    panel = SimpleNamespace(
        _viewer=SimpleNamespace(dims=SimpleNamespace(current_step=(0,))),
        _canvas=SimpleNamespace(draw=lambda: None),
        _ax_ann=axes,
        _ax_pred=axes,
        _cursor_lines=[],
        _annotations=np.array([0, 1, 1]),
        _predictions=np.array([0, 0, 1]),
        _annotation_names={0: "other", 1: "swim"},
        _prediction_names={0: "forward", 1: "left"},
        _ann_legend=SimpleNamespace(update_names=ann_legend.append),
        _pred_legend=SimpleNamespace(update_names=pred_legend.append),
        _style_axes=lambda: None,
        _draw_track=lambda ax, labels: None,
        _update_frame_status=lambda: None,
    )

    EthogramPanel._refresh(panel)

    assert ann_legend == [{0: "other", 1: "swim"}]
    assert pred_legend == [{0: "forward", 1: "left"}]


def test_an_empty_ethogram_is_redrawn_without_a_warning(caplog):
    # A recording with no bouts and no predictions yet empties both rows.
    caplog.set_level(logging.DEBUG, logger="poser")
    axes = SimpleNamespace(cla=lambda: None, axvline=lambda *args, **kwargs: object())
    panel = SimpleNamespace(
        _viewer=SimpleNamespace(dims=SimpleNamespace(current_step=(0,))),
        _canvas=SimpleNamespace(draw=lambda: None),
        _ax_ann=axes,
        _ax_pred=axes,
        _cursor_lines=[],
        _annotations=None,
        _predictions=None,
        _annotation_names={},
        _prediction_names={},
        _ann_legend=SimpleNamespace(update_names=lambda names: None),
        _pred_legend=SimpleNamespace(update_names=lambda names: None),
        _style_axes=lambda: None,
        _update_frame_status=lambda: None,
    )

    EthogramPanel._refresh(panel)

    assert _messages(caplog, logging.WARNING) == []
    assert "Nothing to draw" in " ".join(_messages(caplog, logging.DEBUG))


def test_unnamed_classes_are_looked_up_in_their_own_rows_names(caplog):
    # Label 1 has a name on the annotation row; the decoder's class 1 has none.
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = SimpleNamespace(
        _ax_ann=object(),
        _annotation_names={0: "other", 1: "swim"},
        _prediction_names={0: "forward"},
    )
    ax = SimpleNamespace(broken_barh=lambda *args, **kwargs: None)  # predictions row

    EthogramPanel._draw_track(panel, ax, np.array([0, 1, 1]))

    assert _messages(caplog, logging.WARNING) == [
        "Classes [1] in the predictions row are not in its names {0: 'forward'}; "
        "shown as cls<N>"
    ]


def test_refresh_without_a_canvas_warns_that_nothing_is_drawn(caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = SimpleNamespace(
        _canvas=None,
        _annotations=np.zeros(5, dtype=np.int64),
        _predictions=None,
        _annotation_names={},
        _prediction_names={},
    )

    EthogramPanel._refresh(panel)

    [message] = _messages(caplog, logging.WARNING)
    assert "No matplotlib canvas" in message
