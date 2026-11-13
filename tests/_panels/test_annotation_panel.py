"""Tests for poser._panels.annotation_panel."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QPushButton

from poser._panels import annotation_panel
from poser._panels.annotation_panel import AnnotationPanel
from poser.core.session import SessionManager
from poser.core.settings import settings

from .conftest import Dialogs


def _panel(tmp_path) -> SimpleNamespace:
    """A stand-in panel: one bout labelled swim on a 10-frame, 3-node recording."""
    session = SessionManager()
    session.add(tmp_path / "rec_DLC.h5")
    session.activate(0)
    session.behaviour_labels = {0: "other", 1: "swim"}
    zeros = np.zeros((3, 10))  # (V, T)
    session.active.coords_data = {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}
    return SimpleNamespace(
        _session=session,
        _entry=session.active,
        _current_bouts=[{"start": 2, "end": 5, "label": 1}],
    )


def test_training_data_is_saved_under_the_chosen_name(tmp_path, monkeypatch):
    # The files were named after the pose file, in a folder picked each time.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "fish1_pose.npy"))
    monkeypatch.setattr(annotation_panel, "QFileDialog", dialogs)
    shown = []
    monkeypatch.setattr(
        annotation_panel,
        "QMessageBox",
        SimpleNamespace(
            information=lambda *args: shown.append(args[-1]),
            warning=lambda *args: shown.append(args[-1]),
            critical=lambda *args: shown.append(args[-1]),
        ),
    )

    AnnotationPanel._on_export(_panel(tmp_path))

    assert dialogs.opened == [tmp_path / "data" / "training" / "rec_DLC_pose.npy"]
    assert np.load(tmp_path / "fish1_pose.npy").shape == (3, 10, 3, 1)  # (C, T, V, M)
    labels = np.load(tmp_path / "fish1_labels.npy")
    assert labels.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    schema = json.loads((tmp_path / "fish1_schema.json").read_text())
    assert schema == {"0": "other", "1": "swim"}


def test_name_without_the_pose_suffix_still_pairs_for_training(tmp_path, monkeypatch):
    # The Train panel pairs <stem>_pose.npy with <stem>_labels.npy.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(
        annotation_panel, "QFileDialog", Dialogs(answer=str(tmp_path / "fish1"))
    )
    monkeypatch.setattr(
        annotation_panel, "QMessageBox", SimpleNamespace(information=lambda *a: None)
    )

    AnnotationPanel._on_export(_panel(tmp_path))

    assert (tmp_path / "fish1_pose.npy").is_file()
    assert (tmp_path / "fish1_labels.npy").is_file()


def test_export_question_says_unlabelled_bout_frames_are_ignored(
    tmp_path, monkeypatch
):
    # It said they would default to label 0 (other), but they are saved as -1.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(
        annotation_panel, "QFileDialog", Dialogs(answer=str(tmp_path / "fish1"))
    )
    asked = []

    def answer_yes(parent, title, text, buttons):
        asked.append(text)
        return annotation_panel.QMessageBox.Yes

    monkeypatch.setattr(
        annotation_panel,
        "QMessageBox",
        SimpleNamespace(Yes=1, No=2, question=answer_yes, information=lambda *a: None),
    )
    panel = _panel(tmp_path)
    panel._current_bouts.append({"start": 6, "end": 8})

    AnnotationPanel._on_export(panel)
    first_labels = np.load(tmp_path / "fish1_labels.npy")
    panel._current_bouts.append({"start": 9, "end": 10})
    AnnotationPanel._on_export(panel)

    assert asked == [
        "1 bout is still unlabelled. Its frames are saved as -1 and ignored in "
        "training. Continue?",
        "2 bouts are still unlabelled. Their frames are saved as -1 and ignored in "
        "training. Continue?",
    ]
    assert first_labels.tolist() == [0, 0, 1, 1, 1, 0, -1, -1, 0, 0]


def _labelling_panel(tmp_path, row: int, bouts: list[dict]) -> SimpleNamespace:
    """The stand-in panel listing bouts, with row selected and label 1 clicked."""
    panel = _panel(tmp_path)
    panel._current_bouts = bouts
    panel._bout_list = SimpleNamespace(
        currentRow=lambda: row,
        count=lambda: len(bouts),
        item=lambda r: SimpleNamespace(setText=lambda text: None),
        setCurrentRow=lambda r: None,
    )
    panel.sender = lambda: SimpleNamespace(property=lambda name: 1)
    panel._current_label_display = SimpleNamespace(setText=lambda text: None)
    panel._update_stats = lambda: None
    panel._emit_annotations = lambda: None
    return panel


def _messages(caplog, level: int) -> list[str]:
    """What the annotation panel logged at level."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "poser._panels.annotation_panel" and record.levelno == level
    ]


def test_assigned_label_is_logged_with_its_bout_and_frames(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = _labelling_panel(tmp_path, row=0, bouts=[{"start": 2, "end": 5}])

    AnnotationPanel._on_assign_label(panel)

    [message] = _messages(caplog, logging.INFO)
    assert "Assigned label 1 (swim) to bout 1, frames 2-5" in message


def test_label_click_with_no_bout_selected_logs_a_warning(tmp_path, caplog):
    # Nothing is assigned, so the bouts stay unlabelled.
    caplog.set_level(logging.DEBUG, logger="poser")
    panel = _labelling_panel(tmp_path, row=-1, bouts=[{"start": 2, "end": 5}])

    AnnotationPanel._on_assign_label(panel)

    [message] = _messages(caplog, logging.WARNING)
    assert "Label not assigned: no bout selected (row -1, 1 bouts loaded)" in message
    assert panel._current_bouts == [{"start": 2, "end": 5}]


def test_label_click_with_no_bout_selected_says_to_select_one(tmp_path, monkeypatch):
    # The click did nothing visible, so it looked lost.
    monkeypatch.setattr(annotation_panel, "QMessageBox", SimpleNamespace())  # no dialog
    panel = _labelling_panel(tmp_path, row=-1, bouts=[])
    shown = []
    panel._current_label_display = SimpleNamespace(setText=shown.append)

    AnnotationPanel._on_assign_label(panel)

    assert shown == ["Select a bout first"]


def test_behaviours_are_numbered_from_1_after_label_0_other(qtbot):
    # The first button was "[0] behaviour_0", and 0 is also every frame outside a bout.
    viewer = SimpleNamespace(dims=SimpleNamespace(set_point=lambda axis, frame: None))
    panel = AnnotationPanel(viewer, session=SessionManager())
    qtbot.addWidget(panel)
    panel.load_bouts([{"start": 2, "end": 5}])
    layout = panel._label_buttons_layout
    buttons = [layout.itemAt(i).widget() for i in range(layout.count())]

    assert [button.text() for button in buttons] == [
        "[0] other",
        "[1] behaviour_1",
        "[2] behaviour_2",
    ]
    buttons[1].click()
    assert panel._current_bouts == [{"start": 2, "end": 5, "label": 1}]


def test_label_0_stays_other_when_the_labels_are_set(tmp_path, caplog):
    # A behaviour given index 0 would share it with every frame outside a bout.
    caplog.set_level(logging.WARNING, logger="poser")
    panel = _panel(tmp_path)
    panel._rebuild_label_buttons = lambda: None

    AnnotationPanel.set_behaviour_labels(panel, {0: "rest", 1: "turn"})

    assert panel._session.behaviour_labels == {0: "other", 1: "turn"}
    [warning] = _messages(caplog, logging.WARNING)
    assert "Label 0 is always other; the name 'rest' given for it is not used" in (
        warning
    )

    AnnotationPanel.set_behaviour_labels(panel, {1: "swim", 2: "turn"})

    assert panel._session.behaviour_labels == {0: "other", 1: "swim", 2: "turn"}


def test_label_names_stay_in_the_session_when_the_panel_is_reopened(qtbot):
    # Each panel kept its own label map, so reopening the panel reset the names.
    session = SessionManager()
    viewer = SimpleNamespace(dims=SimpleNamespace(set_point=lambda axis, frame: None))
    panel = AnnotationPanel(viewer, session=session, behaviour_labels={1: "swim"})
    qtbot.addWidget(panel)
    panel._new_label_edit.setText("turn")
    panel._on_add_label()

    reopened = AnnotationPanel(viewer, session=session)
    qtbot.addWidget(reopened)

    assert session.behaviour_labels == {0: "other", 1: "swim", 2: "turn"}
    layout = reopened._label_buttons_layout
    assert [layout.itemAt(i).widget().text() for i in range(layout.count())] == [
        "[0] other",
        "[1] swim",
        "[2] turn",
    ]


def _removing_panel(tmp_path, chosen: int) -> SimpleNamespace:
    """The stand-in panel with labels swim (1) and turn (2), chosen picked to remove."""
    panel = _panel(tmp_path)  # lists the shown entry's one bout, labelled swim
    panel._session.behaviour_labels[2] = "turn"
    panel._entry.bouts = panel._current_bouts
    panel._remove_label_combo = SimpleNamespace(currentData=lambda: chosen)
    panel._rebuild_label_buttons = lambda: None
    panel.load_bouts = lambda bouts: None
    return panel


def test_remove_takes_the_chosen_label_and_unlabels_its_bouts(tmp_path, monkeypatch):
    # Remove always took the highest index, and bouts kept a label that had no name.
    asked = []

    def answer_yes(parent, title, text, buttons):
        asked.append(text)
        return annotation_panel.QMessageBox.Yes

    monkeypatch.setattr(
        annotation_panel,
        "QMessageBox",
        SimpleNamespace(Yes=1, No=2, question=answer_yes),
    )
    panel = _removing_panel(tmp_path, chosen=1)
    fish2 = panel._session.add(tmp_path / "fish2_DLC.h5")
    fish2.bouts = [
        {"start": 0, "end": 4, "label": 1, "label_source": "hand"},
        {"start": 6, "end": 9, "label": 2},
    ]
    listed = []
    panel.load_bouts = listed.append

    AnnotationPanel._on_remove_label(panel)

    assert panel._session.behaviour_labels == {0: "other", 2: "turn"}
    assert asked == [
        "2 bouts are labelled swim in this session. Remove the label and leave them "
        "unlabelled?"
    ]
    assert panel._entry.bouts == [{"start": 2, "end": 5}]
    assert fish2.bouts == [{"start": 0, "end": 4}, {"start": 6, "end": 9, "label": 2}]
    assert listed == [panel._entry.bouts]


def test_removing_a_label_that_bouts_carry_can_be_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(
        annotation_panel,
        "QMessageBox",
        SimpleNamespace(Yes=1, No=2, question=lambda *args: 2),
    )
    panel = _removing_panel(tmp_path, chosen=1)

    AnnotationPanel._on_remove_label(panel)

    assert panel._session.behaviour_labels == {0: "other", 1: "swim", 2: "turn"}
    assert panel._current_bouts == [{"start": 2, "end": 5, "label": 1}]


def test_removing_a_label_one_bout_carries_asks_about_that_bout(tmp_path, monkeypatch):
    # The question read "1 bouts in this session are labelled swim".
    asked = []

    def answer_no(parent, title, text, buttons):
        asked.append(text)
        return annotation_panel.QMessageBox.No

    monkeypatch.setattr(
        annotation_panel,
        "QMessageBox",
        SimpleNamespace(Yes=1, No=2, question=answer_no),
    )
    panel = _removing_panel(tmp_path, chosen=1)  # its one bout is labelled swim

    AnnotationPanel._on_remove_label(panel)

    assert asked == [
        "1 bout is labelled swim in this session. Remove the label and leave it "
        "unlabelled?"
    ]


def test_the_last_behaviour_label_is_not_removed(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="poser")
    shown = []
    monkeypatch.setattr(
        annotation_panel,
        "QMessageBox",
        SimpleNamespace(warning=lambda *args: shown.append(args[-1])),
    )
    panel = _removing_panel(tmp_path, chosen=1)
    del panel._session.behaviour_labels[2]

    AnnotationPanel._on_remove_label(panel)

    assert panel._session.behaviour_labels == {0: "other", 1: "swim"}
    assert shown == ["At least one behaviour label is required."]
    [warning] = _messages(caplog, logging.WARNING)
    assert "Behaviour label 1 (swim) not removed: at least one behaviour label is " in (
        warning
    )


def test_remove_lists_the_behaviour_labels_and_keeps_the_one_chosen(qtbot):
    # Label 0 (other) is never listed; rebuilding the list must not change the choice.
    viewer = SimpleNamespace(dims=SimpleNamespace(set_point=lambda axis, frame: None))
    panel = AnnotationPanel(viewer, session=SessionManager())
    qtbot.addWidget(panel)
    combo = panel._remove_label_combo
    combo.setCurrentIndex(1)  # [2] behaviour_2
    panel._new_label_edit.setText("swim")
    panel._on_add_label()

    assert [combo.itemText(i) for i in range(combo.count())] == [
        "[1] behaviour_1",
        "[2] behaviour_2",
        "[3] swim",
    ]
    [remove] = [b for b in panel.findChildren(QPushButton) if b.text() == "Remove"]
    remove.click()
    assert panel._session.behaviour_labels == {0: "other", 1: "behaviour_1", 3: "swim"}
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "[1] behaviour_1",
        "[3] swim",
    ]


def test_loaded_bouts_are_logged_with_their_label_counts(
    tmp_path, monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG, logger="poser")
    monkeypatch.setattr(
        annotation_panel,
        "QListWidgetItem",
        lambda text: SimpleNamespace(setData=lambda role, value: None),
    )
    panel = _panel(tmp_path)  # a 10-frame recording
    panel._bout_list = SimpleNamespace(
        currentItem=lambda: None,
        clear=lambda: None,
        addItem=lambda item: None,
        blockSignals=lambda block: None,
        setCurrentRow=lambda row: None,
    )
    panel._on_bout_selected = lambda row, seek: None
    panel._update_stats = lambda: None
    panel._bouts_to_labels = lambda: AnnotationPanel._bouts_to_labels(panel)
    panel._emit_annotations = lambda: AnnotationPanel._emit_annotations(panel)
    panel.annotations_changed = SimpleNamespace(emit=lambda labels: None)
    # Detectors pad bouts, so the last one can run past the end of the recording.
    bouts = [{"start": 2, "end": 5, "label": 1}, {"start": 6, "end": 14}]

    AnnotationPanel.load_bouts(panel, bouts)

    [loaded] = _messages(caplog, logging.INFO)
    assert "Loaded 2 bouts into the annotation list: 1 labelled, 1 unlabelled" in (
        loaded
    )
    [short] = _messages(caplog, logging.WARNING)
    assert "Label array has 10 frames, shorter than the last bout end 14" in short


def test_loading_selects_the_first_unlabelled_bout_and_leaves_the_viewer(qtbot):
    # A label click right after loading did nothing: no bout was selected.
    seeks = []
    viewer = SimpleNamespace(
        dims=SimpleNamespace(set_point=lambda axis, frame: seeks.append(frame))
    )
    panel = AnnotationPanel(viewer, session=SessionManager())
    qtbot.addWidget(panel)

    panel.load_bouts(
        [
            {"start": 2, "end": 5, "label": 1},
            {"start": 6, "end": 9},
            {"start": 10, "end": 12},
        ]
    )

    assert panel._bout_list.currentRow() == 1
    assert panel._current_label_display.text() == "Current: [-1] unlabelled"
    assert seeks == []
    panel._bout_list.setCurrentRow(2)  # as a click, an arrow key or Next does
    assert seeks == [10]


def test_reloading_keeps_the_selected_bout_while_it_is_listed(qtbot):
    viewer = SimpleNamespace(dims=SimpleNamespace(set_point=lambda axis, frame: None))
    panel = AnnotationPanel(viewer, session=SessionManager())
    qtbot.addWidget(panel)
    bouts = [{"start": 2, "end": 5}, {"start": 10, "end": 12}]
    panel.load_bouts(bouts)
    panel._bout_list.setCurrentRow(1)

    bouts.insert(0, {"start": 0, "end": 1})  # a bout added before the selected one
    panel.load_bouts(bouts)
    assert panel._bout_list.currentRow() == 2

    # Detecting again makes new bouts; the one with the same frames stays selected.
    panel.load_bouts([{"start": 0, "end": 1}, {"start": 10, "end": 12}])
    assert panel._bout_list.currentRow() == 1

    # Once it is gone and every bout is labelled, the first bout is selected.
    panel.load_bouts(
        [{"start": 2, "end": 5, "label": 1}, {"start": 6, "end": 9, "label": 1}]
    )
    assert panel._bout_list.currentRow() == 0

    panel.load_bouts([])
    assert panel._current_label_display.text() == "Current: —"


def test_label_track_is_as_long_as_a_data_panel_recording(tmp_path):
    # The Data panel keeps coords as (V, T) DataFrames; the track was V frames long.
    panel = _panel(tmp_path)  # one bout labelled swim, frames 2-5
    table = pd.DataFrame(np.zeros((3, 10)))  # (V, T)
    panel._session.active.coords_data = {"ind1": {"x": table, "y": table, "ci": table}}

    labels = AnnotationPanel._bouts_to_labels(panel)

    assert labels.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_labels_are_written_to_the_bouts_of_the_entry_shown(tmp_path, qtbot):
    # After another file was activated, the panel kept listing and labelling the
    # previous file's bouts.
    session = SessionManager()
    fish1 = session.add(tmp_path / "fish1_DLC.h5")
    fish2 = session.add(tmp_path / "fish2_DLC.h5")
    fish1.bouts = [{"start": 2, "end": 5}, {"start": 10, "end": 12}]
    fish2.bouts = [{"start": 0, "end": 4}, {"start": 10, "end": 12}]
    viewer = SimpleNamespace(dims=SimpleNamespace(set_point=lambda axis, frame: None))
    panel = AnnotationPanel(viewer, session=session)
    qtbot.addWidget(panel)
    panel.show_entry(fish1)
    panel._bout_list.setCurrentRow(1)

    panel.show_entry(fish2)

    # fish2's bout over the same frames is another bout, so it is not kept selected.
    assert panel._bout_list.currentRow() == 0
    layout = panel._label_buttons_layout
    layout.itemAt(1).widget().click()  # [1] behaviour_1
    assert fish2.bouts[0] == {"start": 0, "end": 4, "label": 1}
    assert fish1.bouts == [{"start": 2, "end": 5}, {"start": 10, "end": 12}]

    panel.show_entry(None)
    assert panel._bout_list.count() == 0


def test_load_bouts_lists_the_active_entry_even_without_bouts(
    tmp_path, qtbot, monkeypatch
):
    # The list kept the bouts of the entry activated before.
    shown = []
    monkeypatch.setattr(
        annotation_panel,
        "QMessageBox",
        SimpleNamespace(information=lambda *args: shown.append(args[1])),
    )
    session = SessionManager()
    fish1 = session.add(tmp_path / "fish1_DLC.h5")
    fish1.bouts = [{"start": 2, "end": 5}]
    session.add(tmp_path / "fish2_DLC.h5")
    viewer = SimpleNamespace(dims=SimpleNamespace(set_point=lambda axis, frame: None))
    panel = AnnotationPanel(viewer, session=session)
    qtbot.addWidget(panel)
    session.activate(0)
    panel._on_load_from_session()
    session.activate(1)  # no bouts on fish2 yet

    panel._on_load_from_session()

    assert panel._bout_list.count() == 0
    assert shown == ["No bouts"]


def test_a_loaded_session_shows_its_bouts_under_its_label_names(tmp_path):
    # Session files kept neither, so Load Session lost the bouts, labels and names.
    saved = SessionManager()
    saved.add(tmp_path / "fish1_DLC.h5")
    saved.activate(0)
    saved.behaviour_labels[3] = "swim"
    saved.active.bouts = [{"start": 2, "end": 5, "label": 3}, {"start": 6, "end": 9}]
    saved.save(tmp_path / "day1.json")
    session = SessionManager()
    listed, buttons = [], []
    panel = SimpleNamespace(
        _session=session,
        _entry=None,
        _bout_list=SimpleNamespace(clear=lambda: None),
        load_bouts=listed.append,
        _rebuild_label_buttons=lambda: buttons.append(dict(session.behaviour_labels)),
    )
    session.load(tmp_path / "day1.json")

    AnnotationPanel.show_entry(panel, session.active)  # as Load Session sends it

    assert listed == [[{"start": 2, "end": 5, "label": 3}, {"start": 6, "end": 9}]]
    assert buttons == [{0: "other", 1: "behaviour_1", 2: "behaviour_2", 3: "swim"}]


def test_label_track_and_training_data_come_from_the_entry_shown(
    tmp_path, monkeypatch
):
    # Both read the active entry, which need not be the one whose bouts are listed.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    dialogs = Dialogs(answer=str(tmp_path / "fish1"))
    monkeypatch.setattr(annotation_panel, "QFileDialog", dialogs)
    monkeypatch.setattr(
        annotation_panel, "QMessageBox", SimpleNamespace(information=lambda *a: None)
    )
    panel = _panel(tmp_path)  # lists a bout of a 10-frame recording
    longer = panel._session.add(tmp_path / "long_DLC.h5")
    zeros = np.zeros((3, 30))  # (V, T)
    longer.coords_data = {"ind1": {"x": zeros, "y": zeros, "ci": zeros}}
    panel._session.activate(1)

    labels = AnnotationPanel._bouts_to_labels(panel)
    AnnotationPanel._on_export(panel)

    assert len(labels) == 10
    assert dialogs.opened == [tmp_path / "data" / "training" / "rec_DLC_pose.npy"]
    assert np.load(tmp_path / "fish1_pose.npy").shape == (3, 10, 3, 1)  # (C, T, V, M)
