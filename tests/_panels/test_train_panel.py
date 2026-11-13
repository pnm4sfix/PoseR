"""Tests for poser._panels.train_panel."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from poser._panels import train_panel
from poser._panels.train_panel import TrainPanel
from poser.core.session import SessionManager
from poser.core.settings import settings


class _Label:
    """Stands in for the QLabel that lists the chosen files."""

    def setText(self, text: str) -> None:
        self.text = text


def test_use_session_files_takes_done_entries_as_paths(tmp_path):
    # pose_path is a str and the file label reads .name from each file, so the
    # button raised AttributeError as soon as one entry was marked done.
    session = SessionManager()
    session.add(str(tmp_path / "done.h5"))
    session.add(str(tmp_path / "pending.h5"))
    session.mark_done(0)
    # The two methods under test, run on a stand-in so no QApplication is needed.
    panel = SimpleNamespace(_session=session, _file_list_label=_Label())
    panel._update_file_label = lambda: TrainPanel._update_file_label(panel)

    TrainPanel._on_use_session_files(panel)

    assert panel._training_files == [tmp_path / "done.h5"]
    assert panel._file_list_label.text == "1 file(s): done.h5"


def test_training_runs_are_written_under_data_runs(tmp_path, monkeypatch):
    # They used to go to poser_runs/ beside the first training file.
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    launched = []
    monkeypatch.setattr(
        train_panel,
        "subprocess",
        SimpleNamespace(
            Popen=lambda cmd, **kwargs: launched.append(cmd), PIPE=-1, STDOUT=-2
        ),
    )
    monkeypatch.setattr(
        train_panel,
        "threading",
        SimpleNamespace(Thread=lambda **kwargs: SimpleNamespace(start=lambda: None)),
    )
    panel = SimpleNamespace(
        _training_files=[tmp_path / "training" / "rec_pose.npy"],
        _training_proc=None,
        _epochs_spin=SimpleNamespace(value=lambda: 1),
        _run_name_edit=SimpleNamespace(text=lambda: "run"),
        _num_class_spin=SimpleNamespace(value=lambda: 2),
        _config_path=None,
        _auto_lr_cb=SimpleNamespace(isChecked=lambda: False),
        _log_text=SimpleNamespace(clear=lambda: None),
        _log_signal=SimpleNamespace(emit=lambda text: None),
        _btn_train=SimpleNamespace(setEnabled=lambda enabled: None),
        _btn_stop=SimpleNamespace(setEnabled=lambda enabled: None),
        _stream_output=lambda: None,
    )

    TrainPanel._on_start_training(panel)

    [cmd] = launched
    assert cmd[cmd.index("--output") + 1] == str(tmp_path / "data" / "runs")


def test_launching_training_logs_its_command(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(
        train_panel,
        "subprocess",
        SimpleNamespace(Popen=lambda cmd, **kwargs: None, PIPE=-1, STDOUT=-2),
    )
    monkeypatch.setattr(
        train_panel,
        "threading",
        SimpleNamespace(Thread=lambda **kwargs: SimpleNamespace(start=lambda: None)),
    )
    panel = SimpleNamespace(
        _training_files=[tmp_path / "training" / "rec_pose.npy"],
        _training_proc=None,
        _epochs_spin=SimpleNamespace(value=lambda: 1),
        _run_name_edit=SimpleNamespace(text=lambda: "run"),
        _num_class_spin=SimpleNamespace(value=lambda: 2),
        _config_path=None,
        _auto_lr_cb=SimpleNamespace(isChecked=lambda: False),
        _log_text=SimpleNamespace(clear=lambda: None),
        _log_signal=SimpleNamespace(emit=lambda text: None),
        _btn_train=SimpleNamespace(setEnabled=lambda enabled: None),
        _btn_stop=SimpleNamespace(setEnabled=lambda enabled: None),
        _stream_output=lambda: None,
    )

    TrainPanel._on_start_training(panel)

    [record] = [r for r in caplog.records if "Training launched" in r.getMessage()]
    assert record.name == "poser._panels.train_panel"
    assert record.levelno == logging.INFO
    message = record.getMessage()
    assert f"train {tmp_path / 'training' / 'rec_pose.npy'} --epochs 1" in message
    assert f"--output {tmp_path / 'data' / 'runs'}" in message


def test_session_files_chosen_for_training_are_logged_in_full(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    session = SessionManager()
    session.add(str(tmp_path / "done.h5"))
    session.add(str(tmp_path / "pending.h5"))
    session.mark_done(0)
    panel = SimpleNamespace(_session=session, _file_list_label=_Label())
    panel._update_file_label = lambda: TrainPanel._update_file_label(panel)

    TrainPanel._on_use_session_files(panel)

    [record] = [r for r in caplog.records if "Training files set" in r.getMessage()]
    assert record.name == "poser._panels.train_panel"
    assert record.levelno == logging.INFO
    assert record.getMessage().endswith(f"session file(s): {tmp_path / 'done.h5'}")


def test_starting_without_training_files_logs_why_nothing_ran(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    warned = []
    monkeypatch.setattr(
        train_panel,
        "QMessageBox",
        SimpleNamespace(warning=lambda *args: warned.append(args)),
    )
    panel = SimpleNamespace(_training_files=[], _training_proc=None, _config_path=None)

    TrainPanel._on_start_training(panel)

    [record] = [r for r in caplog.records if "no training files" in r.getMessage()]
    assert record.name == "poser._panels.train_panel"
    assert record.levelno == logging.WARNING
    assert len(warned) == 1


def test_a_schema_with_a_removed_label_sets_num_class_past_its_highest_index(
    tmp_path, monkeypatch
):
    # Removing a behaviour label leaves a gap, so counting the names gave too few.
    for name in ("fish1_pose.npy", "fish1_labels.npy"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "fish1_schema.json").write_text('{"0": "other", "3": "swim"}')
    monkeypatch.setattr(
        train_panel,
        "QFileDialog",
        SimpleNamespace(getExistingDirectory=lambda *args: str(tmp_path)),
    )
    chosen = []
    panel = SimpleNamespace(
        _file_list_label=_Label(),
        _num_class_spin=SimpleNamespace(
            setValue=chosen.append, value=lambda: chosen[-1]
        ),
        _log_signal=SimpleNamespace(emit=lambda text: None),
    )
    panel._update_file_label = lambda: TrainPanel._update_file_label(panel)

    TrainPanel._on_add_numpy_folder(panel)

    assert chosen == [4]  # labels 0 to 3
