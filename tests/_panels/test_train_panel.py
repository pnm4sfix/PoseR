"""Tests for poser._panels.train_panel."""

from __future__ import annotations

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
