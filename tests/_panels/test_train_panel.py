"""Tests for poser._panels.train_panel."""

from __future__ import annotations

from types import SimpleNamespace

from poser._panels.train_panel import TrainPanel
from poser.core.session import SessionManager


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
