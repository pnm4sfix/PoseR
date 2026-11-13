"""Tests for poser.core.session."""

from __future__ import annotations

import logging

from poser.core.session import SessionManager


def _session(n: int) -> SessionManager:
    session = SessionManager()
    for i in range(n):
        session.add(f"recording_{i}.h5")
    return session


def test_removing_an_earlier_entry_keeps_the_last_entry_active():
    # The active index used to stay put while the queue shifted under it,
    # so it pointed one past the end and session.active raised IndexError.
    session = _session(3)
    active = session.activate(2)

    session.remove(0)

    assert session.active is active


def test_removing_an_earlier_entry_keeps_the_same_recording_active():
    # With entries left after the stale index, the next recording silently
    # became the active one.
    session = _session(4)
    active = session.activate(1)

    session.remove(0)

    assert session.active is active


def test_removing_a_later_entry_keeps_the_same_recording_active():
    session = _session(3)
    active = session.activate(0)

    session.remove(2)

    assert session.active is active


def test_removing_the_active_entry_leaves_none_active():
    session = _session(3)
    session.activate(1)

    session.remove(1)

    assert session.active is None


def test_each_session_starts_with_its_own_behaviour_label_map():
    # Label names lived in the Annotation panel and were lost when it was reopened.
    first, second = SessionManager(), SessionManager()

    first.behaviour_labels[3] = "swim"

    assert second.behaviour_labels == {0: "other", 1: "behaviour_1", 2: "behaviour_2"}


def test_activating_an_entry_logs_its_bouts_and_coords(caplog):
    # The Annotation panel takes its bouts from whichever entry is active.
    caplog.set_level(logging.DEBUG, logger="poser")
    session = _session(2)
    session.entries[1].bouts = [{"start": 0, "end": 5}, {"start": 9, "end": 14}]
    caplog.clear()

    session.activate(1)

    [record] = [
        record
        for record in caplog.records
        if record.name == "poser.core.session" and record.levelno == logging.INFO
    ]
    assert "Activated entry 1 (recording_1)" in record.getMessage()
    assert "2 bouts, coords for 0 individual(s)" in record.getMessage()


def test_loading_a_missing_session_file_logs_a_warning(tmp_path, caplog):
    caplog.set_level(logging.DEBUG, logger="poser")
    session = _session(1)
    missing = tmp_path / "day1.json"

    session.load(missing)

    assert len(session.entries) == 1
    [record] = [
        record
        for record in caplog.records
        if record.name == "poser.core.session" and record.levelno == logging.WARNING
    ]
    assert f"Session file {missing} does not exist" in record.getMessage()


def test_restoring_a_session_is_logged_not_printed(tmp_path, caplog, capsys):
    caplog.set_level(logging.DEBUG, logger="poser")
    path = tmp_path / "day1.json"
    saved = _session(2)
    saved.activate(1)
    saved.entries[1].bouts = [{"start": 0, "end": 5}]
    saved.save(path)
    # Session files hold no bouts, so the Annotation panel finds none after a reload.
    assert (
        "poser.core.session",
        logging.DEBUG,
        "Session files keep no bouts or coords; bouts of 1 entries not saved",
    ) in caplog.record_tuples
    caplog.clear()

    SessionManager().load(path)

    assert capsys.readouterr().out == ""
    [record] = [
        record
        for record in caplog.records
        if record.name == "poser.core.session" and record.levelno == logging.INFO
    ]
    assert f"Session restored from {path}: 2 entries, active=1" in record.getMessage()
    assert (
        "poser.core.session",
        logging.DEBUG,
        "Restored entries have no bouts or coords: session files omit them",
    ) in caplog.record_tuples
