"""Tests for poser.core.session."""

from __future__ import annotations

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
