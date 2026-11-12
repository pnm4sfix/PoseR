"""Stand-ins shared by the panel tests, most of which run without a QApplication."""

from __future__ import annotations

from pathlib import Path


class Dialogs:
    """Stands in for QFileDialog: records where each dialog opened, then answers.

    An empty answer is the user pressing Cancel.
    """

    DontConfirmOverwrite = 0x4  # a QFileDialog option, passed through as options

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.opened: list[Path] = []

    def getSaveFileName(self, parent, caption, directory="", filter="", options=None):
        self.opened.append(Path(directory))
        return self.answer, ""

    def getExistingDirectory(self, parent, caption, directory=""):
        self.opened.append(Path(directory))
        return self.answer
