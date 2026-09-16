"""Exception hierarchy for PoseR.

Every error PoseR raises deliberately inherits from PoseRError, so a caller can
catch the whole family with one clause and still narrow when it needs to.
Errors from third-party libraries are wrapped rather than allowed to escape, so
the caller never has to know that a reader happens to use PyTables.
"""

from __future__ import annotations


class PoseRError(Exception):
    """Base class for every error PoseR raises deliberately."""


class PoseFormatError(PoseRError):
    """A pose file could not be read as any supported format."""


class UnsupportedFormatError(PoseFormatError):
    """A file extension is not one PoseR knows how to read.

    Narrower than PoseFormatError: the file was rejected on its name, before
    anything tried to parse it.
    """
