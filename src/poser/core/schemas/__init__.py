"""Typed data models shared across the core layer.

Modules here hold data only. They import stdlib and poser.core.exceptions and
nothing else, so they stay free of cycles as the rest of core grows models.
"""

from __future__ import annotations

from .batch import BatchMode, BatchResult

__all__ = ["BatchMode", "BatchResult"]
