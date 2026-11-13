"""Skeleton registry — YAML + Python entry-point discovery."""
from .registry import SkeletonRegistry, get_skeleton, list_skeletons
from .base import BaseSkeletonSpec

__all__ = ["SkeletonRegistry", "get_skeleton", "list_skeletons", "BaseSkeletonSpec"]
