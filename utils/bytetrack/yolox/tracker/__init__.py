"""Tracker package initialization."""
try:
    from .byte_tracker import BYTETracker
    __all__ = ["BYTETracker"]
except ImportError:
    __all__ = []
