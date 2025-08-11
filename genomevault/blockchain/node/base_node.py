from __future__ import annotations

"""Base Node module."""
from abc import ABC, abstractmethod


class BaseNode(ABC):
    """Minimal node lifecycle interface."""

    def __init__(self) -> None:
        """Initialize instance."""
        self.running = False

    def start(self) -> None:
        """Start."""
        self.running = True

    def stop(self) -> None:
        """Stop."""
        self.running = False

    @abstractmethod
    def handle(self, msg: dict) -> dict:
        """Handle incoming message and return response."""
