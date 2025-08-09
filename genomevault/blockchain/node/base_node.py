from __future__ import annotations
from abc import ABC, abstractmethod

class BaseNode(ABC):
    """Minimal node lifecycle interface."""

    def __init__(self) -> None:
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    @abstractmethod
    def handle(self, msg: dict) -> dict:
        """Handle incoming message and return response."""