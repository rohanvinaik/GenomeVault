from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseCircuit(ABC):
    @abstractmethod
    def public_statement(self) -> Dict[str, Any]: ...
    @abstractmethod
    def witness(self) -> Dict[str, Any]: ...

    def prove(self) -> bytes:
        # Deterministic placeholder proof
        s = str(sorted(self.public_statement().items())).encode()
        w = str(sorted(self.witness().items())).encode()
        return b"CIRCUIT:" + s + b"|" + w

    def verify(self, proof: bytes) -> bool:
        return isinstance(proof, (bytes, bytearray)) and proof.startswith(b"CIRCUIT:")