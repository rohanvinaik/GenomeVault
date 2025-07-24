"""
Circuit Factory for creating circuit instances.
Provides a centralized way to create and manage circuit types.
"""

from typing import Any, Dict, Type

from ..proofs.models import CircuitType
from .base import BaseCircuit
from .biomarkers import ClinicalBiomarkerCircuit
from .diabetes import DiabetesRiskCircuit


class CircuitFactory:
    """Factory for creating circuit instances"""

    _circuits: Dict[CircuitType, Type[BaseCircuit]] = {
        CircuitType.DIABETES_RISK: DiabetesRiskCircuit,
        CircuitType.BIOMARKER_THRESHOLD: ClinicalBiomarkerCircuit,
    }

    @classmethod
    def register(cls, circuit_type: CircuitType, circuit_class: Type[BaseCircuit]):
        """Register a new circuit type"""
        cls._circuits[circuit_type] = circuit_class

    @classmethod
    def create(cls, circuit_type: CircuitType, **kwargs) -> BaseCircuit:
        """Create a circuit instance"""
        if circuit_type not in cls._circuits:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

        circuit_class = cls._circuits[circuit_type]

        # Handle special cases that need additional parameters
        if circuit_type == CircuitType.BIOMARKER_THRESHOLD:
            biomarker_name = kwargs.get("biomarker_name", "generic")
            return circuit_class(biomarker_name)
        else:
            return circuit_class()

    @classmethod
    def list_available(cls) -> list:
        """List all available circuit types"""
        return list(cls._circuits.keys())


# Convenience function
def create_circuit(circuit_type: CircuitType, **kwargs) -> BaseCircuit:
    """Create a circuit instance using the factory"""
    return CircuitFactory.create(circuit_type, **kwargs)
