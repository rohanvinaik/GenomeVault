"""
Clinical SNP Database Module

Provides queryable clinical variant database functionality for GenomeVault.
"""

from .database import (
    ClinicalSNP,
    ClinicalCondition,
    ClinicalAnnotation,
    PopulationFrequency,
    FunctionalImpact,
    ClinicalSNPDatabase,
    ClinicalDatabaseBuilder
)

__all__ = [
    'ClinicalSNP',
    'ClinicalCondition',
    'ClinicalAnnotation',
    'PopulationFrequency',
    'FunctionalImpact',
    'ClinicalSNPDatabase',
    'ClinicalDatabaseBuilder',
]
