"""
Clinical validation module for GenomeVault
Uses REAL ZK proofs from the existing implementation
"""
from .core import ClinicalValidator
from .data_sources import NHANESDataSource, PimaDataSource
from .zk_wrapper import ZKProver, ProofData

__all__ = [
    'ClinicalValidator', 
    'NHANESDataSource', 
    'PimaDataSource',
    'ZKProver',
    'ProofData'
]
