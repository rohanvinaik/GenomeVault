"""
GenomeVault Proteomics Processing

Handles proteomics data processing including mass spectrometry data analysis,
protein quantification, and post-translational modification detection.
"""

import os
import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from scipy import stats
import tempfile
import re

from ..utils import get_logger, get_config, secure_hash
from ..utils.logging import log_operation

logger = get_logger(__name__)
config = get_config()


@dataclass
class ProteinMeasurement:
    """Single protein measurement"""
    protein_id: str
    gene_id: str
    protein_name: str
    abundance: float  # Normalized abundance
    raw_intensity: float
    peptide_count: int
    unique_peptides: int
    sequence_coverage: float  # Percentage of protein sequence covered
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'protein_id': self.protein_id,
            'gene_id': self.gene_id,
            'protein_name': self.protein_name,
            'abundance': self.abundance,
            'raw_intensity': self.raw_intensity,
            'peptide_count': self.peptide_count,
            'unique_peptides': self.unique_peptides,
            'sequence_coverage': self.sequence_coverage,
            'modifications': self.modifications,
            'confidence': self.confidence
        }


@dataclass
class Peptide:
    """Peptide identification"""
    sequence: str
    protein_id: str
    start_position: int
    end_position: int
    mass: float
    charge: int
    intensity: float
    retention_time: float
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    
    @property
    def length(self) -> int:
        """Peptide length"""
        return len(self.sequence)
    
    def get_modified_sequence(self) -> str:
        """Get sequence with modification annotations"""
        if not self.modifications:
            return self.sequence
        
        # Add modification annotations
        mod_seq = list(self.sequence)
        for mod in sorted(self.modifications, key=lambda x: x['position'], reverse=True):
            pos = mod['position']
            mod_type = mod['type']
            if 0 <= pos < len(mod_seq):
                mod_seq[pos] = f"{mod_seq[pos]}[{mod_type}]"
        
        return ''.join(mod_seq)


@dataclass
class ProteomicsProfile:
    """Complete proteomics profile"""
    sample_id: str
    proteins: List[ProteinMeasurement]
    peptides: List[Peptide]
    total_proteins: int
    total_peptides: int
    quantification_method: str
    normalization_method: str
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_protein_matrix(self) -> pd.DataFrame:
        """Get protein abundance matrix"""
        data = [p.to_dict() for p in self.proteins]
        return pd.DataFrame(data)
    
    def filter_by_confidence(self, min_confidence: float = 0.95) -> List[ProteinMeasurement]:
        """Filter proteins by confidence score"""
        return [p for p in self.proteins if p.confidence >= min_confidence]
    
    def get_modification_summary(self) -> Dict[str, int]:
        """Get summary of post-translational modifications"""
        mod_counts = {}
        for protein in self.proteins:
            for mod in protein.modifications:
                mod_type = mod.get('type', 'unknown')
                mod_counts[mod_type] = mod_counts.get(mod_type, 0) + 1
        return mod_counts


class ProteomicsProcessor:
    """Main processor for proteomics data"""
    
    SUPPORTED_FORMATS = {'.mzML', '.mzXML', '.mgf', '.raw', '.txt'}
    COMMON_MODIFICATIONS = {
        'Phosphorylation': {'mass': 79.966331, 'residues': ['S', 'T', 'Y']},
        'Acetylation': {'mass': 42.010565, 'residues': ['K']},
        'Methylation': {'mass': 14.015650, 'residues': ['K', 'R']},
        'Ubiquitination': {'mass': 114.042927, 'residues': ['K']},
        'Oxidation': {'mass': 15.994915, 'residues': ['M']}
    }
    
    def __init__(self,
                 database_path: Optional[Path] = None,
                 temp_dir: Optional[Path] = None,
                 max_threads: Optional[int] = None):
        """
        Initialize proteomics processor
        
        Args:
            database_path: Path to protein database (FASTA)
            temp_dir: Temporary directory for processing
            max_threads: Maximum threads to use
        """
        self.database_path = database_path or self._get_default_database()
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "genomevault_proteomics"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_threads = max_threads or config.processing.max_cores
        
        # Load protein database
        self.protein_db = self._load_protein_database()
    
    def _get_default_database(self) -> Path:
        """Get default protein database"""
        db_dir = config.storage.data_dir / "databases"
        return db_dir / "uniprot_human.fasta"
    
    def _load_protein_database(self) -> Dict[str, Dict[str, Any]]:
        """Load protein database"""
        protein_db = {}
        
        if not self.database_path.exists():
            logger.warning("Protein database not found")
            return protein_db
        
        # Simple FASTA parser
        current_id = None
        current_seq = []
        
        with open(self.database_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous protein
                    if current_id and current_seq:
                        protein_db[current_id] = {
                            'sequence': ''.join(current_seq),
                            'length': len(''.join(current_seq))
                        }
                    
                    # Parse header
                    parts = line[1:].split('|')
                    if len(parts) >= 2:
                        current_id = parts[1]  # UniProt ID
                    else:
                        current_id = line[1:].split()[0]
                    
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # Save last protein
            if current_id and current_seq:
                protein_db[current_id] = {
                    'sequence': ''.join(current_seq),
                    'length': len(''.join(current_seq))
                }
        
        logger.info(f"Loaded {len(protein_db)} proteins from database")
        return protein_db
    
    @log_operation("process_proteomics")
    def process(self, 
                input_path: Path, 
                sample_id: str,
                quantification_method: str = "label_free") -> ProteomicsProfile:
        """
        Process proteomics data
        
        Args:
            input_path: Path to mass spec data
            sample_id: Sample identifier
            quantification_method: Quantification method (label_free, tmt, silac)
            
        Returns:
            ProteomicsProfile with quantified proteins
        """
        logger.info(f"Processing proteomics data for sample {sample_id}")
        
        # Validate input
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create working directory
        work_dir = self.temp_dir / f"prot_{sample_id}_{os.getpid()}"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Process based on file format
            if input_path.suffix.lower() == '.txt':
                # Assume MaxQuant output
                proteins, peptides = self._process_maxquant_output(input_path)
            elif input_path.suffix.lower() in {'.mzml', '.mzxml'}:
                # Process raw MS data (simplified)
                proteins, peptides = self._process_ms_data(input_path, work_dir)
            else:
                raise ValueError(f"Unsupported file format: {input_path.suffix}")
            
            # Normalize protein abundances
            normalized_proteins = self._normalize_abundances(
                proteins, method=quantification_method
            )
            
            # Detect modifications
            modified_proteins = self._detect_modifications(normalized_proteins, peptides)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(modified_proteins, peptides)
            
            # Create profile
            profile = ProteomicsProfile(
                sample_id=sample_id,
                proteins=modified_proteins,
                peptides=peptides,
                total_proteins=len(modified_proteins),
                total_peptides=len(peptides),
                quantification_method=quantification_method,
                normalization_method='median_centering',
                quality_metrics=quality_metrics,
                metadata={
                    'database': str(self.database_path),
                    'processor_version': '1.0.0'
                }
            )
            
            logger.info(f"Proteomics processing complete. {len(profile.proteins)} proteins quantified")
            return profile
            
        finally:
            # Cleanup
            if work_dir.exists():
                import shutil
                shutil.rmtree(work_dir)
    
    def _process_maxquant_output(self, txt_path: Path) -> Tuple[List[ProteinMeasurement], List[Peptide]]:
        """Process MaxQuant proteinGroups.txt output"""
        # Read protein groups
        df = pd.read_csv(txt_path, sep='\t')
        
        proteins = []
        peptides = []
        
        # Extract protein information
        for _, row in df.iterrows():
            # Skip reverse hits and contaminants
            if pd.notna(row.get('Reverse')) or pd.notna(row.get('Potential contaminant')):
                continue
            
            # Parse protein IDs
            protein_ids = str(row.get('Majority protein IDs', '')).split(';')
            if not protein_ids or protein_ids == ['']:
                continue
            
            protein_id = protein_ids[0]
            
            # Get gene name
            gene_names = str(row.get('Gene names', '')).split(';')
            gene_id = gene_names[0] if gene_names and gene_names[0] else protein_id
            
            # Get protein name
            protein_names = str(row.get('Protein names', '')).split(';')
            protein_name = protein_names[0] if protein_names and protein_names[0] else protein_id
            
            # Get quantification
            intensity = row.get('Intensity', 0)
            if intensity <= 0:
                continue
            
            # Create protein measurement
            protein = ProteinMeasurement(
                protein_id=protein_id,
                gene_id=gene_id,
                protein_name=protein_name,
                abundance=np.log2(intensity + 1),  # Log transform
                raw_intensity=intensity,
                peptide_count=int(row.get('Peptide counts (all)', 0)),
                unique_peptides=int(row.get('Peptide counts (unique)', 0)),
                sequence_coverage=float(row.get('Sequence coverage [%]', 0)),
                confidence=1.0 - float(row.get('PEP', 0))  # Posterior error probability
            )
            proteins.append(protein)
        
        return proteins, peptides
    
    def _process_ms_data(self, ms_path: Path, work_dir: Path) -> Tuple[List[ProteinMeasurement], List[Peptide]]:
        """Process raw MS data (simplified implementation)"""
        # This is a placeholder - real implementation would use tools like
        # SearchGUI/PeptideShaker, Comet, or MSFragger
        
        logger.warning("Raw MS data processing not fully implemented. Returning mock data.")
        
        # Generate mock data for demonstration
        proteins = []
        peptides = []
        
        # Create some example proteins
        example_proteins = [
            ('P04264', 'KRT1', 'Keratin-1', 1000000),
            ('P02768', 'ALB', 'Albumin', 5000000),
            ('P00738', 'HP', 'Haptoglobin', 800000)
        ]
        
        for protein_id, gene_id, name, intensity in example_proteins:
            protein = ProteinMeasurement(
                protein_id=protein_id,
                gene_id=gene_id,
                protein_name=name,
                abundance=np.log2(intensity + 1),
                raw_intensity=intensity,
                peptide_count=np.random.randint(5, 20),
                unique_peptides=np.random.randint(2, 10),
                sequence_coverage=np.random.uniform(20, 80)
            )
            proteins.append(protein)
        
        return proteins, peptides
    
    def _normalize_abundances(self, 
                            proteins: List[ProteinMeasurement],
                            method: str = "median_centering") -> List[ProteinMeasurement]:
        """Normalize protein abundances"""
        if not proteins:
            return proteins
        
        # Extract abundances
        abundances = np.array([p.abundance for p in proteins])
        
        if method == "median_centering":
            # Median centering normalization
            median_abundance = np.median(abundances)
            normalized = abundances - median_abundance
            
        elif method == "total_intensity":
            # Total intensity normalization
            total = np.sum(abundances)
            normalized = abundances / total * 1e6 if total > 0 else abundances
            
        elif method == "quantile":
            # Quantile normalization
            ranks = stats.rankdata(abundances)
            quantiles = (ranks - 0.5) / len(ranks)
            normalized = stats.norm.ppf(quantiles, loc=np.mean(abundances), scale=np.std(abundances))
            
        else:
            logger.warning(f"Unknown normalization method: {method}")
            normalized = abundances
        
        # Update protein abundances
        for i, protein in enumerate(proteins):
            protein.abundance = normalized[i]
        
        return proteins
    
    def _detect_modifications(self, 
                            proteins: List[ProteinMeasurement],
                            peptides: List[Peptide]) -> List[ProteinMeasurement]:
        """Detect post-translational modifications"""
        # This is a simplified implementation
        # Real implementation would analyze peptide mass shifts
        
        for protein in proteins:
            # Simulate modification detection
            if protein.protein_name and 'kinase' in protein.protein_name.lower():
                # Kinases are often phosphorylated
                protein.modifications.append({
                    'type': 'Phosphorylation',
                    'position': np.random.randint(10, 100),
                    'confidence': 0.95
                })
            
            if protein.protein_name and 'histone' in protein.protein_name.lower():
                # Histones have various modifications
                protein.modifications.extend([
                    {
                        'type': 'Acetylation',
                        'position': np.random.randint(1, 20),
                        'confidence': 0.90
                    },
                    {
                        'type': 'Methylation',
                        'position': np.random.randint(1, 20),
                        'confidence': 0.85
                    }
                ])
        
        return proteins
    
    def _calculate_quality_metrics(self, 
                                 proteins: List[ProteinMeasurement],
                                 peptides: List[Peptide]) -> Dict[str, float]:
        """Calculate proteomics quality metrics"""
        if not proteins:
            return {}
        
        # Protein-level metrics
        abundances = [p.abundance for p in proteins]
        coverages = [p.sequence_coverage for p in proteins]
        peptide_counts = [p.peptide_count for p in proteins]
        
        metrics = {
            'total_proteins': len(proteins),
            'total_peptides': len(peptides),
            'mean_sequence_coverage': np.mean(coverages),
            'median_sequence_coverage': np.median(coverages),
            'mean_peptides_per_protein': np.mean(peptide_counts),
            'proteins_with_single_peptide': sum(1 for p in peptide_counts if p == 1),
            'dynamic_range': np.max(abundances) - np.min(abundances) if abundances else 0,
            'cv_abundance': np.std(abundances) / np.mean(abundances) if abundances and np.mean(abundances) > 0 else 0
        }
        
        # Modification statistics
        total_modifications = sum(len(p.modifications) for p in proteins)
        proteins_with_mods = sum(1 for p in proteins if p.modifications)
        
        metrics['total_modifications'] = total_modifications
        metrics['proteins_with_modifications'] = proteins_with_mods
        metrics['modification_rate'] = proteins_with_mods / len(proteins) * 100 if proteins else 0
        
        return metrics
    
    def merge_replicates(self, 
                        profiles: List[ProteomicsProfile],
                        method: str = "median") -> ProteomicsProfile:
        """Merge technical replicates"""
        if not profiles:
            raise ValueError("No profiles to merge")
        
        if len(profiles) == 1:
            return profiles[0]
        
        # Collect all unique proteins
        all_proteins = {}
        for profile in profiles:
            for protein in profile.proteins:
                if protein.protein_id not in all_proteins:
                    all_proteins[protein.protein_id] = []
                all_proteins[protein.protein_id].append(protein)
        
        # Merge proteins
        merged_proteins = []
        for protein_id, protein_list in all_proteins.items():
            if method == "median":
                merged_abundance = np.median([p.abundance for p in protein_list])
                merged_intensity = np.median([p.raw_intensity for p in protein_list])
            elif method == "mean":
                merged_abundance = np.mean([p.abundance for p in protein_list])
                merged_intensity = np.mean([p.raw_intensity for p in protein_list])
            else:
                raise ValueError(f"Unknown merge method: {method}")
            
            # Use first protein as template
            template = protein_list[0]
            merged_protein = ProteinMeasurement(
                protein_id=protein_id,
                gene_id=template.gene_id,
                protein_name=template.protein_name,
                abundance=merged_abundance,
                raw_intensity=merged_intensity,
                peptide_count=int(np.mean([p.peptide_count for p in protein_list])),
                unique_peptides=int(np.mean([p.unique_peptides for p in protein_list])),
                sequence_coverage=np.mean([p.sequence_coverage for p in protein_list]),
                modifications=template.modifications,  # Keep modifications from first
                confidence=np.mean([p.confidence for p in protein_list])
            )
            merged_proteins.append(merged_protein)
        
        # Create merged profile
        merged_profile = ProteomicsProfile(
            sample_id=f"{profiles[0].sample_id}_merged",
            proteins=merged_proteins,
            peptides=[],  # Don't merge peptides for simplicity
            total_proteins=len(merged_proteins),
            total_peptides=0,
            quantification_method=profiles[0].quantification_method,
            normalization_method=profiles[0].normalization_method,
            quality_metrics=self._calculate_quality_metrics(merged_proteins, []),
            metadata={
                'merged_from': [p.sample_id for p in profiles],
                'merge_method': method
            }
        )
        
        return merged_profile
