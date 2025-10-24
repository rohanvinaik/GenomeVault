"""
genomevault/clinical_db/database.py

Core Clinical SNP Database Implementation
"""

import json
import gzip
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ClinicalCondition:
    """Clinical condition associated with a variant"""
    name: str
    omim_id: Optional[str] = None
    inheritance: Optional[str] = None
    penetrance: Optional[str] = None


@dataclass
class ClinicalAnnotation:
    """Clinical annotation metadata"""
    review_status: str
    stars: int
    assertion_criteria: Optional[str] = None
    evidence_level: Optional[str] = None


@dataclass
class PopulationFrequency:
    """Population allele frequencies"""
    gnomad_global: Optional[float] = None
    gnomad_afr: Optional[float] = None
    gnomad_eur: Optional[float] = None
    gnomad_eas: Optional[float] = None
    gnomad_amr: Optional[float] = None
    gnomad_sas: Optional[float] = None


@dataclass
class FunctionalImpact:
    """Functional impact predictions"""
    consequence: str
    protein_change: Optional[str] = None
    transcript_id: Optional[str] = None
    sift_score: Optional[float] = None
    polyphen_score: Optional[float] = None


@dataclass
class ClinicalSNP:
    """Complete clinical SNP record"""
    snp_id: str
    chromosome: str
    position: int
    ref_allele: str
    alt_alleles: List[str]
    gene: Optional[str] = None
    clinical_significance: str = "uncertain_significance"
    conditions: List[ClinicalCondition] = None
    clinical_annotations: Optional[ClinicalAnnotation] = None
    population_frequencies: Optional[PopulationFrequency] = None
    functional_impact: Optional[FunctionalImpact] = None
    sources: Dict[str, str] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.sources is None:
            self.sources = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ClinicalSNP':
        """Create from dictionary"""
        # Convert nested dicts to dataclasses
        if 'conditions' in data and data['conditions']:
            data['conditions'] = [
                ClinicalCondition(**c) if isinstance(c, dict) else c 
                for c in data['conditions']
            ]
        if 'clinical_annotations' in data and data['clinical_annotations']:
            data['clinical_annotations'] = ClinicalAnnotation(**data['clinical_annotations'])
        if 'population_frequencies' in data and data['population_frequencies']:
            data['population_frequencies'] = PopulationFrequency(**data['population_frequencies'])
        if 'functional_impact' in data and data['functional_impact']:
            data['functional_impact'] = FunctionalImpact(**data['functional_impact'])
        
        return cls(**data)
    
    def is_pathogenic(self) -> bool:
        """Check if variant is pathogenic"""
        return self.clinical_significance.lower() in [
            'pathogenic', 'likely_pathogenic'
        ]
    
    def is_pharmacogenomic(self) -> bool:
        """Check if variant has pharmacogenomic relevance"""
        return 'pharmgkb_id' in self.sources


class ClinicalSNPDatabase:
    """
    Fast in-memory clinical SNP database with multi-level indexing
    
    Usage:
        db = ClinicalSNPDatabase('data/clinical_snps_v1.0.0.json.gz')
        snps = db.query_position('chr11', 5227002)
        brca_variants = db.query_gene('BRCA1')
    """
    
    def __init__(self, db_path: str, preload: bool = True):
        """
        Initialize database
        
        Args:
            db_path: Path to clinical SNP database (JSON or JSON.gz)
            preload: Load and index database immediately
        """
        self.db_path = Path(db_path)
        self.data = None
        self.metadata = None
        self.indices = None
        
        if preload:
            self.load()
    
    def load(self):
        """Load database and build indices"""
        logger.info(f"Loading clinical database from {self.db_path}")
        
        # Load database
        self.data = self._load_database()
        self.metadata = self.data.get('metadata', {})
        
        # Build indices
        logger.info("Building indices...")
        self.indices = self._build_indices()
        
        logger.info(f"Database loaded: {self.metadata.get('total_snps', 0)} SNPs")
    
    def _load_database(self) -> Dict:
        """Load JSON database from disk"""
        if self.db_path.suffix == '.gz':
            with gzip.open(self.db_path, 'rt') as f:
                return json.load(f)
        else:
            with open(self.db_path, 'r') as f:
                return json.load(f)
    
    def _build_indices(self) -> Dict:
        """Build fast lookup indices"""
        indices = {
            'position': defaultdict(list),  # (chr, pos) -> [SNPs]
            'gene': defaultdict(list),      # gene -> [SNPs]
            'condition': defaultdict(list), # condition -> [SNPs]
            'rsid': {},                     # rsid -> SNP
            'pathogenic': [],               # All pathogenic variants
            'pharmaco': []                  # All pharmacogenomic variants
        }
        
        # Iterate through all SNPs
        snps_data = self.data.get('snps', {})
        for chromosome, positions in snps_data.items():
            for position, snp_list in positions.items():
                position = int(position)
                
                for snp_dict in snp_list:
                    snp = ClinicalSNP.from_dict(snp_dict)
                    
                    # Position index
                    indices['position'][(chromosome, position)].append(snp)
                    
                    # Gene index
                    if snp.gene:
                        indices['gene'][snp.gene].append(snp)
                    
                    # Condition index
                    for condition in snp.conditions:
                        condition_key = condition.name.lower()
                        indices['condition'][condition_key].append(snp)
                    
                    # rsID index
                    indices['rsid'][snp.snp_id] = snp
                    
                    # Pathogenic index
                    if snp.is_pathogenic():
                        indices['pathogenic'].append(snp)
                    
                    # Pharmacogenomic index
                    if snp.is_pharmacogenomic():
                        indices['pharmaco'].append(snp)
        
        return indices
    
    def query_position(self, chromosome: str, position: int) -> List[ClinicalSNP]:
        """
        Query SNPs at specific genomic position
        
        Args:
            chromosome: Chromosome (e.g., 'chr11' or '11')
            position: Genomic position (1-based)
            
        Returns:
            List of ClinicalSNP objects at this position
        """
        # Normalize chromosome format
        if not chromosome.startswith('chr'):
            chromosome = f'chr{chromosome}'
        
        return self.indices['position'].get((chromosome, position), [])
    
    def query_region(self, chromosome: str, start: int, end: int) -> List[ClinicalSNP]:
        """
        Query SNPs in genomic region
        
        Args:
            chromosome: Chromosome
            start: Start position (inclusive)
            end: End position (inclusive)
            
        Returns:
            List of ClinicalSNP objects in region
        """
        results = []
        for pos in range(start, end + 1):
            results.extend(self.query_position(chromosome, pos))
        return results
    
    def query_gene(self, gene_symbol: str) -> List[ClinicalSNP]:
        """
        Query all clinical SNPs in a gene
        
        Args:
            gene_symbol: Gene symbol (e.g., 'BRCA1')
            
        Returns:
            List of ClinicalSNP objects in gene
        """
        return self.indices['gene'].get(gene_symbol, [])
    
    def query_condition(self, condition: str) -> List[ClinicalSNP]:
        """
        Query SNPs associated with a clinical condition
        
        Args:
            condition: Condition name (case-insensitive)
            
        Returns:
            List of ClinicalSNP objects associated with condition
        """
        return self.indices['condition'].get(condition.lower(), [])
    
    def query_rsid(self, rs_id: str) -> Optional[ClinicalSNP]:
        """
        Query specific SNP by dbSNP ID
        
        Args:
            rs_id: dbSNP ID (e.g., 'rs334')
            
        Returns:
            ClinicalSNP object if found, None otherwise
        """
        return self.indices['rsid'].get(rs_id)
    
    def get_pathogenic_variants(self) -> List[ClinicalSNP]:
        """Get all pathogenic/likely pathogenic variants"""
        return self.indices['pathogenic']
    
    def get_pharmacogenomic_variants(self) -> List[ClinicalSNP]:
        """Get all pharmacogenomic variants"""
        return self.indices['pharmaco']
    
    def analyze_vcf_file(self, vcf_path: str) -> Dict:
        """
        Analyze VCF file for clinical variants
        
        Args:
            vcf_path: Path to VCF file
            
        Returns:
            Dictionary with analysis results
        """
        # For now, return a placeholder
        # TODO: Implement VCF parsing
        return {
            'total_variants': 0,
            'clinical_hits': 0,
            'pathogenic_count': 0,
            'pharmaco_count': 0,
            'detailed_results': []
        }
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            'total_snps': self.metadata.get('total_snps', 0),
            'pathogenic_count': len(self.indices['pathogenic']),
            'pharmaco_count': len(self.indices['pharmaco']),
            'genes_covered': len(self.indices['gene']),
            'conditions_covered': len(self.indices['condition']),
            'genome_build': self.metadata.get('genome_build', 'Unknown'),
            'version': self.metadata.get('version', 'Unknown'),
            'build_date': self.metadata.get('build_date', 'Unknown')
        }


class ClinicalDatabaseBuilder:
    """
    Build clinical SNP database from source data
    
    Usage:
        builder = ClinicalDatabaseBuilder()
        builder.add_clinvar_variants('clinvar.vcf')
        builder.save('clinical_snps_v1.0.0.json')
    """
    
    def __init__(self, genome_build: str = "GRCh38"):
        self.genome_build = genome_build
        self.snps = defaultdict(lambda: defaultdict(list))  # chr -> pos -> [SNPs]
        self.genes = defaultdict(dict)
        self.conditions = defaultdict(dict)
        
    def add_snp(self, snp: ClinicalSNP):
        """Add SNP to database"""
        chr_key = snp.chromosome
        pos_key = str(snp.position)
        
        self.snps[chr_key][pos_key].append(snp.to_dict())
        
        # Update gene index
        if snp.gene:
            if snp.gene not in self.genes:
                self.genes[snp.gene] = {
                    'chromosome': snp.chromosome,
                    'snp_ids': []
                }
            self.genes[snp.gene]['snp_ids'].append(snp.snp_id)
        
        # Update condition index
        for condition in snp.conditions:
            condition_key = condition.name.lower()
            if condition_key not in self.conditions:
                self.conditions[condition_key] = {
                    'name': condition.name,
                    'omim_id': condition.omim_id,
                    'associated_snps': []
                }
            self.conditions[condition_key]['associated_snps'].append(snp.snp_id)
    
    def save(self, output_path: str, compress: bool = True):
        """Save database to file"""
        output_path = Path(output_path)
        
        # Count total SNPs
        total_snps = sum(
            len(snp_list)
            for positions in self.snps.values()
            for snp_list in positions.values()
        )
        
        # Build metadata
        metadata = {
            'version': '1.0.0',
            'build_date': '2025-10-24',
            'genome_build': self.genome_build,
            'total_snps': total_snps,
            'sources': ['ClinVar', 'PharmGKB', 'GWAS Catalog']
        }
        
        # Build database structure
        database = {
            'metadata': metadata,
            'snps': dict(self.snps),
            'genes': dict(self.genes),
            'conditions': dict(self.conditions)
        }
        
        # Save to file
        if compress or output_path.suffix == '.gz':
            if not output_path.suffix == '.gz':
                output_path = output_path.with_suffix(output_path.suffix + '.gz')
            with gzip.open(output_path, 'wt') as f:
                json.dump(database, f, indent=2)
        else:
            with open(output_path, 'w') as f:
                json.dump(database, f, indent=2)
        
        logger.info(f"Database saved to {output_path}")
        logger.info(f"Total SNPs: {metadata['total_snps']}")
