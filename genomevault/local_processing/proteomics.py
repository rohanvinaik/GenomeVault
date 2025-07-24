"""
Proteomics Processing Module

Handles mass spectrometry proteomics data including:
- Protein identification and quantification
- Post-translational modifications (PTMs)
- Peptide spectrum matching
- Label-free and labeled quantification
"""

import gzip
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from genomevault.core.config import get_config
from genomevault.core.exceptions import ProcessingError, ValidationError
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)
config = get_config()


class QuantificationMethod(Enum):
    """Proteomics quantification methods"""

    LABEL_FREE = "label_free"
    TMT = "tmt"  # Tandem Mass Tag
    ITRAQ = "itraq"  # Isobaric Tags for Relative and Absolute Quantification
    SILAC = "silac"  # Stable Isotope Labeling by Amino acids in Cell culture
    DIA = "dia"  # Data-Independent Acquisition
    SPECTRAL_COUNTING = "spectral_counting"


class ModificationType(Enum):
    """Common post-translational modifications"""

    PHOSPHORYLATION = "phosphorylation"
    ACETYLATION = "acetylation"
    METHYLATION = "methylation"
    UBIQUITINATION = "ubiquitination"
    GLYCOSYLATION = "glycosylation"
    OXIDATION = "oxidation"
    DEAMIDATION = "deamidation"


@dataclass
class Peptide:
    """Individual peptide identification"""

    sequence: str
    modified_sequence: str
    charge: int
    mass: float
    retention_time: float
    intensity: float
    score: float
    modifications: List[Tuple[int, ModificationType, float]] = field(default_factory=list)
    protein_ids: List[str] = field(default_factory=list)
    is_unique: bool = True
    missed_cleavages: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "sequence": self.sequence,
            "modified_sequence": self.modified_sequence,
            "charge": self.charge,
            "mass": self.mass,
            "retention_time": self.retention_time,
            "intensity": self.intensity,
            "score": self.score,
            "modifications": [(pos, mod.value, mass) for pos, mod, mass in self.modifications],
            "protein_ids": self.protein_ids,
            "is_unique": self.is_unique,
            "missed_cleavages": self.missed_cleavages,
        }


@dataclass
class ProteinMeasurement:
    """Protein quantification data"""

    protein_id: str
    gene_name: str
    description: str
    sequence_coverage: float
    num_peptides: int
    num_unique_peptides: int
    abundance: float
    normalized_abundance: float
    modifications: Dict[ModificationType, int] = field(default_factory=dict)
    peptides: List[Peptide] = field(default_factory=list)
    confidence_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "protein_id": self.protein_id,
            "gene_name": self.gene_name,
            "description": self.description,
            "sequence_coverage": self.sequence_coverage,
            "num_peptides": self.num_peptides,
            "num_unique_peptides": self.num_unique_peptides,
            "abundance": self.abundance,
            "normalized_abundance": self.normalized_abundance,
            "modifications": {mod.value: count for mod, count in self.modifications.items()},
            "num_peptides_detailed": len(self.peptides),
            "confidence_score": self.confidence_score,
        }


@dataclass
class ProteomicsProfile:
    """Complete proteomics profile for a sample"""

    sample_id: str
    proteins: List[ProteinMeasurement]
    quantification_method: QuantificationMethod
    quality_metrics: Dict[str, Any]
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def filter_by_abundance(self, min_abundance: float) -> List[ProteinMeasurement]:
        """Filter proteins by abundance threshold"""
        return [p for p in self.proteins if p.normalized_abundance >= min_abundance]

    def get_protein_by_gene(self, gene_name: str) -> Optional[ProteinMeasurement]:
        """Get protein measurement by gene name"""
        for protein in self.proteins:
            if protein.gene_name == gene_name:
                return protein
        return None

    def get_modified_proteins(self, modification: ModificationType) -> List[ProteinMeasurement]:
        """Get proteins with specific modification"""
        return [
            p
            for p in self.proteins
            if modification in p.modifications and p.modifications[modification] > 0
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        data = [p.to_dict() for p in self.proteins]
        return pd.DataFrame(data)

    def calculate_pathway_enrichment(self, pathway_genes: Set[str]) -> Dict[str, float]:
        """Calculate enrichment for a gene set/pathway"""
        detected_genes = {p.gene_name for p in self.proteins if p.normalized_abundance > 0}
        overlap = detected_genes.intersection(pathway_genes)

        return {
            "overlap_count": len(overlap),
            "pathway_size": len(pathway_genes),
            "detected_in_pathway": len(overlap),
            "enrichment_ratio": (len(overlap) / len(pathway_genes) if pathway_genes else 0),
            "detected_genes": list(overlap),
        }


class ProteomicsProcessor:
    """Process mass spectrometry proteomics data"""

    def __init__(
        self,
        protein_database: Optional[Path] = None,
        modifications_config: Optional[Path] = None,
        min_peptides: int = 2,
        fdr_threshold: float = 0.01,
        max_threads: int = 4,
    ):
        """
        Initialize proteomics processor

        Args:
            protein_database: Path to protein sequence database (FASTA)
            modifications_config: Path to PTM configuration
            min_peptides: Minimum peptides for protein identification
            fdr_threshold: False discovery rate threshold
            max_threads: Maximum threads for processing
        """
        self.protein_database = protein_database
        self.modifications_config = modifications_config
        self.min_peptides = min_peptides
        self.fdr_threshold = fdr_threshold
        self.max_threads = max_threads
        self.protein_sequences = self._load_protein_database()
        self.modification_masses = self._load_modifications()

        logger.info("Initialized ProteomicsProcessor")

    def _load_protein_database(self) -> Dict[str, Dict[str, Any]]:
        """Load protein sequences from FASTA database"""
        if not self.protein_database or not self.protein_database.exists():
            logger.warning("No protein database provided, using mock data")
            # Mock protein database
            return {
                "P53_HUMAN": {
                    "sequence": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPRVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD",
                    "gene_name": "TP53",
                    "description": "Tumor protein p53",
                },
                "BRCA1_HUMAN": {
                    "sequence": "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLCKNDITKRSLQESTRFSQLVEELLKIICAFQLDTGLEYANSYNFAKKENNSPEHLKDEVSIIQSMGYRNRAKRLLQSEPENPSLQETSLSVQLSNLGTVRTLRTKQRIQPQKTSVYIELGSDSSEDTVNKATYCSVGDQELLQITPQGTRDEISLDSAKKAACEFSETDVTNTEHHQPSNNDLNTTEKRAAERHPEKYQGSSVSNLHVEPCGTNTHASSLQHENSSLLLTKDRMNVEKAEFCNKSKQPGLARSQHNRWAGSKETCNDRRTPSTEKKVDLNADPLCERKEWNKQKLPCSENPRDTEDVPWITLNSSIQKVNEWFSRSDELLGSDDSHDGESESNAKVADVLDVLNEVDEYSGSSEKIDLLASDPHEALICKSERVHSKSVESNIEDKIFGKTYRKKASLPNLSHVTENLIIGAFVTEPQIIQERPLTNKLKRKRRPTSGLHPEDFIKKADLAVQKTPEMINQGTNQTEQNGQVMNITNSGHENKTKGDSIQNEKNPNPIESLEKESAFKTKAEPISSSISNMELELNIHNSKAPKKNRLRRKSSTRHIHALELVVSRNLSPPNCTELQIDSCSSSEEIKKKKYNQMPVRHSRNLQLMEGKEPATGAKKSNKPNEQTSKRHDSDTFPELKLTNAPGSFTKCSNTSELKEFVNPSLPREEKEEKLETVKVSNNAEDPKDLMLSGERVLQTERSVESSSISLVPGTDYGTQESISLLEVSTLGKAKTEPNKCVSQCAAFENPKGLIHGCSKDNRNDTEGFKYPLGHEVNHSRETSIEMEESELDAQYLQNTFKVSKRQSFAPFSNPGNAEEECATFSAHSGSLKKQSPKVTFECEQKEENQGKNESNIKPVQTVNITAGFPVVGQKDKPVDNAKCSIKGGSRFCLSSQFRGNETGLITPNKHGLLQNPYRIPPLFPIKSFVKTKCKKNLLEENFEEHSMSPEREMGNENIPSTVSTISRNNIRENVFKEASSSNINEVGSSTNEVGSSINEIGSSDENIQAELGRNRGPKLNAMLRLGVLQPEVYKQSLPGSNCKHPEIKKQEYEEVVQTVNTDFSPYLISDNLEQPMGSSHASQVCSETPDDLLDDGEIKEDTSFAENDIKESSAVFSKSVQKGELSRSPSPFTHTHLAQGYRRGAKKLESSEENLSSEDEELPCFQHLLFGKVNNIPSQSTRHSTVATECLSKNTEENLLSLKNSLNDCSNQVILAKASQEHHLSEETKCSASLFSSQCSELEDLTANTNTQDPFLIGSSKQMRHQSESQGVGLSDKELVSDDEERGTGLEENNQEEQSMDSNLGEAASGCESETSVSEDCSGLSSQSDILTTQQRDTMQHNLIKLQQEMAELEAVLEQHGSQPSNSYPSIISDSSALEDLRNPEQSTSEKAVLTSQKSSEYPISQNPEGLSADKFEVSADSSTSKNKEPGVERSSPSKCPSLDDRWYMHSCSGSLQNRNYPSQEELIKVVDVEEQQLEESGPHDLTETSYLPRQDLEGTPYLESGISLFSDDPESDPSEDRAPESARVGNIPSSTSALKVPQLKVAESAQSPAAAHTTDTAGYNAMEESVSREKPELTASTERVNKRMSMVVSGLTPEEFMLVYKFARKHHITLTNLITEETTHVVMKTDAEFVCERTLKYFLGIAGGKWVVSYFWVTQSIKERKMLNEHDFEVRGDVVNGRNHQGPKRARESQDRKIFRGLEICCYGPFTNMPTDQLEWMVQLCGASVVKELSSFTLGTGVHPIVVVQPDAWTEDNGFHAIGQMCEAPVVTREWVLDSVALYQCQELDTYLIPQIPHSHY",
                    "gene_name": "BRCA1",
                    "description": "Breast cancer type 1 susceptibility protein",
                },
                "EGFR_HUMAN": {
                    "sequence": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
                    "gene_name": "EGFR",
                    "description": "Epidermal growth factor receptor",
                },
            }

        # In production, would parse FASTA file
        proteins = {}
        # ... FASTA parsing code ...
        return proteins

    def _load_modifications(self) -> Dict[ModificationType, float]:
        """Load modification mass shifts"""
        # Standard modification masses
        return {
            ModificationType.PHOSPHORYLATION: 79.966331,
            ModificationType.ACETYLATION: 42.010565,
            ModificationType.METHYLATION: 14.015650,
            ModificationType.UBIQUITINATION: 114.042927,
            ModificationType.OXIDATION: 15.994915,
            ModificationType.DEAMIDATION: 0.984016,
        }

    def process(
        self,
        input_path: Path,
        sample_id: str,
        input_format: str = "maxquant",
        quantification_method: QuantificationMethod = QuantificationMethod.LABEL_FREE,
    ) -> ProteomicsProfile:
        """
        Process proteomics data

        Args:
            input_path: Path to proteomics data
            sample_id: Sample identifier
            input_format: Input format ('maxquant', 'mzml', 'mgf')
            quantification_method: Quantification method used

        Returns:
            ProteomicsProfile with protein measurements
        """
        logger.info("Processing proteomics data for {sample_id}")

        try:
            # Load data based on format
            if input_format == "maxquant":
                protein_data, peptide_data = self._load_maxquant_output(input_path)
            elif input_format == "mzml":
                protein_data, peptide_data = self._process_mzml(input_path)
            else:
                raise ValidationError("Unsupported format: {input_format}")

            # Create protein measurements
            proteins = self._create_protein_measurements(protein_data, peptide_data)

            # Apply FDR filtering
            filtered_proteins = self._apply_fdr_filter(proteins)

            # Normalize abundances
            normalized_proteins = self._normalize_abundances(
                filtered_proteins, quantification_method
            )

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(normalized_proteins, peptide_data)

            # Create profile
            profile = ProteomicsProfile(
                sample_id=sample_id,
                proteins=normalized_proteins,
                quantification_method=quantification_method,
                quality_metrics=quality_metrics,
                processing_metadata={
                    "processor_version": "1.0.0",
                    "processed_at": datetime.now().isoformat(),
                    "input_format": input_format,
                    "fdr_threshold": self.fdr_threshold,
                    "min_peptides": self.min_peptides,
                },
            )

            logger.info("Successfully processed {len(normalized_proteins)} proteins")
            return profile

        except Exception as e:
            logger.error("Error processing proteomics data: {str(e)}")
            raise ProcessingError("Failed to process proteomics data: {str(e)}")

    def _load_maxquant_output(self, input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load MaxQuant output files"""
        logger.info("Loading MaxQuant output from {input_path}")

        # In production, would load proteinGroups.txt and peptides.txt
        # Generate mock data for demonstration
        np.random.seed(42)

        # Mock protein data
        n_proteins = 2000
        protein_ids = list(self.protein_sequences.keys()) + [
            "PROT{i:04d}_HUMAN" for i in range(n_proteins - len(self.protein_sequences))
        ]

        protein_data = pd.DataFrame(
            {
                "Protein IDs": protein_ids[:n_proteins],
                "Gene names": [
                    self.protein_sequences.get(pid, {}).get("gene_name", "GENE{i}")
                    for i, pid in enumerate(protein_ids[:n_proteins])
                ],
                "Protein names": [
                    self.protein_sequences.get(pid, {}).get("description", "Protein {i}")
                    for i, pid in enumerate(protein_ids[:n_proteins])
                ],
                "Peptides": np.random.poisson(10, n_proteins),
                "Unique peptides": np.random.poisson(8, n_proteins),
                "Sequence coverage [%]": np.random.uniform(5, 80, n_proteins),
                "Intensity": np.random.lognormal(20, 2, n_proteins),
                "Score": np.random.uniform(50, 500, n_proteins),
                "PEP": 10 ** -np.random.uniform(2, 10, n_proteins),  # Posterior error probability
            }
        )

        # Mock peptide data
        n_peptides = 20000
        peptide_data = pd.DataFrame(
            {
                "Sequence": [self._generate_random_peptide() for _ in range(n_peptides)],
                "Proteins": np.random.choice(protein_ids[:n_proteins], n_peptides),
                "Charge": np.random.choice([2, 3, 4], n_peptides, p=[0.6, 0.3, 0.1]),
                "Mass": np.random.normal(1500, 500, n_peptides),
                "Retention time": np.random.uniform(0, 120, n_peptides),
                "Intensity": np.random.lognormal(18, 2, n_peptides),
                "Score": np.random.uniform(20, 200, n_peptides),
                "Modifications": [self._generate_random_modifications() for _ in range(n_peptides)],
            }
        )

        return protein_data, peptide_data

    def _process_mzml(self, input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process mzML mass spectrometry data"""
        logger.info("Processing mzML file {input_path}")
        # In production, would use pyteomics or similar to parse mzML
        # For now, return mock data
        return self._load_maxquant_output(input_path)

    def _generate_random_peptide(self, length: int = None) -> str:
        """Generate random peptide sequence"""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        if length is None:
            length = np.random.randint(7, 30)
        return "".join(np.random.choice(list(amino_acids), length))

    def _generate_random_modifications(self) -> str:
        """Generate random modification string"""
        if np.random.random() < 0.7:
            return ""  # No modifications

        mods = []
        if np.random.random() < 0.5:
            mods.append("Phospho (STY)")
        if np.random.random() < 0.3:
            mods.append("Acetyl (K)")
        if np.random.random() < 0.2:
            mods.append("Oxidation (M)")

        return "; ".join(mods)

    def _parse_modifications(self, mod_string: str) -> List[Tuple[int, ModificationType, float]]:
        """Parse modification string into structured format"""
        if not mod_string:
            return []

        modifications = []
        # Simple parsing - in production would be more sophisticated
        if "Phospho" in mod_string:
            modifications.append((1, ModificationType.PHOSPHORYLATION, 79.966331))
        if "Acetyl" in mod_string:
            modifications.append((1, ModificationType.ACETYLATION, 42.010565))
        if "Oxidation" in mod_string:
            modifications.append((1, ModificationType.OXIDATION, 15.994915))

        return modifications

    def _create_protein_measurements(
        self, protein_data: pd.DataFrame, peptide_data: pd.DataFrame
    ) -> List[ProteinMeasurement]:
        """Create protein measurements from data"""
        proteins = []

        for _, prot_row in protein_data.iterrows():
            # Get peptides for this protein
            protein_peptides = peptide_data[peptide_data["Proteins"] == prot_row["Protein IDs"]]

            # Parse peptides
            peptides = []
            mod_counts = defaultdict(int)

            for _, pep_row in protein_peptides.iterrows():
                modifications = self._parse_modifications(pep_row.get("Modifications", ""))

                # Count modifications
                for _, mod_type, _ in modifications:
                    mod_counts[mod_type] += 1

                peptide = Peptide(
                    sequence=pep_row["Sequence"],
                    modified_sequence=pep_row["Sequence"],  # Simplified
                    charge=int(pep_row["Charge"]),
                    mass=float(pep_row["Mass"]),
                    retention_time=float(pep_row["Retention time"]),
                    intensity=float(pep_row["Intensity"]),
                    score=float(pep_row["Score"]),
                    modifications=modifications,
                    protein_ids=[prot_row["Protein IDs"]],
                    is_unique=True,  # Simplified
                )
                peptides.append(peptide)

            # Create protein measurement
            protein = ProteinMeasurement(
                protein_id=prot_row["Protein IDs"],
                gene_name=prot_row.get("Gene names", ""),
                description=prot_row.get("Protein names", ""),
                sequence_coverage=float(prot_row.get("Sequence coverage [%]", 0)),
                num_peptides=int(prot_row.get("Peptides", 0)),
                num_unique_peptides=int(prot_row.get("Unique peptides", 0)),
                abundance=float(prot_row.get("Intensity", 0)),
                normalized_abundance=float(prot_row.get("Intensity", 0)),  # Will normalize later
                modifications=dict(mod_counts),
                peptides=peptides[:10],  # Limit peptides for memory
                confidence_score=1.0 - float(prot_row.get("PEP", 0)),
            )

            proteins.append(protein)

        return proteins

    def _apply_fdr_filter(self, proteins: List[ProteinMeasurement]) -> List[ProteinMeasurement]:
        """Apply FDR filtering to proteins"""
        # Filter by minimum peptides
        filtered = [p for p in proteins if p.num_peptides >= self.min_peptides]

        # Sort by confidence and apply FDR
        filtered.sort(key=lambda p: p.confidence_score, reverse=True)

        # Simple FDR calculation - in production would use more sophisticated methods
        n_targets = len(filtered)
        n_decoys = int(n_targets * self.fdr_threshold)

        if n_decoys > 0:
            threshold_score = filtered[n_targets - n_decoys].confidence_score
            filtered = [p for p in filtered if p.confidence_score >= threshold_score]

        logger.info("Filtered to {len(filtered)} proteins at {self.fdr_threshold} FDR")
        return filtered

    def _normalize_abundances(
        self, proteins: List[ProteinMeasurement], method: QuantificationMethod
    ) -> List[ProteinMeasurement]:
        """Normalize protein abundances"""
        if not proteins:
            return proteins

        abundances = np.array([p.abundance for p in proteins])

        if method == QuantificationMethod.LABEL_FREE:
            # Log transform and median normalization
            log_abundances = np.log2(abundances + 1)
            median_shift = np.median(log_abundances)
            normalized = 2 ** (log_abundances - median_shift + 20)  # Scale to reasonable range

        elif method == QuantificationMethod.TMT:
            # TMT-specific normalization
            # Simplified - in production would handle reporter ion intensities
            total = np.sum(abundances)
            normalized = (abundances / total) * 1e9

        else:
            # Default normalization
            total = np.sum(abundances)
            normalized = (abundances / total) * 1e9

        # Update normalized abundances
        for i, protein in enumerate(proteins):
            protein.normalized_abundance = float(normalized[i])

        return proteins

    def _calculate_quality_metrics(
        self, proteins: List[ProteinMeasurement], peptide_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate quality control metrics"""
        if not proteins:
            return {}

        abundances = [p.normalized_abundance for p in proteins]
        coverages = [p.sequence_coverage for p in proteins]

        metrics = {
            "total_proteins": len(proteins),
            "total_peptides": len(peptide_data),
            "median_sequence_coverage": float(np.median(coverages)),
            "mean_sequence_coverage": float(np.mean(coverages)),
            "proteins_above_50_coverage": sum(1 for c in coverages if c > 50),
            "median_abundance": float(np.median(abundances)),
            "dynamic_range": (
                float(np.log10(max(abundances) / min(abundances))) if min(abundances) > 0 else 0
            ),
            "proteins_with_modifications": sum(1 for p in proteins if p.modifications),
            "phosphoproteins": sum(
                1 for p in proteins if ModificationType.PHOSPHORYLATION in p.modifications
            ),
            "mass_accuracy": 5.0,  # Mock value in ppm
            "retention_time_precision": 0.2,  # Mock value in minutes
        }

        # Modification statistics
        all_mods = defaultdict(int)
        for protein in proteins:
            for mod_type, count in protein.modifications.items():
                all_mods[mod_type.value] += count

        metrics["modifications_summary"] = dict(all_mods)

        return metrics

    def differential_expression(
        self,
        group1_profiles: List[ProteomicsProfile],
        group2_profiles: List[ProteomicsProfile],
        min_fold_change: float = 2.0,
        fdr_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Perform differential protein expression analysis

        Args:
            group1_profiles: Control group profiles
            group2_profiles: Treatment group profiles
            min_fold_change: Minimum fold change threshold
            fdr_threshold: FDR threshold for significance

        Returns:
            DataFrame with differential expression results
        """
        logger.info("Performing differential protein expression analysis")

        # Collect all proteins
        all_proteins = set()
        for profile in group1_profiles + group2_profiles:
            all_proteins.update([p.protein_id for p in profile.proteins])

        results = []

        for protein_id in all_proteins:
            # Get abundances for each group
            group1_abundances = []
            group2_abundances = []

            for profile in group1_profiles:
                protein = next((p for p in profile.proteins if p.protein_id == protein_id), None)
                if protein:
                    group1_abundances.append(protein.normalized_abundance)

            for profile in group2_profiles:
                protein = next((p for p in profile.proteins if p.protein_id == protein_id), None)
                if protein:
                    group2_abundances.append(protein.normalized_abundance)

            if len(group1_abundances) >= 2 and len(group2_abundances) >= 2:
                # Log transform for statistics
                log_group1 = np.log2(np.array(group1_abundances) + 1)
                log_group2 = np.log2(np.array(group2_abundances) + 1)

                # Perform t-test
                from scipy import stats

                t_stat, p_value = stats.ttest_ind(log_group1, log_group2)

                # Calculate fold change
                mean1 = np.mean(group1_abundances)
                mean2 = np.mean(group2_abundances)
                fold_change = mean2 / mean1 if mean1 > 0 else 0
                log2_fc = np.log2(fold_change) if fold_change > 0 else 0

                # Get gene name
                example_protein = next(
                    (
                        p
                        for profile in group1_profiles + group2_profiles
                        for p in profile.proteins
                        if p.protein_id == protein_id
                    ),
                    None,
                )
                gene_name = example_protein.gene_name if example_protein else ""

                results.append(
                    {
                        "protein_id": protein_id,
                        "gene_name": gene_name,
                        "mean_abundance_group1": mean1,
                        "mean_abundance_group2": mean2,
                        "fold_change": fold_change,
                        "log2_fold_change": log2_fc,
                        "p_value": p_value,
                        "t_statistic": t_stat,
                    }
                )

        if not results:
            logger.warning("No proteins found for differential expression")
            return pd.DataFrame()

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Multiple testing correction
        from statsmodels.stats.multitest import multipletests

        _, fdr_values, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
        results_df["fdr"] = fdr_values

        # Mark significant proteins
        results_df["significant"] = (results_df["fdr"] < fdr_threshold) & (
            np.abs(results_df["log2_fold_change"]) >= np.log2(min_fold_change)
        )

        # Sort by p-value
        results_df.sort_values("p_value", inplace=True)

        logger.info("Found {results_df['significant'].sum()} differentially expressed proteins")

        return results_df

    def export_results(
        self, profile: ProteomicsProfile, output_path: Path, format: str = "tsv"
    ) -> None:
        """Export proteomics results to file"""
        logger.info("Exporting proteomics results to {output_path}")

        df = profile.to_dataframe()

        if format == "tsv":
            df.to_csv(output_path, sep="\t", index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        else:
            raise ValidationError("Unsupported export format: {format}")

        logger.info("Successfully exported {len(df)} proteins")
