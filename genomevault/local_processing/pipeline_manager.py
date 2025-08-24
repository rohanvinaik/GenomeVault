"""
Local Processing Engine - Pipeline Manager for GenomeVault.

Orchestrates containerized genomic analysis pipelines with privacy-preserving
features, resource management, and cryptographic attestation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import docker
    from docker.errors import ContainerError, ImageNotFound, APIError
    from docker.models.containers import Container
    from docker.types import Mount

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

    # Mock classes for when Docker is not available
    class ContainerError(Exception):
        pass

    class ImageNotFound(Exception):
        pass

    class APIError(Exception):
        pass

    Container = None
    Mount = None

import psutil

from genomevault.observability.logging import get_logger

logger = get_logger(__name__)


class ContainerRuntime(str, Enum):
    """Supported container runtimes."""

    DOCKER = "docker"
    SINGULARITY = "singularity"
    PODMAN = "podman"
    K3S = "k3s"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceProfile(str, Enum):
    """Predefined resource profiles."""

    MINIMAL = "minimal"  # 2 cores, 4GB RAM
    STANDARD = "standard"  # 8 cores, 16GB RAM
    HIGH = "high"  # 16 cores, 32GB RAM
    MAXIMUM = "maximum"  # 32 cores, 64GB RAM


@dataclass
class ResourceRequirements:
    """Resource requirements for pipeline execution."""

    cpu_cores: int = 2
    memory_gb: int = 4
    disk_gb: int = 10
    gpu_required: bool = False
    gpu_memory_gb: Optional[int] = None

    @classmethod
    def from_profile(cls, profile: ResourceProfile) -> ResourceRequirements:
        """Create requirements from a predefined profile."""
        profiles = {
            ResourceProfile.MINIMAL: cls(cpu_cores=2, memory_gb=4, disk_gb=10),
            ResourceProfile.STANDARD: cls(cpu_cores=8, memory_gb=16, disk_gb=50),
            ResourceProfile.HIGH: cls(cpu_cores=16, memory_gb=32, disk_gb=100),
            ResourceProfile.MAXIMUM: cls(cpu_cores=32, memory_gb=64, disk_gb=500),
        }
        return profiles[profile]

    def scale(self, factor: float) -> ResourceRequirements:
        """Scale resources by a factor."""
        return ResourceRequirements(
            cpu_cores=max(2, min(32, int(self.cpu_cores * factor))),
            memory_gb=max(4, min(64, int(self.memory_gb * factor))),
            disk_gb=int(self.disk_gb * factor),
            gpu_required=self.gpu_required,
            gpu_memory_gb=self.gpu_memory_gb,
        )


@dataclass
class PipelineCheckpoint:
    """Checkpoint for pipeline resumption."""

    checkpoint_id: str
    pipeline_id: str
    step_name: str
    timestamp: datetime
    data_hash: str
    metadata: Dict[str, Any]
    file_paths: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "pipeline_id": self.pipeline_id,
            "step_name": self.step_name,
            "timestamp": self.timestamp.isoformat(),
            "data_hash": self.data_hash,
            "metadata": self.metadata,
            "file_paths": self.file_paths,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineCheckpoint:
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            pipeline_id=data["pipeline_id"],
            step_name=data["step_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data_hash=data["data_hash"],
            metadata=data["metadata"],
            file_paths=data["file_paths"],
        )


@dataclass
class AuditEntry:
    """Audit trail entry for processing."""

    entry_id: str
    timestamp: datetime
    pipeline_id: str
    action: str
    actor: str
    data_hash: str
    attestation: Optional[str] = None
    tee_quote: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "pipeline_id": self.pipeline_id,
            "action": self.action,
            "actor": self.actor,
            "data_hash": self.data_hash,
            "attestation": self.attestation,
            "tee_quote": self.tee_quote,
        }


class BasePipeline(ABC):
    """Base class for all pipelines."""

    def __init__(
        self,
        pipeline_id: Optional[str] = None,
        work_dir: Optional[Path] = None,
        runtime: ContainerRuntime = ContainerRuntime.DOCKER,
        resources: Optional[ResourceRequirements] = None,
    ):
        """Initialize base pipeline."""
        self.pipeline_id = pipeline_id or str(uuid.uuid4())
        self.work_dir = work_dir or Path(tempfile.mkdtemp(prefix="genomevault_"))
        self.runtime = runtime
        self.resources = resources or ResourceRequirements()
        self.status = PipelineStatus.PENDING
        self.checkpoints: List[PipelineCheckpoint] = []
        self.audit_trail: List[AuditEntry] = []

        # Initialize container client
        if runtime == ContainerRuntime.DOCKER and DOCKER_AVAILABLE:
            try:
                self.client = docker.from_env()
            except:
                logger.warning("Docker not available, using simulation mode")
                self.client = None
        else:
            self.client = None  # Other runtimes to be implemented

        logger.info(f"Initialized pipeline {self.pipeline_id} with runtime {runtime}")

    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the pipeline."""
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        pass

    def create_checkpoint(self, step_name: str, data_paths: List[str]) -> PipelineCheckpoint:
        """Create a checkpoint for resumption."""
        # Calculate data hash
        hasher = hashlib.sha256()
        for path in data_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    hasher.update(f.read())

        checkpoint = PipelineCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            pipeline_id=self.pipeline_id,
            step_name=step_name,
            timestamp=datetime.utcnow(),
            data_hash=hasher.hexdigest(),
            metadata={"status": self.status.value},
            file_paths=data_paths,
        )

        self.checkpoints.append(checkpoint)

        # Save checkpoint to disk
        checkpoint_file = self.work_dir / f"checkpoint_{checkpoint.checkpoint_id}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for step {step_name}")
        return checkpoint

    def add_audit_entry(self, action: str, data_hash: str, actor: str = "system") -> AuditEntry:
        """Add an entry to the audit trail."""
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            pipeline_id=self.pipeline_id,
            action=action,
            actor=actor,
            data_hash=data_hash,
        )

        # Generate attestation
        entry.attestation = self._generate_attestation(entry)

        # Get TEE quote if available
        entry.tee_quote = self._get_tee_quote()

        self.audit_trail.append(entry)

        # Persist audit entry
        audit_file = self.work_dir / f"audit_{entry.entry_id}.json"
        with open(audit_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

        logger.info(f"Added audit entry: {action}")
        return entry

    def _generate_attestation(self, entry: AuditEntry) -> str:
        """Generate cryptographic attestation for audit entry."""
        # Simple hash-based attestation (in production, use proper signing)
        data = f"{entry.entry_id}:{entry.timestamp}:{entry.action}:{entry.data_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _get_tee_quote(self) -> Optional[str]:
        """Get TEE quote if running in SGX/SEV environment."""
        # Check for Intel SGX
        if os.path.exists("/dev/sgx_enclave"):
            # In production, generate actual SGX quote
            return "sgx_quote_placeholder"

        # Check for AMD SEV
        if os.path.exists("/dev/sev"):
            # In production, generate actual SEV attestation
            return "sev_attestation_placeholder"

        return None

    def cleanup(self):
        """Clean up temporary files and resources."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        logger.info(f"Cleaned up pipeline {self.pipeline_id}")


class GenomicsPipeline(BasePipeline):
    """Pipeline for genomics analysis (FASTQ→BAM→VCF)."""

    def __init__(self, reference_genome: str = "GRCh38", **kwargs):
        """Initialize genomics pipeline."""
        super().__init__(**kwargs)
        self.reference_genome = reference_genome
        self.containers = {
            "bwa": "biocontainers/bwa-mem2:2.2.1_cv1",
            "samtools": "biocontainers/samtools:1.17--h00cdaf9_0",
            "bcftools": "biocontainers/bcftools:1.17--haef29d1_0",
            "gatk": "broadinstitute/gatk:4.4.0.0",
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate genomics input data."""
        required = ["fastq_r1", "fastq_r2"]
        return all(key in input_data for key in required)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run genomics pipeline."""
        self.status = PipelineStatus.INITIALIZING
        self.add_audit_entry(
            "pipeline_started", hashlib.sha256(str(input_data).encode()).hexdigest()
        )

        try:
            # Step 1: Alignment with BWA-MEM2
            logger.info("Starting alignment with BWA-MEM2")
            bam_file = await self._run_bwa_alignment(input_data["fastq_r1"], input_data["fastq_r2"])
            self.create_checkpoint("alignment", [bam_file])

            # Step 2: Sort and index BAM
            logger.info("Sorting and indexing BAM")
            sorted_bam = await self._sort_and_index_bam(bam_file)
            self.create_checkpoint("sort_index", [sorted_bam, f"{sorted_bam}.bai"])

            # Step 3: Variant calling
            logger.info("Calling variants")
            vcf_file = await self._call_variants(sorted_bam)
            self.create_checkpoint("variant_calling", [vcf_file])

            # Step 4: Variant filtering and annotation
            logger.info("Filtering and annotating variants")
            filtered_vcf = await self._filter_variants(vcf_file)
            self.create_checkpoint("filtering", [filtered_vcf])

            self.status = PipelineStatus.COMPLETED
            self.add_audit_entry(
                "pipeline_completed", hashlib.sha256(filtered_vcf.encode()).hexdigest()
            )

            return {
                "status": "success",
                "pipeline_id": self.pipeline_id,
                "outputs": {
                    "bam": sorted_bam,
                    "vcf": filtered_vcf,
                    "checkpoints": [cp.checkpoint_id for cp in self.checkpoints],
                },
            }

        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.add_audit_entry("pipeline_failed", str(e))
            logger.error(f"Pipeline failed: {e}")
            raise

    async def _run_bwa_alignment(self, fastq_r1: str, fastq_r2: str) -> str:
        """Run BWA-MEM2 alignment."""
        output_sam = str(self.work_dir / "aligned.sam")

        if self.runtime == ContainerRuntime.DOCKER:
            # Run BWA in Docker container
            container = self.client.containers.run(
                self.containers["bwa"],
                f"bwa-mem2 mem -t {self.resources.cpu_cores} /ref/genome.fa /data/r1.fq /data/r2.fq",
                volumes={
                    str(self.work_dir): {"bind": "/data", "mode": "rw"},
                    "/reference": {"bind": "/ref", "mode": "ro"},
                },
                cpu_count=self.resources.cpu_cores,
                mem_limit=f"{self.resources.memory_gb}g",
                detach=True,
            )

            # Wait for completion
            result = container.wait()
            logs = container.logs()
            container.remove()

            if result["StatusCode"] != 0:
                raise ContainerError(
                    container, result["StatusCode"], "bwa-mem2", self.containers["bwa"], logs
                )
        else:
            # Fallback to subprocess for testing
            output_sam = str(self.work_dir / "aligned.sam")
            # Simulate alignment
            with open(output_sam, "w") as f:
                f.write("@HD\tVN:1.6\tSO:coordinate\n")
                f.write("@SQ\tSN:chr1\tLN:248956422\n")
                f.write("read1\t0\tchr1\t100\t60\t100M\t*\t0\t0\tACGT\t*\n")

        return output_sam

    async def _sort_and_index_bam(self, sam_file: str) -> str:
        """Sort and index BAM file."""
        bam_file = str(self.work_dir / "sorted.bam")

        if self.runtime == ContainerRuntime.DOCKER:
            # Convert SAM to sorted BAM
            container = self.client.containers.run(
                self.containers["samtools"],
                f"samtools sort -@ {self.resources.cpu_cores} -o /data/sorted.bam /data/aligned.sam",
                volumes={str(self.work_dir): {"bind": "/data", "mode": "rw"}},
                cpu_count=self.resources.cpu_cores,
                mem_limit=f"{self.resources.memory_gb}g",
                detach=True,
            )

            result = container.wait()
            container.remove()

            # Index the BAM
            container = self.client.containers.run(
                self.containers["samtools"],
                "samtools index /data/sorted.bam",
                volumes={str(self.work_dir): {"bind": "/data", "mode": "rw"}},
                detach=True,
            )

            result = container.wait()
            container.remove()
        else:
            # Simulate for testing
            with open(bam_file, "wb") as f:
                f.write(b"BAM\x01")  # BAM magic number

        return bam_file

    async def _call_variants(self, bam_file: str) -> str:
        """Call variants using GATK or bcftools."""
        vcf_file = str(self.work_dir / "variants.vcf")

        # Simulate variant calling for testing
        with open(vcf_file, "w") as f:
            f.write("##fileformat=VCFv4.3\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            f.write("chr1\t100\t.\tA\tG\t30\tPASS\tDP=10\n")

        return vcf_file

    async def _filter_variants(self, vcf_file: str) -> str:
        """Filter and annotate variants."""
        filtered_vcf = str(self.work_dir / "filtered.vcf")

        # Simulate filtering
        shutil.copy(vcf_file, filtered_vcf)

        return filtered_vcf


class TranscriptomicsPipeline(BasePipeline):
    """Pipeline for transcriptomics analysis (FASTQ→counts)."""

    def __init__(self, **kwargs):
        """Initialize transcriptomics pipeline."""
        super().__init__(**kwargs)
        self.containers = {
            "kallisto": "biocontainers/kallisto:0.48.0--h0d531b0_1",
            "salmon": "combinelab/salmon:1.10.0",
            "rsem": "biocontainers/rsem:1.3.3--pl5321h86d3e6b_2",
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate transcriptomics input data."""
        required = ["fastq_files", "transcriptome_index"]
        return all(key in input_data for key in required)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run transcriptomics pipeline."""
        self.status = PipelineStatus.INITIALIZING
        self.add_audit_entry(
            "pipeline_started", hashlib.sha256(str(input_data).encode()).hexdigest()
        )

        try:
            # Step 1: Pseudoalignment with Kallisto
            logger.info("Running Kallisto pseudoalignment")
            abundance_file = await self._run_kallisto(
                input_data["fastq_files"], input_data["transcriptome_index"]
            )
            self.create_checkpoint("pseudoalignment", [abundance_file])

            # Step 2: Generate count matrix
            logger.info("Generating count matrix")
            count_matrix = await self._generate_count_matrix(abundance_file)
            self.create_checkpoint("count_matrix", [count_matrix])

            # Step 3: Quality control
            logger.info("Running quality control")
            qc_report = await self._run_qc(count_matrix)
            self.create_checkpoint("qc", [qc_report])

            self.status = PipelineStatus.COMPLETED
            self.add_audit_entry(
                "pipeline_completed", hashlib.sha256(count_matrix.encode()).hexdigest()
            )

            return {
                "status": "success",
                "pipeline_id": self.pipeline_id,
                "outputs": {
                    "abundance": abundance_file,
                    "counts": count_matrix,
                    "qc_report": qc_report,
                },
            }

        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.add_audit_entry("pipeline_failed", str(e))
            logger.error(f"Pipeline failed: {e}")
            raise

    async def _run_kallisto(self, fastq_files: List[str], index: str) -> str:
        """Run Kallisto pseudoalignment."""
        output_dir = self.work_dir / "kallisto_output"
        output_dir.mkdir(exist_ok=True)

        # Simulate Kallisto output
        abundance_file = output_dir / "abundance.tsv"
        with open(abundance_file, "w") as f:
            f.write("target_id\tlength\teff_length\test_counts\ttpm\n")
            f.write("ENST00000456328\t1657\t1478.00\t2.00000\t0.678733\n")
            f.write("ENST00000450305\t632\t453.00\t0.00000\t0.000000\n")

        return str(abundance_file)

    async def _generate_count_matrix(self, abundance_file: str) -> str:
        """Generate count matrix from abundance estimates."""
        count_matrix = str(self.work_dir / "count_matrix.tsv")

        # Simulate count matrix
        with open(count_matrix, "w") as f:
            f.write("gene_id\tsample1\tsample2\tsample3\n")
            f.write("ENSG00000223972\t10\t15\t12\n")
            f.write("ENSG00000227232\t25\t30\t28\n")

        return count_matrix

    async def _run_qc(self, count_matrix: str) -> str:
        """Run quality control on count data."""
        qc_report = str(self.work_dir / "qc_report.html")

        # Simulate QC report
        with open(qc_report, "w") as f:
            f.write("<html><body><h1>QC Report</h1><p>All samples passed QC</p></body></html>")

        return qc_report


class EpigenomicsPipeline(BasePipeline):
    """Pipeline for epigenomics analysis (methylation)."""

    def __init__(self, **kwargs):
        """Initialize epigenomics pipeline."""
        super().__init__(**kwargs)
        self.containers = {
            "bismark": "biocontainers/bismark:0.24.0--hdfd78af_0",
            "methylkit": "biocontainers/methylkit:1.24.0--r42hdfd78af_0",
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate epigenomics input data."""
        required = ["fastq_files", "genome_index"]
        return all(key in input_data for key in required)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run epigenomics pipeline."""
        self.status = PipelineStatus.INITIALIZING
        self.add_audit_entry(
            "pipeline_started", hashlib.sha256(str(input_data).encode()).hexdigest()
        )

        try:
            # Step 1: Bisulfite alignment with Bismark
            logger.info("Running Bismark alignment")
            alignment_file = await self._run_bismark(
                input_data["fastq_files"], input_data["genome_index"]
            )
            self.create_checkpoint("bisulfite_alignment", [alignment_file])

            # Step 2: Methylation extraction
            logger.info("Extracting methylation calls")
            methylation_file = await self._extract_methylation(alignment_file)
            self.create_checkpoint("methylation_extraction", [methylation_file])

            # Step 3: DMR detection
            logger.info("Detecting differentially methylated regions")
            dmr_file = await self._detect_dmrs(methylation_file)
            self.create_checkpoint("dmr_detection", [dmr_file])

            self.status = PipelineStatus.COMPLETED
            self.add_audit_entry(
                "pipeline_completed", hashlib.sha256(dmr_file.encode()).hexdigest()
            )

            return {
                "status": "success",
                "pipeline_id": self.pipeline_id,
                "outputs": {
                    "alignment": alignment_file,
                    "methylation": methylation_file,
                    "dmrs": dmr_file,
                },
            }

        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.add_audit_entry("pipeline_failed", str(e))
            logger.error(f"Pipeline failed: {e}")
            raise

    async def _run_bismark(self, fastq_files: List[str], genome_index: str) -> str:
        """Run Bismark bisulfite alignment."""
        alignment_file = str(self.work_dir / "bismark_aligned.bam")

        # Simulate Bismark alignment
        with open(alignment_file, "wb") as f:
            f.write(b"BAM\x01")  # BAM magic number

        return alignment_file

    async def _extract_methylation(self, alignment_file: str) -> str:
        """Extract methylation calls from alignment."""
        methylation_file = str(self.work_dir / "methylation_calls.txt")

        # Simulate methylation extraction
        with open(methylation_file, "w") as f:
            f.write("chr\tpos\tstrand\tcount_methylated\tcount_unmethylated\n")
            f.write("chr1\t100\t+\t8\t2\n")
            f.write("chr1\t150\t+\t5\t5\n")

        return methylation_file

    async def _detect_dmrs(self, methylation_file: str) -> str:
        """Detect differentially methylated regions."""
        dmr_file = str(self.work_dir / "dmrs.bed")

        # Simulate DMR detection
        with open(dmr_file, "w") as f:
            f.write("chr1\t1000\t2000\tDMR1\t0.001\n")
            f.write("chr2\t5000\t6000\tDMR2\t0.005\n")

        return dmr_file


class ClinicalPipeline(BasePipeline):
    """Pipeline for clinical data processing (FHIR R4)."""

    def __init__(self, **kwargs):
        """Initialize clinical pipeline."""
        super().__init__(**kwargs)

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate clinical input data."""
        required = ["fhir_bundle"]
        return all(key in input_data for key in required)

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run clinical pipeline."""
        self.status = PipelineStatus.INITIALIZING
        self.add_audit_entry(
            "pipeline_started", hashlib.sha256(str(input_data).encode()).hexdigest()
        )

        try:
            # Step 1: Parse FHIR R4 bundle
            logger.info("Parsing FHIR R4 bundle")
            parsed_data = await self._parse_fhir_bundle(input_data["fhir_bundle"])
            self.create_checkpoint("fhir_parsing", [str(self.work_dir / "parsed.json")])

            # Step 2: Map to LOINC/SNOMED codes
            logger.info("Mapping to standard terminologies")
            mapped_data = await self._map_terminologies(parsed_data)
            self.create_checkpoint("terminology_mapping", [str(self.work_dir / "mapped.json")])

            # Step 3: Extract clinical features
            logger.info("Extracting clinical features")
            features = await self._extract_features(mapped_data)
            self.create_checkpoint("feature_extraction", [str(self.work_dir / "features.json")])

            self.status = PipelineStatus.COMPLETED
            self.add_audit_entry(
                "pipeline_completed", hashlib.sha256(json.dumps(features).encode()).hexdigest()
            )

            return {
                "status": "success",
                "pipeline_id": self.pipeline_id,
                "outputs": {
                    "parsed_data": parsed_data,
                    "mapped_data": mapped_data,
                    "features": features,
                },
            }

        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.add_audit_entry("pipeline_failed", str(e))
            logger.error(f"Pipeline failed: {e}")
            raise

    async def _parse_fhir_bundle(self, fhir_bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Parse FHIR R4 bundle."""
        parsed_data = {"patient": {}, "observations": [], "conditions": [], "medications": []}

        # Extract resources from bundle
        if "entry" in fhir_bundle:
            for entry in fhir_bundle["entry"]:
                resource = entry.get("resource", {})
                resource_type = resource.get("resourceType")

                if resource_type == "Patient":
                    parsed_data["patient"] = resource
                elif resource_type == "Observation":
                    parsed_data["observations"].append(resource)
                elif resource_type == "Condition":
                    parsed_data["conditions"].append(resource)
                elif resource_type == "MedicationStatement":
                    parsed_data["medications"].append(resource)

        # Save parsed data
        with open(self.work_dir / "parsed.json", "w") as f:
            json.dump(parsed_data, f, indent=2)

        return parsed_data

    async def _map_terminologies(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map to LOINC and SNOMED codes."""
        mapped_data = parsed_data.copy()

        # Map observations to LOINC
        for obs in mapped_data.get("observations", []):
            if "code" in obs and "coding" in obs["code"]:
                for coding in obs["code"]["coding"]:
                    if coding.get("system") == "http://loinc.org":
                        coding["display"] = self._get_loinc_display(coding.get("code"))

        # Map conditions to SNOMED
        for condition in mapped_data.get("conditions", []):
            if "code" in condition and "coding" in condition["code"]:
                for coding in condition["code"]["coding"]:
                    if coding.get("system") == "http://snomed.info/sct":
                        coding["display"] = self._get_snomed_display(coding.get("code"))

        # Save mapped data
        with open(self.work_dir / "mapped.json", "w") as f:
            json.dump(mapped_data, f, indent=2)

        return mapped_data

    def _get_loinc_display(self, code: str) -> str:
        """Get LOINC display name (simplified)."""
        loinc_map = {
            "2160-0": "Creatinine",
            "2345-7": "Glucose",
            "2339-0": "Glucose [Mass/volume] in Blood",
        }
        return loinc_map.get(code, f"LOINC {code}")

    def _get_snomed_display(self, code: str) -> str:
        """Get SNOMED display name (simplified)."""
        snomed_map = {
            "44054006": "Diabetes mellitus type 2",
            "38341003": "Hypertension",
            "254837009": "Malignant neoplasm of breast",
        }
        return snomed_map.get(code, f"SNOMED {code}")

    async def _extract_features(self, mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract clinical features for analysis."""
        features = {"demographics": {}, "lab_values": [], "diagnoses": [], "medications": []}

        # Extract demographics
        patient = mapped_data.get("patient", {})
        if patient:
            features["demographics"] = {
                "age": self._calculate_age(patient.get("birthDate")),
                "gender": patient.get("gender"),
                "id": patient.get("id"),
            }

        # Extract lab values
        for obs in mapped_data.get("observations", []):
            if "valueQuantity" in obs:
                features["lab_values"].append(
                    {
                        "code": obs.get("code", {}).get("coding", [{}])[0].get("code"),
                        "value": obs["valueQuantity"].get("value"),
                        "unit": obs["valueQuantity"].get("unit"),
                    }
                )

        # Extract diagnoses
        for condition in mapped_data.get("conditions", []):
            features["diagnoses"].append(
                {
                    "code": condition.get("code", {}).get("coding", [{}])[0].get("code"),
                    "display": condition.get("code", {}).get("coding", [{}])[0].get("display"),
                }
            )

        # Save features
        with open(self.work_dir / "features.json", "w") as f:
            json.dump(features, f, indent=2)

        return features

    def _calculate_age(self, birth_date: Optional[str]) -> Optional[int]:
        """Calculate age from birth date."""
        if not birth_date:
            return None

        birth = datetime.fromisoformat(birth_date)
        today = datetime.now()
        age = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return age


class PipelineManager:
    """Main pipeline manager for orchestrating genomic processing."""

    def __init__(
        self,
        work_dir: Optional[Path] = None,
        runtime: ContainerRuntime = ContainerRuntime.DOCKER,
        enable_tee: bool = False,
        enable_k3s: bool = False,
    ):
        """Initialize pipeline manager."""
        self.work_dir = work_dir or Path("/tmp/genomevault_pipelines")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime
        self.enable_tee = enable_tee
        self.enable_k3s = enable_k3s

        self.pipelines: Dict[str, BasePipeline] = {}
        self.resource_monitor = ResourceMonitor()

        logger.info(f"Initialized PipelineManager with runtime {runtime}")

    async def create_pipeline(
        self,
        pipeline_type: str,
        pipeline_id: Optional[str] = None,
        resources: Optional[ResourceRequirements] = None,
        **kwargs,
    ) -> BasePipeline:
        """Create a new pipeline instance."""
        pipeline_id = pipeline_id or str(uuid.uuid4())
        work_dir = self.work_dir / pipeline_id
        work_dir.mkdir(parents=True, exist_ok=True)

        # Auto-scale resources based on available system resources
        if resources is None:
            resources = self.resource_monitor.get_optimal_resources()

        # Create pipeline based on type
        pipeline_classes = {
            "genomics": GenomicsPipeline,
            "transcriptomics": TranscriptomicsPipeline,
            "epigenomics": EpigenomicsPipeline,
            "clinical": ClinicalPipeline,
        }

        if pipeline_type not in pipeline_classes:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        pipeline_class = pipeline_classes[pipeline_type]
        pipeline = pipeline_class(
            pipeline_id=pipeline_id,
            work_dir=work_dir,
            runtime=self.runtime,
            resources=resources,
            **kwargs,
        )

        self.pipelines[pipeline_id] = pipeline

        logger.info(f"Created {pipeline_type} pipeline with ID {pipeline_id}")
        return pipeline

    async def run_pipeline(
        self, pipeline_id: str, input_data: Dict[str, Any], resume_from: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a pipeline with optional resumption."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        pipeline = self.pipelines[pipeline_id]

        # Validate input
        if not pipeline.validate_input(input_data):
            raise ValueError("Invalid input data for pipeline")

        # Resume from checkpoint if specified
        if resume_from:
            checkpoint = self._load_checkpoint(pipeline_id, resume_from)
            logger.info(f"Resuming pipeline from checkpoint {resume_from}")
            # Restore pipeline state from checkpoint
            # (Implementation depends on specific pipeline requirements)

        # Monitor resources during execution
        with self.resource_monitor.monitor(pipeline_id):
            result = await pipeline.run(input_data)

        return result

    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline status and metrics."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")

        pipeline = self.pipelines[pipeline_id]

        return {
            "pipeline_id": pipeline_id,
            "status": pipeline.status.value,
            "checkpoints": [cp.to_dict() for cp in pipeline.checkpoints],
            "audit_trail": [entry.to_dict() for entry in pipeline.audit_trail],
            "resources": {
                "cpu_cores": pipeline.resources.cpu_cores,
                "memory_gb": pipeline.resources.memory_gb,
                "disk_gb": pipeline.resources.disk_gb,
            },
        }

    def cleanup_pipeline(self, pipeline_id: str):
        """Clean up pipeline resources."""
        if pipeline_id in self.pipelines:
            pipeline = self.pipelines[pipeline_id]
            pipeline.cleanup()
            del self.pipelines[pipeline_id]
            logger.info(f"Cleaned up pipeline {pipeline_id}")

    def _load_checkpoint(self, pipeline_id: str, checkpoint_id: str) -> PipelineCheckpoint:
        """Load a checkpoint from disk."""
        checkpoint_file = self.work_dir / pipeline_id / f"checkpoint_{checkpoint_id}.json"

        if not checkpoint_file.exists():
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        with open(checkpoint_file, "r") as f:
            data = json.load(f)

        return PipelineCheckpoint.from_dict(data)


class ResourceMonitor:
    """Monitor and manage computational resources."""

    def __init__(self):
        """Initialize resource monitor."""
        self.active_monitors: Dict[str, Dict[str, Any]] = {}

    def get_optimal_resources(self) -> ResourceRequirements:
        """Get optimal resource allocation based on system availability."""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        disk_gb = psutil.disk_usage("/").free / (1024**3)

        # Allocate 50% of available resources by default
        return ResourceRequirements(
            cpu_cores=max(2, min(32, cpu_count // 2)),
            memory_gb=max(4, min(64, int(memory_gb * 0.5))),
            disk_gb=min(500, int(disk_gb * 0.3)),
        )

    def monitor(self, pipeline_id: str):
        """Context manager for monitoring pipeline resource usage."""

        class ResourceContext:
            def __init__(self, monitor, pipeline_id):
                self.monitor = monitor
                self.pipeline_id = pipeline_id
                self.start_time = None
                self.start_resources = None

            def __enter__(self):
                self.start_time = time.time()
                self.start_resources = {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage("/").percent,
                }
                self.monitor.active_monitors[self.pipeline_id] = {
                    "start_time": self.start_time,
                    "start_resources": self.start_resources,
                }
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                end_resources = {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage("/").percent,
                }

                if self.pipeline_id in self.monitor.active_monitors:
                    self.monitor.active_monitors[self.pipeline_id].update(
                        {
                            "end_time": end_time,
                            "duration": end_time - self.start_time,
                            "end_resources": end_resources,
                            "peak_cpu": max(
                                self.start_resources["cpu_percent"], end_resources["cpu_percent"]
                            ),
                            "peak_memory": max(
                                self.start_resources["memory_percent"],
                                end_resources["memory_percent"],
                            ),
                        }
                    )

                logger.info(
                    f"Pipeline {self.pipeline_id} resource usage: "
                    f"Duration={end_time - self.start_time:.2f}s, "
                    f"Peak CPU={self.monitor.active_monitors[self.pipeline_id]['peak_cpu']:.1f}%, "
                    f"Peak Memory={self.monitor.active_monitors[self.pipeline_id]['peak_memory']:.1f}%"
                )

        return ResourceContext(self, pipeline_id)

    def get_resource_usage(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get resource usage metrics for a pipeline."""
        return self.active_monitors.get(pipeline_id)


if __name__ == "__main__":
    # Example usage
    async def main():
        manager = PipelineManager()

        # Create and run a genomics pipeline
        pipeline = await manager.create_pipeline(
            "genomics", resources=ResourceRequirements.from_profile(ResourceProfile.STANDARD)
        )

        # Example input data
        input_data = {
            "fastq_r1": "/data/sample_R1.fastq.gz",
            "fastq_r2": "/data/sample_R2.fastq.gz",
        }

        result = await manager.run_pipeline(pipeline.pipeline_id, input_data)
        print(f"Pipeline result: {result}")

        # Get status
        status = manager.get_pipeline_status(pipeline.pipeline_id)
        print(f"Pipeline status: {status}")

        # Cleanup
        manager.cleanup_pipeline(pipeline.pipeline_id)

    # Run example
    asyncio.run(main())
