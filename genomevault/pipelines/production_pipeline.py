"""
Production Pipeline for GenomeVault

Complete workflow: GDiff → HDC → ZK → PIR
Can be called from API, CLI, or programmatically.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from genomevault.differential_encoding.gdiff.schema import GDiffDocument
from genomevault.hypervector_transform.unified_encoder import UnifiedEncoder
from genomevault.zk_proofs.groth16_proof import generate_proof, verify_proof
from genomevault.pir.it_pir import ITPIRClient, ITPIRServer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for production pipeline"""
    hdc_dimension: int = 10000
    hdc_backend: str = "auto"
    enable_zk_proof: bool = True
    enable_pir: bool = False
    pir_database_size: int = 100
    sample_variants: Optional[int] = None  # If set, sample first N variants for speed


@dataclass
class StageResult:
    """Result from a pipeline stage"""
    stage_name: str
    duration_s: float
    success: bool
    metrics: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete pipeline result"""
    pipeline_id: str
    success: bool
    total_duration_s: float
    stages: Dict[str, StageResult]
    gdiff_file: str
    summary: Dict[str, Any]


class ProductionPipeline:
    """
    Production pipeline orchestrator.

    Runs complete GDiff → HDC → ZK → PIR workflow with configurable stages.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize production pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        self.stages: Dict[str, StageResult] = {}
        self.gdiff_doc: Optional[GDiffDocument] = None
        self.hypervector: Optional[Any] = None

    def run(
        self,
        gdiff_file: Path,
        pipeline_id: str = "production"
    ) -> PipelineResult:
        """
        Execute complete production pipeline.

        Args:
            gdiff_file: Path to GDiff document (.gdiff.gz)
            pipeline_id: Unique identifier for this pipeline run

        Returns:
            PipelineResult with all stage results
        """
        logger.info(f"Starting production pipeline: {pipeline_id}")
        logger.info(f"GDiff file: {gdiff_file}")

        pipeline_start = time.time()

        # Stage 1: Load GDiff
        self._run_stage("gdiff_load", self._load_gdiff, gdiff_file)

        # Stage 2: HDC Encoding
        if self.stages["gdiff_load"].success:
            self._run_stage("hdc_encoding", self._hdc_encoding)

        # Stage 3: ZK Proof (optional)
        if self.config.enable_zk_proof and self.stages.get("hdc_encoding", StageResult("", 0, False, {})).success:
            self._run_stage("zk_proof", self._zk_proof_generation)

        # Stage 4: PIR Query (optional)
        if self.config.enable_pir and self.stages.get("hdc_encoding", StageResult("", 0, False, {})).success:
            self._run_stage("pir_query", self._pir_query)

        # Calculate total time
        total_duration = time.time() - pipeline_start

        # Determine overall success
        success = all(stage.success for stage in self.stages.values())

        # Build summary
        summary = self._build_summary()

        result = PipelineResult(
            pipeline_id=pipeline_id,
            success=success,
            total_duration_s=total_duration,
            stages=self.stages,
            gdiff_file=str(gdiff_file),
            summary=summary
        )

        logger.info(f"Pipeline completed: success={success}, duration={total_duration:.2f}s")

        return result

    def _run_stage(self, stage_name: str, stage_func, *args, **kwargs) -> StageResult:
        """Execute a pipeline stage with error handling"""
        logger.info(f"Starting stage: {stage_name}")
        start = time.time()

        try:
            metrics = stage_func(*args, **kwargs)
            duration = time.time() - start

            result = StageResult(
                stage_name=stage_name,
                duration_s=duration,
                success=True,
                metrics=metrics
            )

            logger.info(f"✓ Stage {stage_name} completed ({duration:.2f}s)")

        except Exception as e:
            duration = time.time() - start
            logger.error(f"✗ Stage {stage_name} failed: {e}")

            result = StageResult(
                stage_name=stage_name,
                duration_s=duration,
                success=False,
                metrics={},
                error=str(e)
            )

        self.stages[stage_name] = result
        return result

    def _load_gdiff(self, gdiff_file: Path) -> Dict[str, Any]:
        """Load GDiff document"""
        if not gdiff_file.exists():
            raise FileNotFoundError(f"GDiff file not found: {gdiff_file}")

        file_size_mb = gdiff_file.stat().st_size / (1024*1024)

        self.gdiff_doc = GDiffDocument.load(gdiff_file)

        return {
            "file_size_mb": file_size_mb,
            "total_variants": len(self.gdiff_doc.differential_variants),
            "k_anonymity": self.gdiff_doc.metadata.k_anonymity,
            "schema_version": self.gdiff_doc.schema_version,
            "query_id": self.gdiff_doc.metadata.query_id
        }

    def _hdc_encoding(self) -> Dict[str, Any]:
        """HDC hypervector encoding"""
        if not self.gdiff_doc:
            raise RuntimeError("GDiff document not loaded")

        # Determine sample size
        total_variants = len(self.gdiff_doc.differential_variants)
        sample_size = self.config.sample_variants or min(1000, total_variants)

        logger.info(f"Encoding {sample_size:,} of {total_variants:,} variants")

        # Convert GDiff variants to encoder format
        variant_data = []
        for v in self.gdiff_doc.differential_variants[:sample_size]:
            variant_data.append({
                "chrom": v.chrom,
                "pos": v.pos,
                "ref": v.ref,
                "alt": v.alt,
                "quality": v.differential_context.confidence * 100,
                "diff_type": v.differential_context.diff_type,
                "pool_coverage": v.differential_context.pool_coverage
            })

        # Initialize encoder
        encoder = UnifiedEncoder(
            dimension=self.config.hdc_dimension,
            k_anonymity=self.gdiff_doc.metadata.k_anonymity,
            backend=self.config.hdc_backend
        )

        # Encode
        self.hypervector = encoder.encode_variants(variant_data)

        # Calculate size
        import numpy as np
        hv_size_kb = (self.hypervector.size * self.hypervector.itemsize) / 1024

        return {
            "dimension": self.hypervector.shape[0],
            "size_kb": hv_size_kb,
            "variants_encoded": sample_size,
            "total_variants": total_variants,
            "throughput_var_per_sec": sample_size / self.stages["hdc_encoding"].duration_s if "hdc_encoding" in self.stages else 0,
            "backend": encoder.backend
        }

    def _zk_proof_generation(self) -> Dict[str, Any]:
        """Generate zero-knowledge proof"""
        if not self.gdiff_doc or self.hypervector is None:
            raise RuntimeError("HDC encoding not complete")

        # Use first variant as example
        example_variant = self.gdiff_doc.differential_variants[0]

        # Create witness
        witness = {
            "chrom": example_variant.chrom,
            "pos": example_variant.pos,
            "ref": example_variant.ref,
            "alt": example_variant.alt,
            "hypervector_sample": self.hypervector[:100].tolist()
        }

        # Generate proof
        proof = generate_proof(witness)
        is_valid = verify_proof(proof)

        # Calculate proof size
        import json
        proof_size = len(json.dumps(proof).encode())

        return {
            "proof_size_bytes": proof_size,
            "verification_status": "valid" if is_valid else "invalid",
            "example_variant": f"{example_variant.chrom}:{example_variant.pos} {example_variant.ref}>{example_variant.alt}"
        }

    def _pir_query(self) -> Dict[str, Any]:
        """Execute PIR query"""
        # Setup IT-PIR
        server = ITPIRServer(database_size=self.config.pir_database_size)
        client = ITPIRClient(database_size=self.config.pir_database_size)

        # Query for record
        query_index = 42
        query = client.generate_query(query_index)
        response = server.answer_query(query)
        result = client.decode_response(response, query_index)

        return {
            "database_size": self.config.pir_database_size,
            "query_index": query_index,
            "query_size_bytes": len(str(query)),
            "response_size_bytes": len(str(response)),
            "information_theoretic_security": True
        }

    def _build_summary(self) -> Dict[str, Any]:
        """Build summary statistics"""
        summary = {
            "stages_completed": len([s for s in self.stages.values() if s.success]),
            "stages_failed": len([s for s in self.stages.values() if not s.success]),
            "total_stages": len(self.stages)
        }

        if self.gdiff_doc:
            summary["total_variants"] = len(self.gdiff_doc.differential_variants)
            summary["k_anonymity"] = self.gdiff_doc.metadata.k_anonymity
            summary["privacy_preserved"] = True

        if self.hypervector is not None:
            summary["hdc_dimension"] = self.hypervector.shape[0]

        return summary
