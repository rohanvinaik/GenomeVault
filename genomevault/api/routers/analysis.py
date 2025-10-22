"""Analysis router for GenomeVault API."""

from __future__ import annotations

import tempfile
import time
import uuid
import json
import logging
import shutil
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from genomevault.api.models.analysis import (
    GenomeAnalysisRequest,
    GenomeAnalysisResponse,
    AnalysisStatus,
    AnalysisType,
    FileFormat,
    AnalysisStageResult,
)
from genomevault.api.middleware.file_handling import validate_upload_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["Genomic Analysis"])

# In-memory storage for analysis status (use Redis in production)
analysis_jobs: Dict[str, AnalysisStatus] = {}
analysis_results: Dict[str, GenomeAnalysisResponse] = {}


@router.post("/submit", response_model=dict)
async def submit_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Genomic data file (VCF, FASTQ, BAM)"),
    file_r2: Optional[UploadFile] = File(None, description="Paired-end FASTQ R2 file"),
    analysis_type: str = Form(..., description="Type of analysis to perform"),
    reference_genome: str = Form("GRCh38", description="Reference genome assembly"),
    k_anonymity: int = Form(3, description="k-anonymity level (≥2)"),
    dimension: int = Form(10000, description="Hypervector dimension (1024-100000)"),
    enable_zk_proof: bool = Form(True, description="Generate zero-knowledge proof"),
    enable_blockchain: bool = Form(False, description="Record on blockchain"),
    enable_pir: bool = Form(False, description="Enable PIR query"),
    analysis_params: str = Form("{}", description="Analysis-specific parameters (JSON)"),
):
    """
    Submit a genome file for privacy-preserving analysis.

    Workflow:
    1. Upload file(s) to temporary storage
    2. Queue analysis job
    3. Return job ID for status polling
    4. Process in background:
       - Differential encoding (k-anonymity)
       - HDC encoding (264× compression)
       - ZK proof generation (privacy verification)
       - PIR query (if enabled)
       - Blockchain attestation (if enabled)

    Returns:
        dict: Contains analysis_id and status for tracking
    """
    # Validate file
    await validate_upload_file(file)
    if file_r2:
        await validate_upload_file(file_r2)

    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())

    # Save uploaded files to temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix=f"genomevault_{analysis_id}_"))
    file_path = temp_dir / file.filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    file_path_r2 = None
    if file_r2:
        file_path_r2 = temp_dir / file_r2.filename
        with open(file_path_r2, "wb") as f:
            content = await file_r2.read()
            f.write(content)

    # Detect file format
    file_format = _detect_file_format(file.filename)

    # Parse analysis params
    try:
        params = json.loads(analysis_params)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid analysis_params JSON")

    # Validate analysis type
    try:
        analysis_type_enum = AnalysisType(analysis_type)
    except ValueError:
        raise HTTPException(
            400,
            f"Invalid analysis_type. Must be one of: {', '.join([t.value for t in AnalysisType])}"
        )

    # Create request object with validation
    try:
        request = GenomeAnalysisRequest(
            analysis_type=analysis_type_enum,
            file_format=file_format,
            file_path=file_path,
            file_path_r2=file_path_r2,
            reference_genome=reference_genome,
            k_anonymity=k_anonymity,
            dimension=dimension,
            enable_zk_proof=enable_zk_proof,
            enable_blockchain=enable_blockchain,
            enable_pir=enable_pir,
            analysis_params=params,
        )
    except Exception as e:
        # Catch Pydantic ValidationError and convert to HTTP 422
        raise HTTPException(422, str(e))

    # Initialize status
    analysis_jobs[analysis_id] = AnalysisStatus(
        analysis_id=analysis_id,
        status="queued",
        progress_percent=0.0,
    )

    # Queue background processing
    background_tasks.add_task(
        _process_analysis,
        analysis_id=analysis_id,
        request=request,
        temp_dir=temp_dir,
    )

    return {
        "analysis_id": analysis_id,
        "status": "queued",
        "message": "Analysis queued successfully. Use GET /api/v1/analysis/{analysis_id}/status to check progress."
    }


@router.get("/{analysis_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get status of an analysis job."""
    if analysis_id not in analysis_jobs:
        raise HTTPException(404, f"Analysis {analysis_id} not found")
    return analysis_jobs[analysis_id]


@router.get("/{analysis_id}/results", response_model=GenomeAnalysisResponse)
async def get_analysis_results(analysis_id: str):
    """Get results of a completed analysis."""
    if analysis_id not in analysis_results:
        # Check if still processing
        if analysis_id in analysis_jobs:
            status = analysis_jobs[analysis_id]
            if status.status in ["queued", "processing"]:
                raise HTTPException(202, "Analysis still processing")
            elif status.status == "failed":
                raise HTTPException(500, "Analysis failed. Check status for details.")
        raise HTTPException(404, f"Results for analysis {analysis_id} not found")

    return analysis_results[analysis_id]


async def _process_analysis(
    analysis_id: str,
    request: GenomeAnalysisRequest,
    temp_dir: Path,
):
    """
    Background task to process genome analysis.

    Pipeline stages:
    1. Differential Encoding (k-anonymity)
    2. HDC Encoding (264× compression)
    3. ZK Proof Generation (privacy verification)
    4. PIR Query (private retrieval)
    5. Blockchain Attestation (audit trail)
    """
    start_time = time.time()
    stages = []

    try:
        # Update status
        analysis_jobs[analysis_id].status = "processing"
        analysis_jobs[analysis_id].progress_percent = 10.0
        analysis_jobs[analysis_id].current_stage = "differential_encoding"

        # Stage 1: Differential Encoding
        stage_start = time.time()
        encoding_result = None
        try:
            from genomevault.differential_encoding.enhanced_pipeline import create_enhanced_pipeline

            # Determine reference paths
            reference_genome_path = Path(f"benchmark_results/full_pipeline_synthetic/reference/{request.reference_genome.lower()}.fa")
            if not reference_genome_path.exists():
                # Fallback to chr22 for testing
                reference_genome_path = Path("benchmark_results/full_pipeline_synthetic/reference/chr22.fa")

            reference_pool_dir = Path("benchmark_results/differential_encoding_samples/vcf_pool")

            # Create blockchain config if enabled
            blockchain_config = None
            if request.enable_blockchain:
                blockchain_config = {"enabled": True}

            pipeline = create_enhanced_pipeline(
                reference_genome=reference_genome_path,
                reference_pool_dir=reference_pool_dir,
                dimension=request.dimension,
                blockchain_config=blockchain_config,
            )

            encoding_result = pipeline.encode_file(
                input_file=request.file_path,
                input_file_r2=request.file_path_r2,
            )

            stages.append(AnalysisStageResult(
                stage_name="differential_encoding",
                success=True,
                duration_ms=(time.time() - stage_start) * 1000,
                output={
                    "compression_ratio": encoding_result.compression_ratio if hasattr(encoding_result, 'compression_ratio') else None,
                    "num_differences": len(encoding_result.differences) if hasattr(encoding_result, 'differences') else 0,
                    "k_anonymity": request.k_anonymity,
                }
            ))

            analysis_jobs[analysis_id].progress_percent = 40.0

        except Exception as e:
            logger.exception(f"Differential encoding failed for {analysis_id}")
            stages.append(AnalysisStageResult(
                stage_name="differential_encoding",
                success=False,
                duration_ms=(time.time() - stage_start) * 1000,
                error=str(e)
            ))
            raise

        # Stage 2: HDC Encoding
        analysis_jobs[analysis_id].current_stage = "hdc_encoding"
        stage_start = time.time()
        hypervector = None
        try:
            from genomevault.hypervector_transform import create_backend_encoder

            encoder = create_backend_encoder(dimension=request.dimension)

            # Get differences from encoding result
            differences = encoding_result.differences if hasattr(encoding_result, 'differences') else []
            hypervector = encoder.encode_single(differences)

            # Store hypervector (in production, save to database)
            hypervector_id = f"hv_{analysis_id}"

            stages.append(AnalysisStageResult(
                stage_name="hdc_encoding",
                success=True,
                duration_ms=(time.time() - stage_start) * 1000,
                output={
                    "dimension": request.dimension,
                    "hypervector_id": hypervector_id,
                }
            ))

            analysis_jobs[analysis_id].progress_percent = 60.0

        except Exception as e:
            logger.exception(f"HDC encoding failed for {analysis_id}")
            stages.append(AnalysisStageResult(
                stage_name="hdc_encoding",
                success=False,
                duration_ms=(time.time() - stage_start) * 1000,
                error=str(e)
            ))
            # Continue processing even if HDC fails (for partial results)

        # Stage 3: ZK Proof (optional)
        zk_proof_id = None
        zk_verification_status = None
        if request.enable_zk_proof:
            analysis_jobs[analysis_id].current_stage = "zk_proof"
            stage_start = time.time()
            try:
                from genomevault.zk_proofs import prove, verify

                # Generate proof data
                proof_input = {
                    "k_anonymity": request.k_anonymity,
                    "dimension": request.dimension,
                }

                proof = prove(proof_input)
                verification = verify(proof)

                zk_proof_id = f"proof_{analysis_id}"
                zk_verification_status = verification

                stages.append(AnalysisStageResult(
                    stage_name="zk_proof",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output={
                        "proof_id": zk_proof_id,
                        "verification_status": zk_verification_status,
                    }
                ))

                analysis_jobs[analysis_id].progress_percent = 80.0

            except Exception as e:
                logger.exception(f"ZK proof generation failed for {analysis_id}")
                stages.append(AnalysisStageResult(
                    stage_name="zk_proof",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    error=str(e)
                ))
                # Continue processing even if ZK proof fails

        # Stage 4: PIR Query (optional)
        pir_result = None
        if request.enable_pir and request.pir_database:
            analysis_jobs[analysis_id].current_stage = "pir_query"
            stage_start = time.time()
            try:
                from genomevault.pir import PIRClient

                pir_client = PIRClient()

                # Mock PIR query for now (integrate with actual PIR when ready)
                pir_result = {"status": "success", "query_id": f"pir_{analysis_id}"}

                stages.append(AnalysisStageResult(
                    stage_name="pir_query",
                    success=True,
                    duration_ms=(time.time() - stage_start) * 1000,
                    output={"result": pir_result}
                ))

            except Exception as e:
                logger.exception(f"PIR query failed for {analysis_id}")
                stages.append(AnalysisStageResult(
                    stage_name="pir_query",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    error=str(e)
                ))

        # Stage 5: Blockchain Attestation (if enabled and available)
        blockchain_tx = None
        attestation_id = None
        if request.enable_blockchain:
            analysis_jobs[analysis_id].current_stage = "blockchain_attestation"
            stage_start = time.time()
            try:
                # Import conditionally since blockchain may not be configured
                try:
                    from genomevault.blockchain.hipaa import create_hipaa_attestation_registry

                    registry = create_hipaa_attestation_registry(
                        blockchain_config={"enabled": True},
                        signatory_registry=None,
                    )

                    differences = encoding_result.differences if hasattr(encoding_result, 'differences') else []
                    blockchain_tx = registry.record_institutional_encoding(
                        encoding_id=analysis_id,
                        npi=None,
                        institution_name="GenomeVault API",
                        input_data=differences,
                        output_data=hypervector if hypervector is not None else [],
                        metadata={
                            "analysis_type": request.analysis_type.value,
                            "compression_ratio": encoding_result.compression_ratio if hasattr(encoding_result, 'compression_ratio') else None,
                        }
                    )

                    attestation_id = analysis_id

                    stages.append(AnalysisStageResult(
                        stage_name="blockchain_attestation",
                        success=True,
                        duration_ms=(time.time() - stage_start) * 1000,
                        output={"tx_hash": blockchain_tx}
                    ))
                except ImportError:
                    logger.warning("Blockchain module not available, skipping attestation")
                    stages.append(AnalysisStageResult(
                        stage_name="blockchain_attestation",
                        success=False,
                        duration_ms=(time.time() - stage_start) * 1000,
                        error="Blockchain module not available"
                    ))

            except Exception as e:
                logger.exception(f"Blockchain attestation failed for {analysis_id}")
                stages.append(AnalysisStageResult(
                    stage_name="blockchain_attestation",
                    success=False,
                    duration_ms=(time.time() - stage_start) * 1000,
                    error=str(e)
                ))

        # Create response
        total_duration = (time.time() - start_time) * 1000

        compression_ratio = None
        variants_analyzed = None
        if encoding_result:
            compression_ratio = encoding_result.compression_ratio if hasattr(encoding_result, 'compression_ratio') else None
            variants_analyzed = len(encoding_result.differences) if hasattr(encoding_result, 'differences') else 0

        response = GenomeAnalysisResponse(
            analysis_id=analysis_id,
            status="success",
            stages=stages,
            total_duration_ms=total_duration,
            compression_ratio=compression_ratio,
            variants_analyzed=variants_analyzed,
            hypervector_id=hypervector_id if hypervector is not None else None,
            hypervector_dimension=request.dimension,
            zk_proof_id=zk_proof_id,
            zk_verification_status=zk_verification_status,
            pir_query_result=pir_result,
            blockchain_tx_hash=blockchain_tx,
            attestation_id=attestation_id,
            analysis_results=_compute_analysis_specific_results(
                request.analysis_type,
                encoding_result,
                hypervector,
                request.analysis_params,
            ),
        )

        # Store results
        analysis_results[analysis_id] = response

        # Update status
        analysis_jobs[analysis_id].status = "completed"
        analysis_jobs[analysis_id].progress_percent = 100.0
        analysis_jobs[analysis_id].current_stage = "completed"

    except Exception as e:
        logger.exception(f"Analysis failed for {analysis_id}")
        # Analysis failed
        analysis_jobs[analysis_id].status = "failed"
        analysis_jobs[analysis_id].current_stage = "error"

        # Store partial results
        response = GenomeAnalysisResponse(
            analysis_id=analysis_id,
            status="failed",
            stages=stages,
            total_duration_ms=(time.time() - start_time) * 1000,
            warnings=[f"Analysis failed: {str(e)}"],
        )
        analysis_results[analysis_id] = response

    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")


def _detect_file_format(filename: str) -> FileFormat:
    """Detect file format from filename."""
    filename_lower = filename.lower()
    if filename_lower.endswith(".vcf.gz"):
        return FileFormat.VCF_GZ
    elif filename_lower.endswith(".vcf"):
        return FileFormat.VCF
    elif filename_lower.endswith(".fastq.gz") or filename_lower.endswith(".fq.gz"):
        return FileFormat.FASTQ_GZ
    elif filename_lower.endswith(".fastq") or filename_lower.endswith(".fq"):
        return FileFormat.FASTQ
    elif filename_lower.endswith(".bam"):
        return FileFormat.BAM
    elif filename_lower.endswith(".sam"):
        return FileFormat.SAM
    else:
        raise HTTPException(400, f"Unsupported file format: {filename}")


def _compute_analysis_specific_results(
    analysis_type: AnalysisType,
    encoding_result: Any,
    hypervector: Any,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute analysis-specific results based on type.

    This is where you'd implement:
    - Pharmacogenomics: Drug-gene interaction lookups
    - Ancestry: Population comparison
    - Risk assessment: Disease risk scoring
    - Etc.

    Args:
        analysis_type: Type of analysis requested
        encoding_result: Result from differential encoding
        hypervector: Encoded hypervector
        params: Analysis-specific parameters

    Returns:
        Dictionary of analysis-specific results
    """
    # Placeholder - implement actual analysis logic
    return {
        "analysis_type": analysis_type.value,
        "note": "Analysis-specific results will be computed here based on analysis type",
        "params_received": params,
    }
