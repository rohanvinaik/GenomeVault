"""
genomevault/api/routers/clinical_query.py

REST API endpoints for clinical variant queries
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import logging

from genomevault.clinical_db.database import ClinicalSNPDatabase, ClinicalSNP

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clinical-db", tags=["clinical-database"])

# Global database instance (loaded at startup)
_clinical_db: Optional[ClinicalSNPDatabase] = None


def get_clinical_database() -> ClinicalSNPDatabase:
    """Get clinical database singleton"""
    global _clinical_db
    if _clinical_db is None:
        raise HTTPException(
            status_code=500,
            detail="Clinical database not loaded. Ensure database file exists at startup."
        )
    return _clinical_db


def init_clinical_database(db_path: str):
    """Initialize clinical database (call at app startup)"""
    global _clinical_db
    try:
        _clinical_db = ClinicalSNPDatabase(db_path)
        logger.info(f"Clinical database loaded: {_clinical_db.get_statistics()}")
    except Exception as e:
        logger.error(f"Failed to load clinical database: {e}")
        _clinical_db = None


# Request/Response Models

class ClinicalQueryRequest(BaseModel):
    """Request to query clinical significance of variants"""
    chromosome: str = Field(..., description="Chromosome (e.g., 'chr11' or '11')")
    positions: List[int] = Field(..., description="List of genomic positions")
    ref_alleles: Optional[List[str]] = Field(None, description="Reference alleles (optional)")
    alt_alleles: Optional[List[str]] = Field(None, description="Alternate alleles (optional)")


class ClinicalSNPResponse(BaseModel):
    """Clinical SNP with all annotations"""
    snp_id: str
    chromosome: str
    position: int
    ref_allele: str
    alt_alleles: List[str]
    gene: Optional[str]
    clinical_significance: str
    conditions: List[Dict]
    clinical_annotations: Optional[Dict]
    sources: Dict[str, str]


class ClinicalQueryResponse(BaseModel):
    """Response with clinical annotations"""
    query: ClinicalQueryRequest
    results: List[ClinicalSNPResponse]
    summary: Dict[str, int]


class DatabaseStatsResponse(BaseModel):
    """Database statistics"""
    total_snps: int
    pathogenic_count: int
    pharmaco_count: int
    genes_covered: int
    conditions_covered: int
    genome_build: str
    version: str
    build_date: str


# Endpoints

@router.get("/status", response_model=Dict)
async def get_status():
    """Check if clinical database is loaded"""
    return {
        "database_loaded": _clinical_db is not None,
        "status": "ok" if _clinical_db else "not_loaded"
    }


@router.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_stats():
    """Get clinical database statistics"""
    db = get_clinical_database()
    return db.get_statistics()


@router.post("/query/positions", response_model=ClinicalQueryResponse)
async def query_positions(request: ClinicalQueryRequest):
    """
    Query clinical significance of specific genomic positions
    
    Example:
    ```json
    {
      "chromosome": "chr11",
      "positions": [5227002],
      "ref_alleles": ["A"],
      "alt_alleles": ["T"]
    }
    ```
    """
    db = get_clinical_database()
    results = []
    
    for i, pos in enumerate(request.positions):
        snps = db.query_position(request.chromosome, pos)
        
        # Filter by alleles if provided
        if request.ref_alleles and request.alt_alleles:
            if i < len(request.ref_alleles) and i < len(request.alt_alleles):
                ref = request.ref_alleles[i]
                alt = request.alt_alleles[i]
                snps = [
                    s for s in snps 
                    if s.ref_allele == ref and alt in s.alt_alleles
                ]
        
        results.extend(snps)
    
    # Convert to response models
    snp_responses = [
        ClinicalSNPResponse(**snp.to_dict())
        for snp in results
    ]
    
    # Calculate summary
    summary = calculate_summary(results)
    
    return ClinicalQueryResponse(
        query=request,
        results=snp_responses,
        summary=summary
    )


@router.get("/query/gene/{gene_symbol}")
async def query_gene(gene_symbol: str):
    """
    Get all clinical variants in a gene
    
    Example: /api/v1/clinical-db/query/gene/BRCA1
    """
    db = get_clinical_database()
    snps = db.query_gene(gene_symbol)
    
    if not snps:
        raise HTTPException(
            status_code=404,
            detail=f"No clinical variants found for gene {gene_symbol}"
        )
    
    return {
        "gene_symbol": gene_symbol,
        "total_variants": len(snps),
        "variants": [s.to_dict() for s in snps]
    }


@router.get("/query/rsid/{rs_id}")
async def query_rsid(rs_id: str):
    """
    Query specific SNP by dbSNP ID
    
    Example: /api/v1/clinical-db/query/rsid/rs334
    """
    db = get_clinical_database()
    snp = db.query_rsid(rs_id)
    
    if not snp:
        raise HTTPException(
            status_code=404,
            detail=f"SNP {rs_id} not found in database"
        )
    
    return ClinicalSNPResponse(**snp.to_dict())


@router.get("/pathogenic")
async def get_pathogenic_variants(
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return")
):
    """Get all pathogenic/likely pathogenic variants"""
    db = get_clinical_database()
    snps = db.get_pathogenic_variants()[:limit]
    
    return {
        "total_pathogenic": len(db.get_pathogenic_variants()),
        "returned": len(snps),
        "variants": [s.to_dict() for s in snps]
    }


# Utility functions

def calculate_summary(snps: List[ClinicalSNP]) -> Dict[str, int]:
    """Calculate summary statistics for SNP list"""
    summary = {
        'total': len(snps),
        'pathogenic': 0,
        'likely_pathogenic': 0,
        'benign': 0,
        'likely_benign': 0,
        'uncertain_significance': 0,
        'other': 0
    }
    
    for snp in snps:
        sig = snp.clinical_significance.lower()
        
        if sig == 'pathogenic':
            summary['pathogenic'] += 1
        elif sig == 'likely_pathogenic':
            summary['likely_pathogenic'] += 1
        elif sig == 'benign':
            summary['benign'] += 1
        elif sig == 'likely_benign':
            summary['likely_benign'] += 1
        elif 'uncertain' in sig or 'vus' in sig:
            summary['uncertain_significance'] += 1
        else:
            summary['other'] += 1
    
    return summary
