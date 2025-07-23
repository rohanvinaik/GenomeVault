"""
PIR query builder for genomic data queries.
Provides high-level interface for constructing privacy-preserving queries.
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from genomevault.utils.logging import logger

from ..client import PIRClient, PIRQuery
from ..reference_data.manager import GenomicRegion, ReferenceDataType


class QueryType(Enum):
    """Types of genomic queries."""

    VARIANT_LOOKUP = "variant_lookup"
    REGION_SCAN = "region_scan"
    GENE_ANNOTATION = "gene_annotation"
    POPULATION_FREQUENCY = "population_frequency"
    PATHWAY_VARIANTS = "pathway_variants"
    CLINICAL_SIGNIFICANCE = "clinical_significance"


@dataclass
class GenomicQuery:
    """High-level genomic query specification."""

    query_type: QueryType
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_cache_key(self) -> str:
        """Generate cache key for query."""
        data = {"type": self.query_type.value, "params": self.parameters}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class QueryResult:
    """Result of a genomic query."""

    query: GenomicQuery
    data: Any
    metadata: Dict[str, Any]
    pir_queries_used: int
    computation_time_ms: float


class PIRQueryBuilder:
    """
    Builds and executes privacy-preserving genomic queries.
    Translates high-level queries to PIR operations.
    """

    def __init__(self, pir_client: PIRClient, index_mapping: Dict[str, Dict[str, int]]):
        """
        Initialize query builder.

        Args:
            pir_client: PIR client for executing queries
            index_mapping: Mapping of genomic positions to database indices
        """
        self.pir_client = pir_client
        self.index_mapping = index_mapping

        # Query cache
        self.cache: Dict[str, QueryResult] = {}
        self.cache_size = 100

        logger.info("PIRQueryBuilder initialized")

    async def execute_query(self, query: GenomicQuery) -> QueryResult:
        """
        Execute a genomic query.

        Args:
            query: Query specification

        Returns:
            Query result
        """
        # Check cache
        cache_key = query.get_cache_key()
        if cache_key in self.cache:
            logger.info("Query result found in cache")
            return self.cache[cache_key]

        # Execute based on query type
        if query.query_type == QueryType.VARIANT_LOOKUP:
            result = await self._execute_variant_lookup(query)
        elif query.query_type == QueryType.REGION_SCAN:
            result = await self._execute_region_scan(query)
        elif query.query_type == QueryType.GENE_ANNOTATION:
            result = await self._execute_gene_annotation(query)
        elif query.query_type == QueryType.POPULATION_FREQUENCY:
            result = await self._execute_population_frequency(query)
        else:
            raise ValueError("Unsupported query type: {query.query_type}")

        # Cache result
        self._add_to_cache(cache_key, result)

        return result

    async def _execute_variant_lookup(self, query: GenomicQuery) -> QueryResult:
        """Execute variant lookup query."""
        import time

        start_time = time.time()

        # Extract parameters
        chromosome = query.parameters["chromosome"]
        position = query.parameters["position"]
        ref_allele = query.parameters.get("ref_allele", "")
        alt_allele = query.parameters.get("alt_allele", "")

        # Build variant key
        var_key = "{chromosome}:{position}"
        if ref_allele and alt_allele:
            var_key += ":{ref_allele}:{alt_allele}"

        # Get database index
        if var_key not in self.index_mapping["variants"]:
            # Variant not found
            return QueryResult(
                query=query,
                data=None,
                metadata={"found": False},
                pir_queries_used=0,
                computation_time_ms=0,
            )

        db_index = self.index_mapping["variants"][var_key]

        # Create PIR query
        pir_query = self.pir_client.create_query(db_index)

        # Execute query
        result_data = await self.pir_client.execute_query(pir_query)

        # Decode result
        decoded = self.pir_client.decode_response(result_data, "genomic")

        computation_time = (time.time() - start_time) * 1000

        return QueryResult(
            query=query,
            data=decoded,
            metadata={"found": True, "index": db_index},
            pir_queries_used=1,
            computation_time_ms=computation_time,
        )

    async def _execute_region_scan(self, query: GenomicQuery) -> QueryResult:
        """Execute region scan query."""
        import time

        start_time = time.time()

        # Extract parameters
        chromosome = query.parameters["chromosome"]
        start_pos = query.parameters["start"]
        end_pos = query.parameters["end"]

        # Find all indices in region
        indices = []
        for pos in range(start_pos, end_pos + 1):
            var_key = "{chromosome}:{pos}"
            if var_key in self.index_mapping["positions"]:
                indices.extend(self.index_mapping["positions"][var_key])

        if not indices:
            return QueryResult(
                query=query,
                data=[],
                metadata={"found": False, "region_size": end_pos - start_pos},
                pir_queries_used=0,
                computation_time_ms=0,
            )

        # Execute batch PIR queries
        results = await self.pir_client.batch_query(indices)

        # Decode results
        decoded_results = []
        for result in results:
            decoded = self.pir_client.decode_response(result, "genomic")
            decoded_results.append(decoded)

        computation_time = (time.time() - start_time) * 1000

        return QueryResult(
            query=query,
            data=decoded_results,
            metadata={
                "found": True,
                "region_size": end_pos - start_pos,
                "variants_found": len(decoded_results),
            },
            pir_queries_used=len(indices),
            computation_time_ms=computation_time,
        )

    async def _execute_gene_annotation(self, query: GenomicQuery) -> QueryResult:
        """Execute gene annotation query."""
        import time

        start_time = time.time()

        # Extract parameters
        gene_symbol = query.parameters["gene_symbol"]

        # Get gene region from index
        if gene_symbol not in self.index_mapping["genes"]:
            return QueryResult(
                query=query,
                data=None,
                metadata={"found": False},
                pir_queries_used=0,
                computation_time_ms=0,
            )

        gene_info = self.index_mapping["genes"][gene_symbol]

        # Query all variants in gene region
        region_query = GenomicQuery(
            query_type=QueryType.REGION_SCAN,
            parameters={
                "chromosome": gene_info["chromosome"],
                "start": gene_info["start"],
                "end": gene_info["end"],
            },
        )

        region_result = await self._execute_region_scan(region_query)

        # Filter for coding variants
        coding_variants = []
        for variant in region_result.data:
            if variant.get("gene_impact") in ["HIGH", "MODERATE"]:
                coding_variants.append(variant)

        computation_time = (time.time() - start_time) * 1000

        return QueryResult(
            query=query,
            data={
                "gene": gene_symbol,
                "variants": coding_variants,
                "total_variants": len(region_result.data),
            },
            metadata={"found": True, "gene_info": gene_info},
            pir_queries_used=region_result.pir_queries_used,
            computation_time_ms=computation_time,
        )

    async def _execute_population_frequency(self, query: GenomicQuery) -> QueryResult:
        """Execute population frequency query."""
        import time

        start_time = time.time()

        # Extract parameters
        variant_list = query.parameters["variants"]
        population = query.parameters["population"]

        # Query each variant
        frequencies = {}
        total_queries = 0

        for variant in variant_list:
            var_query = GenomicQuery(query_type=QueryType.VARIANT_LOOKUP, parameters=variant)

            result = await self._execute_variant_lookup(var_query)
            total_queries += result.pir_queries_used

            if result.data and "population_frequencies" in result.data:
                pop_freq = result.data["population_frequencies"].get(population, 0.0)
                var_key = "{variant['chromosome']}:{variant['position']}"
                frequencies[var_key] = pop_freq

        computation_time = (time.time() - start_time) * 1000

        return QueryResult(
            query=query,
            data=frequencies,
            metadata={"population": population, "variants_queried": len(variant_list)},
            pir_queries_used=total_queries,
            computation_time_ms=computation_time,
        )

    def _add_to_cache(self, key: str, result: QueryResult):
        """Add result to cache with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = result

    def build_variant_query(
        self,
        chromosome: str,
        position: int,
        ref_allele: Optional[str] = None,
        alt_allele: Optional[str] = None,
    ) -> GenomicQuery:
        """Build a variant lookup query."""
        params = {"chromosome": chromosome, "position": position}

        if ref_allele:
            params["ref_allele"] = ref_allele
        if alt_allele:
            params["alt_allele"] = alt_allele

        return GenomicQuery(query_type=QueryType.VARIANT_LOOKUP, parameters=params)

    def build_region_query(self, chromosome: str, start: int, end: int) -> GenomicQuery:
        """Build a region scan query."""
        return GenomicQuery(
            query_type=QueryType.REGION_SCAN,
            parameters={"chromosome": chromosome, "start": start, "end": end},
        )

    def build_gene_query(self, gene_symbol: str) -> GenomicQuery:
        """Build a gene annotation query."""
        return GenomicQuery(
            query_type=QueryType.GENE_ANNOTATION, parameters={"gene_symbol": gene_symbol}
        )

    def build_population_frequency_query(
        self, variants: List[Dict], population: str
    ) -> GenomicQuery:
        """Build a population frequency query."""
        return GenomicQuery(
            query_type=QueryType.POPULATION_FREQUENCY,
            parameters={"variants": variants, "population": population},
        )

    async def query_clinical_variants(
        self, gene_list: List[str], significance_filter: Optional[str] = None
    ) -> QueryResult:
        """
        Query for clinically significant variants in genes.

        Args:
            gene_list: List of gene symbols
            significance_filter: Filter by clinical significance

        Returns:
            Query result with clinical variants
        """
        import time

        start_time = time.time()

        all_clinical_variants = []
        total_queries = 0

        for gene in gene_list:
            # Query gene
            gene_query = self.build_gene_query(gene)
            gene_result = await self.execute_query(gene_query)

            if gene_result.data:
                total_queries += gene_result.pir_queries_used

                # Filter for clinical significance
                for variant in gene_result.data.get("variants", []):
                    clin_sig = variant.get("clinical_significance")

                    if clin_sig and (not significance_filter or clin_sig == significance_filter):
                        all_clinical_variants.append(
                            {"gene": gene, "variant": variant, "significance": clin_sig}
                        )

        computation_time = (time.time() - start_time) * 1000

        return QueryResult(
            query=GenomicQuery(
                query_type=QueryType.CLINICAL_SIGNIFICANCE,
                parameters={"genes": gene_list, "significance_filter": significance_filter},
            ),
            data=all_clinical_variants,
            metadata={
                "genes_queried": len(gene_list),
                "variants_found": len(all_clinical_variants),
            },
            pir_queries_used=total_queries,
            computation_time_ms=computation_time,
        )

    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query statistics."""
        query_types = {}
        total_pir_queries = 0
        total_computation_time = 0

        for result in self.cache.values():
            query_type = result.query.query_type.value
            if query_type not in query_types:
                query_types[query_type] = 0
            query_types[query_type] += 1

            total_pir_queries += result.pir_queries_used
            total_computation_time += result.computation_time_ms

        return {
            "cache_size": len(self.cache),
            "query_types": query_types,
            "total_pir_queries": total_pir_queries,
            "avg_computation_time_ms": (
                total_computation_time / len(self.cache) if self.cache else 0
            ),
            "index_mapping_stats": {
                "variants": len(self.index_mapping.get("variants", {})),
                "positions": len(self.index_mapping.get("positions", {})),
                "genes": len(self.index_mapping.get("genes", {})),
            },
        }


# Example usage
if __name__ == "__main__":
    # This is a demonstration of the query builder interface
    # In practice, would be used with actual PIR client and index

    # Example index mapping
    index_mapping = {
        "variants": {"chr1:100000:A:G": 42, "chr1:100100:C:T": 43, "chr1:100200:G:A": 44},
        "positions": {"chr1:100000": [42], "chr1:100100": [43], "chr1:100200": [44]},
        "genes": {"BRCA1": {"chromosome": "chr17", "start": 43044295, "end": 43125483}},
    }

    # Would create actual PIR client here
    # pir_client = PIRClient(servers, database_size)
    # builder = PIRQueryBuilder(pir_client, index_mapping)

    # Example queries
    print("Example Query Types:")

    # Variant lookup
    var_query = GenomicQuery(
        query_type=QueryType.VARIANT_LOOKUP,
        parameters={"chromosome": "chr1", "position": 100000, "ref_allele": "A", "alt_allele": "G"},
    )
    print("Variant Query: {var_query.parameters}")

    # Region scan
    region_query = GenomicQuery(
        query_type=QueryType.REGION_SCAN,
        parameters={"chromosome": "chr1", "start": 100000, "end": 100500},
    )
    print("Region Query: {region_query.parameters}")

    # Gene annotation
    gene_query = GenomicQuery(
        query_type=QueryType.GENE_ANNOTATION, parameters={"gene_symbol": "BRCA1"}
    )
    print("Gene Query: {gene_query.parameters}")
