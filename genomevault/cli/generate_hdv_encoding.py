"""
CLI for generating HDV encodings from GDiff documents.

This tool generates task-specific hyperdimensional vector encodings from
comprehensive GDiff (Genomic Differential) documents using pre-configured
analysis schemas.

Usage:
    python -m genomevault.cli.generate_hdv_encoding \
        --vcf query.vcf.gz \
        --reference-pool path/to/ref_pool \
        --schema clinical_risk \
        --k 3
"""

import typer
import time
from pathlib import Path
from typing import Optional, List

from genomevault.differential_encoding.hdv_cache import HDVCacheManager
from genomevault.differential_encoding.gdiff.selective_hdv_encoder import SelectiveHDVEncoder
from genomevault.differential_encoding.gdiff.analysis_schemas import (
    get_schema,
    list_schemas,
    get_schema_summary,
)
from genomevault.differential_encoding.gdiff.schema import GDiffDocument

app = typer.Typer(
    name="generate-hdv-encoding",
    help="Generate HDV encodings from GDiff documents",
    no_args_is_help=True
)


@app.command()
def generate(
    vcf: str = typer.Option(..., "--vcf", help="Path to query VCF file"),
    reference_pool: str = typer.Option(..., "--reference-pool", help="Path to reference pool directory"),
    schema: str = typer.Option("clinical_risk", "--schema", help="Analysis schema to use"),
    k_anonymity: Optional[int] = typer.Option(None, "--k", help="k-anonymity level (default: auto)"),
    cache_dir: str = typer.Option("data/hdv_cache", "--cache-dir", help="HDV cache directory"),
    gdiff_path: Optional[str] = typer.Option(None, "--gdiff", help="Path to existing GDiff file (if already generated)"),
    force: bool = typer.Option(False, "--force", help="Force regeneration even if cached"),
    encrypt: bool = typer.Option(False, "--encrypt", help="Enable AES-256-GCM encryption for GDiff files"),
    encryption_password: Optional[str] = typer.Option(None, "--encryption-password", help="Password for encryption (prompted if not provided)"),
):
    """
    Generate HDV encoding from GDiff document.

    This command:
    1. Loads or generates GDiff for the query VCF + reference pool
    2. Applies the specified analysis schema (feature selection)
    3. Generates task-specific HDV encoding
    4. Caches the result for future queries

    Example:
        python -m genomevault.cli.generate_hdv_encoding \\
            --vcf query.vcf.gz \\
            --reference-pool benchmark_results/.../layer2_reference_pool \\
            --schema clinical_risk \\
            --k 3
    """
    typer.echo("=" * 80)
    typer.echo("HDV ENCODING GENERATOR (GDiff → Selective HDV)")
    typer.echo("=" * 80)

    # Handle encryption password prompt if needed
    if encrypt and not encryption_password:
        encryption_password = typer.prompt("Enter encryption password", hide_input=True)
        confirm_password = typer.prompt("Confirm encryption password", hide_input=True)

        if encryption_password != confirm_password:
            typer.echo("❌ ERROR: Passwords do not match")
            raise typer.Exit(1)

    # Initialize cache manager with encryption if enabled
    cache = HDVCacheManager(
        cache_root=Path(cache_dir),
        enable_encryption=encrypt,
        encryption_password=encryption_password
    )

    if encrypt:
        typer.echo("🔒 Encryption: AES-256-GCM enabled")

    # Auto-detect reference pool
    ref_pool_path = Path(reference_pool)
    if not ref_pool_path.exists():
        typer.echo(f"❌ ERROR: Reference pool not found: {reference_pool}")
        raise typer.Exit(1)

    available_refs = list(ref_pool_path.glob("*.vcf.gz"))
    num_refs = len(available_refs)

    if num_refs == 0:
        typer.echo(f"❌ ERROR: No VCF files found in reference pool")
        raise typer.Exit(1)

    # Auto-select k-anonymity
    if k_anonymity is None:
        k_anonymity = min(num_refs + 1, 11)
        typer.echo(f"Auto-selected k-anonymity: k={k_anonymity} (using {num_refs} references)")
    else:
        if k_anonymity > num_refs + 1:
            typer.echo(f"⚠️ WARNING: k={k_anonymity} requires {k_anonymity-1} references, but only {num_refs} available")
            typer.echo(f"Using k={num_refs+1} instead")
            k_anonymity = num_refs + 1

    # Generate query ID
    reference_ids = [ref.stem for ref in available_refs]
    query_id = cache.generate_query_id(vcf, reference_ids)

    typer.echo(f"\n📋 Configuration:")
    typer.echo(f"  Query VCF: {Path(vcf).name}")
    typer.echo(f"  Query ID: {query_id[:16]}...")
    typer.echo(f"  References: {num_refs}")
    typer.echo(f"  k-anonymity: {k_anonymity}")
    typer.echo(f"  Schema: {schema}")

    # Check if already cached
    if not force and cache.hdv_exists(query_id, k_anonymity, schema):
        typer.echo(f"\n✅ HDV already cached for k={k_anonymity}, schema={schema}")
        typer.echo(f"   Use --force to regenerate")

        hdv_path = cache.get_hdv(query_id, k_anonymity, schema)
        typer.echo(f"   Location: {hdv_path}")

        # Show stats
        stats = cache.get_cache_stats(query_id)
        typer.echo(f"\n📊 Cache Statistics:")
        typer.echo(f"   Total encodings: {stats['num_encodings']}")
        typer.echo(f"   k-levels: {stats['k_levels_available']}")
        typer.echo(f"   Schemas: {stats['schemas_available']}")
        return

    # Validate schema
    try:
        schema_obj = get_schema(schema)
    except KeyError:
        typer.echo(f"\n❌ ERROR: Unknown schema '{schema}'")
        typer.echo(f"\nAvailable schemas: {', '.join(list_schemas())}")
        raise typer.Exit(1)

    typer.echo(f"\n📐 Schema Details:")
    typer.echo(f"  Dimension: {schema_obj.dimension}D")
    typer.echo(f"  Features: {len(schema_obj.feature_categories)}")
    typer.echo(f"  Privacy level: {schema_obj.privacy_level}")
    typer.echo(f"  Expected encoding time: {schema_obj.encoding_time_ms} ms")
    typer.echo(f"  Expected HDV size: {schema_obj.hdv_size_bytes / 1024:.2f} KB")

    # Step 1: Load or generate GDiff
    typer.echo(f"\n[STEP 1/3] Loading GDiff document...")

    if gdiff_path and Path(gdiff_path).exists():
        typer.echo(f"  Loading existing GDiff: {Path(gdiff_path).name}")
        try:
            gdiff = GDiffDocument.load(Path(gdiff_path))
            typer.echo(f"  ✓ Loaded {len(gdiff.differential_variants)} variants")
        except Exception as e:
            typer.echo(f"  ❌ ERROR loading GDiff: {e}")
            raise typer.Exit(1)
    else:
        # Check if GDiff exists in cache
        gdiff_path_cached = cache.get_gdiff_path(query_id)

        if gdiff_path_cached.exists():
            typer.echo(f"  Loading cached GDiff: {gdiff_path_cached.name}")
            try:
                gdiff = GDiffDocument.load(gdiff_path_cached)
                typer.echo(f"  ✓ Loaded {len(gdiff.differential_variants)} variants")
            except Exception as e:
                typer.echo(f"  ❌ ERROR loading cached GDiff: {e}")
                raise typer.Exit(1)
        else:
            typer.echo(f"  ❌ ERROR: No GDiff found")
            typer.echo(f"\n  GDiff must be generated first. Options:")
            typer.echo(f"  1. Run differential encoding pipeline to generate GDiff")
            typer.echo(f"  2. Provide existing GDiff with --gdiff path/to/file.gdiff.gz")
            typer.echo(f"\n  Expected location: {gdiff_path_cached}")
            raise typer.Exit(1)

    # Step 2: Validate compatibility
    typer.echo(f"\n[STEP 2/3] Validating schema compatibility...")

    try:
        from genomevault.differential_encoding.gdiff.analysis_schemas import validate_schema_compatibility
        validate_schema_compatibility(schema_obj, gdiff)
        typer.echo(f"  ✓ Schema compatible with GDiff")
    except ValueError as e:
        typer.echo(f"  ❌ ERROR: {e}")
        raise typer.Exit(1)

    # Step 3: Generate HDV encoding
    typer.echo(f"\n[STEP 3/3] Generating HDV encoding...")

    encoder = SelectiveHDVEncoder(seed=42)

    start_time = time.time()
    try:
        hdv_encoding = encoder.encode(gdiff, schema_obj)
        encoding_duration = (time.time() - start_time) * 1000  # ms

        typer.echo(f"  ✓ HDV generated in {encoding_duration:.2f} ms")
        typer.echo(f"  Dimension: {hdv_encoding.dimension}D")
        typer.echo(f"  Variants encoded: {hdv_encoding.num_variants_encoded}")
        typer.echo(f"  Features used: {', '.join(hdv_encoding.features_used)}")
        typer.echo(f"  Size: {hdv_encoding.hdv_size_bytes / 1024:.2f} KB")

    except Exception as e:
        typer.echo(f"  ❌ ERROR generating HDV: {e}")
        raise typer.Exit(1)

    # Store in cache
    typer.echo(f"\n💾 Caching HDV encoding...")

    try:
        stored_path = cache.store_hdv(
            query_id=query_id,
            k_anonymity=k_anonymity,
            schema_name=schema,
            hdv_encoding=hdv_encoding,
            gdiff_path=gdiff_path_cached if not gdiff_path else Path(gdiff_path)
        )

        typer.echo(f"  ✓ Cached at: {stored_path}")

    except Exception as e:
        typer.echo(f"  ❌ ERROR caching HDV: {e}")
        raise typer.Exit(1)

    # Summary
    typer.echo(f"\n{'='*80}")
    typer.echo("✅ HDV ENCODING COMPLETE")
    typer.echo(f"{'='*80}")

    typer.echo(f"\n📊 Final Statistics:")
    typer.echo(f"  Query ID: {query_id[:16]}...")
    typer.echo(f"  k-anonymity: {k_anonymity}")
    typer.echo(f"  Schema: {schema}")
    typer.echo(f"  Dimension: {hdv_encoding.dimension}D")
    typer.echo(f"  Size: {hdv_encoding.hdv_size_bytes / 1024:.2f} KB")
    typer.echo(f"  Encoding time: {encoding_duration:.2f} ms")
    typer.echo(f"  Features: {', '.join(hdv_encoding.features_used)}")

    # Show updated cache stats
    stats = cache.get_cache_stats(query_id)
    typer.echo(f"\n📦 Cache Status:")
    typer.echo(f"  Total encodings: {stats['num_encodings']}")
    typer.echo(f"  k-levels: {stats['k_levels_available']}")
    typer.echo(f"  Schemas: {stats['schemas_available']}")

    typer.echo(f"\n✅ Ready for privacy-preserving queries!")
    typer.echo(f"{'='*80}")


@app.command()
def batch(
    vcf: str = typer.Option(..., "--vcf", help="Path to query VCF file"),
    reference_pool: str = typer.Option(..., "--reference-pool", help="Path to reference pool directory"),
    schemas: str = typer.Option("clinical_risk,pharmacogenomics", "--schemas", help="Comma-separated schemas"),
    k_levels: str = typer.Option("3,7,13", "--k-levels", help="Comma-separated k-anonymity levels"),
    cache_dir: str = typer.Option("data/hdv_cache", "--cache-dir", help="HDV cache directory"),
    gdiff_path: Optional[str] = typer.Option(None, "--gdiff", help="Path to existing GDiff file"),
):
    """
    Generate multiple HDV encodings in batch mode.

    Generates HDVs for multiple schemas and k-anonymity levels in one run.
    Useful for pre-populating the cache with common configurations.

    Example:
        python -m genomevault.cli.generate_hdv_encoding batch \\
            --vcf query.vcf.gz \\
            --reference-pool benchmark_results/.../layer2_reference_pool \\
            --schemas clinical_risk,pharmacogenomics,ancestry_inference \\
            --k-levels 3,7,13
    """
    typer.echo("=" * 80)
    typer.echo("BATCH HDV ENCODING GENERATOR")
    typer.echo("=" * 80)

    schema_list = [s.strip() for s in schemas.split(",")]
    k_list = [int(k.strip()) for k in k_levels.split(",")]

    total = len(schema_list) * len(k_list)
    typer.echo(f"\n📋 Batch Configuration:")
    typer.echo(f"  Schemas: {', '.join(schema_list)}")
    typer.echo(f"  k-levels: {', '.join(map(str, k_list))}")
    typer.echo(f"  Total encodings: {total}")

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, (schema, k) in enumerate([(s, k) for s in schema_list for k in k_list], 1):
        typer.echo(f"\n{'='*80}")
        typer.echo(f"[{i}/{total}] Generating: schema={schema}, k={k}")
        typer.echo(f"{'='*80}")

        try:
            # Call generate command programmatically
            from typer.testing import CliRunner
            runner = CliRunner()

            result = runner.invoke(app, [
                "generate",
                "--vcf", vcf,
                "--reference-pool", reference_pool,
                "--schema", schema,
                "--k", str(k),
                "--cache-dir", cache_dir,
            ] + (["--gdiff", gdiff_path] if gdiff_path else []))

            if result.exit_code == 0:
                if "already cached" in result.stdout:
                    skip_count += 1
                    typer.echo(f"  ⏭️ Skipped (already cached)")
                else:
                    success_count += 1
                    typer.echo(f"  ✅ Success")
            else:
                error_count += 1
                typer.echo(f"  ❌ Error")

        except Exception as e:
            error_count += 1
            typer.echo(f"  ❌ Error: {e}")

    # Final summary
    typer.echo(f"\n{'='*80}")
    typer.echo("BATCH GENERATION COMPLETE")
    typer.echo(f"{'='*80}")
    typer.echo(f"\n📊 Summary:")
    typer.echo(f"  Total: {total}")
    typer.echo(f"  Success: {success_count}")
    typer.echo(f"  Skipped: {skip_count}")
    typer.echo(f"  Errors: {error_count}")


if __name__ == "__main__":
    app()
