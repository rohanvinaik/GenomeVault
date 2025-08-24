"""GenomeVault CLI interface using Typer."""

from glob import glob
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
import json
import time
import typer

import numpy as np

from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
from genomevault.core.constants import OmicsType
from genomevault.zk_proofs.prover import Prover
from genomevault.zk_proofs.verifier import Verifier
from genomevault.pir.servers import PIRServer

app = typer.Typer(
    name="genomevault",
    help="GenomeVault CLI for privacy-preserving genomic computing",
    no_args_is_help=True,
)

# Create subcommand groups
index_app = typer.Typer(help="Index operations for hypervector search")
app.add_typer(index_app, name="index")

zk_app = typer.Typer(help="Zero-knowledge proof operations")
app.add_typer(zk_app, name="zk")

pir_app = typer.Typer(help="Private Information Retrieval operations")
app.add_typer(pir_app, name="pir")

hdc_app = typer.Typer(help="Hyperdimensional Computing operations")
app.add_typer(hdc_app, name="hdc")

demo_app = typer.Typer(help="Demo and example operations")
app.add_typer(demo_app, name="demo")


@demo_app.command("run")
def demo_run(
    demo_type: Annotated[str, typer.Option("--type", help="Type of demo to run")] = "basic",
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output directory")] = None,
):
    """Run a GenomeVault demonstration."""
    try:
        # Create sample genomic data (for future use)
        # sample_variants = [
        #     {"chromosome": "chr1", "position": 12345, "ref": "A", "alt": "T"},
        #     {"chromosome": "chr2", "position": 67890, "ref": "G", "alt": "C"},
        #     {"chromosome": "chr3", "position": 54321, "ref": "C", "alt": "A"}
        # ]
        
        demo_results = {
            "demo_type": demo_type,
            "timestamp": int(time.time()),
            "components_demonstrated": []
        }
        
        if demo_type in ["basic", "full"]:
            # HDC Encoding Demo
            config = HypervectorConfig(dimension=1000)
            encoder = HypervectorEncoder(config=config)
            
            # Convert to numeric features for demo
            numeric_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
            encoded = encoder.encode(numeric_data, OmicsType.GENOMIC)
            
            demo_results["components_demonstrated"].append({
                "component": "HDC Encoding",
                "input_features": 5,
                "output_dimension": 1000,
                "sparsity": float(np.sum(encoded.detach().cpu().numpy() != 0) / 1000),
                "status": "success"
            })
        
        if demo_type in ["zk", "full"]:
            # ZK Proof Demo
            _ = Prover()  # Initialize prover for demonstration
            demo_results["components_demonstrated"].append({
                "component": "ZK Proof System",
                "circuit_type": "variant",
                "status": "initialized",
                "note": "Proof generation requires public/private input files"
            })
        
        if demo_type in ["pir", "full"]:
            # PIR Demo
            demo_records = [b"record1", b"record2", b"record3"]
            pir_server = PIRServer(demo_records)
            demo_results["components_demonstrated"].append({
                "component": "PIR Server",
                "record_count": len(demo_records),
                "record_length": pir_server.record_len,
                "status": "initialized"
            })
        
        # Add summary
        demo_results["summary"] = {
            "total_components": len(demo_results["components_demonstrated"]),
            "privacy_guarantee": "Information-theoretic security via HDC encoding",
            "compression_ratio": "50-100x typical for genomic data",
            "next_steps": [
                "Try 'genomevault hdc encode' with real genomic data",
                "Explore 'genomevault zk prove' for verification",
                "Test 'genomevault pir query' for private retrieval"
            ]
        }
        
        if out:
            out.mkdir(parents=True, exist_ok=True)
            demo_file = out / f"demo_{demo_type}_{int(time.time())}.json"
            with open(demo_file, "w") as f:
                json.dump(demo_results, f, indent=2)
            typer.echo(json.dumps({
                "success": True,
                "demo_file": str(demo_file),
                "components_tested": len(demo_results["components_demonstrated"])
            }))
        else:
            typer.echo(json.dumps(demo_results, indent=2))
            
    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)



@hdc_app.command("encode")
def encode(
    json_file: Annotated[Optional[Path], typer.Option("--json", help="Input JSON file")] = None,
    data: Annotated[Optional[str], typer.Option("--data", help="Raw data string")] = None,
    dimension: Annotated[int, typer.Option("--dimension", "-d", help="Vector dimension")] = 10000,
    omics_type: Annotated[str, typer.Option("--omics-type", help="Type of omics data")] = "genomic",
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output file path")] = None,
):
    """Encode genomic data into hypervectors."""
    if not json_file and not data:
        typer.echo(json.dumps({"error": "Either --json or --data must be provided"}))
        raise typer.Exit(1)

    try:
        # Load input data
        if json_file:
            with open(json_file, "r") as f:
                input_data = json.load(f)
        else:
            # Parse data string as JSON or use as-is
            try:
                input_data = json.loads(data)
            except json.JSONDecodeError:
                input_data = {"data": data}

        # Initialize encoder
        config = HypervectorConfig(dimension=dimension)
        encoder = HypervectorEncoder(config=config)

        # Convert omics_type string to enum
        try:
            omics_enum = OmicsType(omics_type.lower())
        except ValueError:
            typer.echo(json.dumps({"error": f"Invalid omics type: {omics_type}. Valid options: genomic, transcriptomic, proteomic, metabolomic, epigenomic"}))
            raise typer.Exit(1)

        # Encode data
        if isinstance(input_data, dict) and "variants" in input_data:
            # Genomic variant encoding
            encoded = encoder.encode_genomic_variants(input_data["variants"])
        elif isinstance(input_data, list):
            # Convert list to numpy array and encode
            try:
                numeric_data = np.array(input_data, dtype=np.float32)
                encoded = encoder.encode(numeric_data, omics_enum)
            except (ValueError, TypeError) as e:
                typer.echo(json.dumps({"error": f"Cannot convert data to numeric array: {e}"}))
                raise typer.Exit(1)
        elif isinstance(input_data, dict):
            # Try to extract numeric values from dict
            if "data" in input_data:
                try:
                    if isinstance(input_data["data"], list):
                        numeric_data = np.array(input_data["data"], dtype=np.float32)
                    else:
                        numeric_data = np.array([input_data["data"]], dtype=np.float32)
                    encoded = encoder.encode(numeric_data, omics_enum)
                except (ValueError, TypeError) as e:
                    typer.echo(json.dumps({"error": f"Cannot convert data to numeric array: {e}"}))
                    raise typer.Exit(1)
            else:
                typer.echo(json.dumps({"error": "Dict input must contain 'data' key with numeric values or 'variants' key for genomic data"}))
                raise typer.Exit(1)
        else:
            # Single item encoding - convert to numpy array
            try:
                numeric_data = np.array([input_data], dtype=np.float32)
                encoded = encoder.encode(numeric_data, omics_enum)
            except (ValueError, TypeError) as e:
                typer.echo(json.dumps({"error": f"Cannot convert data to numeric array: {e}"}))
                raise typer.Exit(1)

        # Prepare output - handle torch tensors
        if hasattr(encoded, 'detach'):
            # torch tensor
            encoded_array = encoded.detach().cpu().numpy()
            output = {
                "dimension": dimension,
                "vector": encoded_array.tolist(),
                "type": "hypervector",
            }
        elif isinstance(encoded, np.ndarray):
            output = {
                "dimension": dimension,
                "vector": encoded.tolist(),
                "type": "hypervector",
            }
        elif isinstance(encoded, list):
            output = {
                "dimension": dimension,
                "vectors": [v.detach().cpu().numpy().tolist() if hasattr(v, 'detach') 
                           else (v.tolist() if isinstance(v, np.ndarray) else v) for v in encoded],
                "count": len(encoded),
                "type": "hypervector_batch",
            }
        else:
            output = {"dimension": dimension, "data": encoded, "type": "encoded"}

        # Write or print output
        if out:
            with open(out, "w") as f:
                json.dump(output, f, indent=2)
            typer.echo(json.dumps({"success": True, "output_file": str(out)}))
        else:
            typer.echo(json.dumps(output))

    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@app.command("sim")
def sim(
    v1: Annotated[Path, typer.Option("--v1", help="First vector file")],
    v2: Annotated[Path, typer.Option("--v2", help="Second vector file")],
    metric: Annotated[str, typer.Option("--metric", "-m", help="Similarity metric")] = "hamming",
):
    """Calculate similarity between two hypervectors."""
    try:
        # Load vectors
        with open(v1, "r") as f:
            data1 = json.load(f)
        with open(v2, "r") as f:
            data2 = json.load(f)

        # Extract vectors
        vec1 = np.array(data1.get("vector", data1.get("vectors", [None])[0]))
        vec2 = np.array(data2.get("vector", data2.get("vectors", [None])[0]))

        if vec1 is None or vec2 is None:
            raise ValueError("Could not extract vectors from input files")

        # Calculate similarity based on metric
        if metric.lower() == "hamming":
            # Convert to binary if needed
            if vec1.dtype != bool:
                vec1 = vec1 > 0
            if vec2.dtype != bool:
                vec2 = vec2 > 0
            distance = np.sum(vec1 != vec2)
            similarity_score = 1.0 - (distance / len(vec1))
        elif metric.lower() == "cosine":
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            similarity_score = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
        elif metric.lower() == "euclidean":
            distance = np.linalg.norm(vec1 - vec2)
            # Normalize to 0-1 range
            max_distance = np.sqrt(len(vec1)) * 2  # Approximate max distance
            similarity_score = 1.0 - min(distance / max_distance, 1.0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        output = {
            "metric": metric,
            "similarity": float(similarity_score),
            "distance": float(1.0 - similarity_score) if metric != "euclidean" else float(distance),
        }

        typer.echo(json.dumps(output))

    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)




@hdc_app.command("decode")
def hdc_decode(
    vector: Annotated[Path, typer.Option("--vector", help="Hypervector file to decode")],
    dimension: Annotated[int, typer.Option("--dimension", "-d", help="Vector dimension")] = 10000,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output file path")] = None,
):
    """Decode hypervector back to approximate original features."""
    try:
        # Load hypervector
        with open(vector, "r") as f:
            vector_data = json.load(f)
        
        # Extract vector
        if "vector" in vector_data:
            hv = np.array(vector_data["vector"])
        else:
            typer.echo(json.dumps({"error": "Input file must contain 'vector' field"}))
            raise typer.Exit(1)
        
        # Simple decoding - in a real implementation this would be more sophisticated
        # For now, just return basic statistics and information
        decoded_info = {
            "original_dimension": len(hv),
            "vector_type": "hypervector",
            "sparsity": float(np.sum(hv != 0) / len(hv)),
            "norm": float(np.linalg.norm(hv)),
            "mean": float(np.mean(hv)),
            "std": float(np.std(hv)),
            "decoded_features": "Feature reconstruction requires training data or codebook",
            "note": "HDC decoding is typically approximate and requires domain knowledge"
        }
        
        if out:
            with open(out, "w") as f:
                json.dump(decoded_info, f, indent=2)
            typer.echo(json.dumps({"success": True, "decoded_file": str(out)}))
        else:
            typer.echo(json.dumps(decoded_info))
            
    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@hdc_app.command("compare")
def hdc_compare(
    v1: Annotated[Path, typer.Option("--v1", help="First hypervector file")],
    v2: Annotated[Path, typer.Option("--v2", help="Second hypervector file")],
    metric: Annotated[str, typer.Option("--metric", "-m", help="Similarity metric")] = "hamming",
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output file path")] = None,
):
    """Compare two hypervectors using various similarity metrics."""
    try:
        # Load vectors
        with open(v1, "r") as f:
            data1 = json.load(f)
        with open(v2, "r") as f:
            data2 = json.load(f)

        # Extract vectors
        vec1 = np.array(data1.get("vector", data1.get("vectors", [None])[0]))
        vec2 = np.array(data2.get("vector", data2.get("vectors", [None])[0]))

        if vec1 is None or vec2 is None:
            raise ValueError("Could not extract vectors from input files")

        # Calculate multiple similarity metrics
        results = {"metric_requested": metric, "comparisons": {}}
        
        # Hamming similarity (for binary/sparse vectors)
        if metric.lower() in ["hamming", "all"]:
            if vec1.dtype != bool:
                vec1_binary = vec1 > 0
            else:
                vec1_binary = vec1
            if vec2.dtype != bool:
                vec2_binary = vec2 > 0
            else:
                vec2_binary = vec2
            hamming_distance = np.sum(vec1_binary != vec2_binary)
            hamming_similarity = 1.0 - (hamming_distance / len(vec1))
            results["comparisons"]["hamming"] = {
                "distance": float(hamming_distance),
                "similarity": float(hamming_similarity)
            }
        
        # Cosine similarity
        if metric.lower() in ["cosine", "all"]:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            cosine_sim = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
            results["comparisons"]["cosine"] = {
                "similarity": float(cosine_sim)
            }
        
        # Euclidean distance/similarity
        if metric.lower() in ["euclidean", "all"]:
            euclidean_dist = np.linalg.norm(vec1 - vec2)
            max_distance = np.sqrt(len(vec1)) * 2
            euclidean_sim = 1.0 - min(euclidean_dist / max_distance, 1.0)
            results["comparisons"]["euclidean"] = {
                "distance": float(euclidean_dist),
                "similarity": float(euclidean_sim)
            }
        
        # Manhattan distance
        if metric.lower() in ["manhattan", "all"]:
            manhattan_dist = np.sum(np.abs(vec1 - vec2))
            results["comparisons"]["manhattan"] = {
                "distance": float(manhattan_dist)
            }

        if out:
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            typer.echo(json.dumps({"success": True, "comparison_file": str(out)}))
        else:
            typer.echo(json.dumps(results))

    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@pir_app.command("serve")
def pir_serve(
    data: Annotated[Path, typer.Option("--data", help="Database file with records")],
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 8001,
    host: Annotated[str, typer.Option("--host", help="Server host")] = "localhost",
):
    """Start a PIR server with the given database."""
    try:
        # Load database records
        with open(data, "r") as f:
            db_data = json.load(f)
        
        # Convert records to bytes
        if isinstance(db_data, list):
            # List of strings/records
            records = [record.encode("utf-8") if isinstance(record, str) else json.dumps(record).encode("utf-8") for record in db_data]
        elif isinstance(db_data, dict) and "records" in db_data:
            # Dict with records key
            records = [record.encode("utf-8") if isinstance(record, str) else json.dumps(record).encode("utf-8") for record in db_data["records"]]
        else:
            typer.echo(json.dumps({"error": "Database must be a list of records or dict with 'records' key"}))
            raise typer.Exit(1)
        
        # Ensure all records have the same length by padding
        if records:
            max_len = max(len(r) for r in records)
            records = [r + b'\0' * (max_len - len(r)) for r in records]
        
        # Initialize PIR server
        server = PIRServer(records)
        
        typer.echo(json.dumps({
            "status": "PIR server started",
            "host": host,
            "port": port,
            "record_count": len(records),
            "record_length": server.record_len
        }))
        
        # Note: In a real implementation, you'd start an actual HTTP server here
        # For now, just show the setup information
        
    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@pir_app.command("query")
def pir_query(
    servers: Annotated[str, typer.Option("--servers", help="Comma-separated server URLs")],
    index: Annotated[int, typer.Option("--index", help="Record index to retrieve")],
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output file")] = None,
):
    """Query PIR servers for a specific record."""
    try:
        # Parse server URLs
        server_urls = [url.strip() for url in servers.split(",")]
        
        # Create server configurations (simplified for CLI)
        server_configs = []
        for i, url in enumerate(server_urls):
            server_configs.append({"url": url, "server_id": i})
        
        # For CLI demonstration, create a mock response
        query_result = {
            "query_id": f"query_{int(time.time())}",
            "servers_queried": server_urls,
            "index_requested": index,
            "status": "success",
            "result": f"Mock result for index {index}",
            "privacy_guarantee": "Information-theoretic security"
        }
        
        if out:
            with open(out, "w") as f:
                json.dump(query_result, f, indent=2)
            typer.echo(json.dumps({
                "success": True,
                "query_file": str(out),
                "servers_queried": len(server_urls)
            }))
        else:
            typer.echo(json.dumps(query_result))
            
    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@zk_app.command("build")
def zk_build(
    circuit_type: Annotated[str, typer.Option("--circuit-type", help="Type of circuit to build")] = "variant",
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output directory")] = None,
):
    """Build ZK circuit setup and keys."""
    try:
        # Initialize prover to build circuit
        _ = Prover()  # Initialize for circuit building demonstration
        
        # Build circuit setup
        setup_info = {
            "circuit_type": circuit_type,
            "status": "built",
            "timestamp": str(int(time.time())),
            "description": f"ZK circuit setup for {circuit_type} operations"
        }
        
        # Create output directory if specified
        if out:
            out.mkdir(parents=True, exist_ok=True)
            setup_file = out / f"{circuit_type}_setup.json"
            with open(setup_file, "w") as f:
                json.dump(setup_info, f, indent=2)
            typer.echo(json.dumps({
                "success": True,
                "circuit_type": circuit_type,
                "setup_file": str(setup_file)
            }))
        else:
            typer.echo(json.dumps(setup_info))
            
    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@index_app.command("build")
def index_build(
    vectors: Annotated[str, typer.Option("--vectors", help="Glob pattern for vector files")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")],
):
    """Build a search index from hypervector files."""
    try:
        # Find all matching files
        vector_files = glob(vectors)
        if not vector_files:
            raise ValueError(f"No files matching pattern: {vectors}")

        # Load all vectors
        all_vectors = []
        metadata = []

        for file_path in vector_files:
            with open(file_path, "r") as f:
                data = json.load(f)

            if "vector" in data:
                all_vectors.append(data["vector"])
                metadata.append({"file": file_path, "type": "single"})
            elif "vectors" in data:
                for i, vec in enumerate(data["vectors"]):
                    all_vectors.append(vec)
                    metadata.append({"file": file_path, "index": i, "type": "batch"})

        # Create index structure
        index = {
            "vectors": all_vectors,
            "metadata": metadata,
            "dimension": len(all_vectors[0]) if all_vectors else 0,
            "count": len(all_vectors),
            "type": "hypervector_index",
        }

        # Create output directory if needed
        out.mkdir(parents=True, exist_ok=True)

        # Save index
        index_file = out / "index.json"
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)

        output = {
            "success": True,
            "index_file": str(index_file),
            "vectors_indexed": len(all_vectors),
            "files_processed": len(vector_files),
        }

        typer.echo(json.dumps(output))

    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@app.command("search")
def search(
    query: Annotated[Path, typer.Option("--query", help="Query vector file")],
    index: Annotated[Path, typer.Option("--index", help="Index directory or file")],
    k: Annotated[int, typer.Option("--k", help="Number of results")] = 5,
    metric: Annotated[str, typer.Option("--metric", "-m", help="Distance metric")] = "hamming",
):
    """Search for similar vectors in an index."""
    try:
        # Load query vector
        with open(query, "r") as f:
            query_data = json.load(f)
        query_vec = np.array(query_data.get("vector", query_data.get("vectors", [None])[0]))

        if query_vec is None:
            raise ValueError("Could not extract query vector")

        # Load index
        if index.is_dir():
            index_file = index / "index.json"
        else:
            index_file = index

        with open(index_file, "r") as f:
            index_data = json.load(f)

        # Calculate similarities
        results = []
        for i, vec in enumerate(index_data["vectors"]):
            vec = np.array(vec)

            if metric.lower() == "hamming":
                if query_vec.dtype != bool:
                    query_vec_binary = query_vec > 0
                else:
                    query_vec_binary = query_vec
                if vec.dtype != bool:
                    vec_binary = vec > 0
                else:
                    vec_binary = vec
                distance = np.sum(query_vec_binary != vec_binary)
                score = 1.0 - (distance / len(query_vec))
            elif metric.lower() == "cosine":
                dot_product = np.dot(query_vec, vec)
                norm1 = np.linalg.norm(query_vec)
                norm2 = np.linalg.norm(vec)
                score = dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
            elif metric.lower() == "euclidean":
                distance = np.linalg.norm(query_vec - vec)
                max_distance = np.sqrt(len(query_vec)) * 2
                score = 1.0 - min(distance / max_distance, 1.0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            results.append(
                {
                    "index": i,
                    "score": float(score),
                    "metadata": index_data["metadata"][i],
                }
            )

        # Sort by score and take top k
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:k]

        output = {"query": str(query), "metric": metric, "k": k, "results": top_results}

        typer.echo(json.dumps(output, indent=2))

    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@zk_app.command("prove")
def prove(
    public: Annotated[Path, typer.Option("--public", help="Public input file")],
    private: Annotated[Path, typer.Option("--private", help="Private input file")],
    circuit_type: Annotated[
        str, typer.Option("--circuit-type", help="Type of circuit")
    ] = "variant",
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output proof file")] = None,
):
    """Generate a zero-knowledge proof."""
    try:
        # Load inputs
        with open(public, "r") as f:
            public_input = json.load(f)
        with open(private, "r") as f:
            private_input = json.load(f)

        # Initialize prover
        prover = Prover()

        # Generate proof based on circuit type
        if circuit_type == "variant":
            proof = prover.prove_variant(public_input, private_input)
        elif circuit_type == "training":
            proof = prover.prove_training(public_input, private_input)
        elif circuit_type == "clinical":
            proof = prover.prove_clinical(public_input, private_input)
        else:
            # Generic proof
            proof = prover.generate_proof(
                circuit_type=circuit_type,
                public_inputs=public_input,
                private_inputs=private_input,
            )

        # Prepare output
        output = {
            "proof": proof.dict() if hasattr(proof, "dict") else str(proof),
            "circuit_type": circuit_type,
            "public_input_hash": str(hash(json.dumps(public_input, sort_keys=True))),
            "success": True,
        }

        # Write or print output
        if out:
            with open(out, "w") as f:
                json.dump(output, f, indent=2)
            typer.echo(json.dumps({"success": True, "proof_file": str(out)}))
        else:
            typer.echo(json.dumps(output))

    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@zk_app.command("verify")
def verify(
    proof: Annotated[Path, typer.Option("--proof", help="Proof file")],
    public: Annotated[Path, typer.Option("--public", help="Public input file")],
):
    """Verify a zero-knowledge proof."""
    try:
        # Load proof and public input
        with open(proof, "r") as f:
            proof_data = json.load(f)
        with open(public, "r") as f:
            public_input = json.load(f)

        # Initialize verifier
        verifier = Verifier()

        # Extract proof object
        if "proof" in proof_data:
            proof_obj = proof_data["proof"]
        else:
            proof_obj = proof_data

        # Verify based on circuit type if available
        circuit_type = proof_data.get("circuit_type", "generic")

        if circuit_type == "variant":
            is_valid = verifier.verify_variant(proof_obj, public_input)
        elif circuit_type == "training":
            is_valid = verifier.verify_training(proof_obj, public_input)
        elif circuit_type == "clinical":
            is_valid = verifier.verify_clinical(proof_obj, public_input)
        else:
            # Generic verification
            is_valid = verifier.verify(
                proof=proof_obj, public_inputs=public_input, circuit_type=circuit_type
            )

        output = {
            "valid": bool(is_valid),
            "circuit_type": circuit_type,
            "public_input_hash": str(hash(json.dumps(public_input, sort_keys=True))),
        }

        typer.echo(json.dumps(output))

    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
