"""GenomeVault CLI interface using Typer."""

import json
from pathlib import Path
from typing import Optional, List
from glob import glob

import typer
from typing_extensions import Annotated
import numpy as np

from genomevault.hypervector_transform.hdc_encoder import HypervectorEncoder
from genomevault.zk_proofs.prover import Prover
from genomevault.zk_proofs.verifier import Verifier

app = typer.Typer(
    name="genomevault",
    help="GenomeVault CLI for privacy-preserving genomic computing",
    no_args_is_help=True,
)


@app.command("encode")
def encode(
    json_file: Annotated[Optional[Path], typer.Option("--json", help="Input JSON file")] = None,
    data: Annotated[Optional[str], typer.Option("--data", help="Raw data string")] = None,
    dimension: Annotated[int, typer.Option("--dimension", "-d", help="Vector dimension")] = 10000,
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output file path")] = None,
):
    """Encode genomic data into hypervectors."""
    if not json_file and not data:
        typer.echo(json.dumps({"error": "Either --json or --data must be provided"}))
        raise typer.Exit(1)
    
    try:
        # Load input data
        if json_file:
            with open(json_file, 'r') as f:
                input_data = json.load(f)
        else:
            # Parse data string as JSON or use as-is
            try:
                input_data = json.loads(data)
            except json.JSONDecodeError:
                input_data = {"data": data}
        
        # Initialize encoder
        encoder = HypervectorEncoder(dimension=dimension)
        
        # Encode data
        if isinstance(input_data, dict) and "variants" in input_data:
            # Genomic variant encoding
            encoded = encoder.encode_genomic_variants(input_data["variants"])
        elif isinstance(input_data, list):
            # List of items to encode
            encoded = [encoder.encode(item) for item in input_data]
        else:
            # Single item encoding
            encoded = encoder.encode(input_data)
        
        # Prepare output
        if isinstance(encoded, np.ndarray):
            output = {
                "dimension": dimension,
                "vector": encoded.tolist(),
                "type": "hypervector"
            }
        elif isinstance(encoded, list):
            output = {
                "dimension": dimension,
                "vectors": [v.tolist() if isinstance(v, np.ndarray) else v for v in encoded],
                "count": len(encoded),
                "type": "hypervector_batch"
            }
        else:
            output = {
                "dimension": dimension,
                "data": encoded,
                "type": "encoded"
            }
        
        # Write or print output
        if out:
            with open(out, 'w') as f:
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
        with open(v1, 'r') as f:
            data1 = json.load(f)
        with open(v2, 'r') as f:
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


# Create index subcommand group
index_app = typer.Typer(help="Index operations for hypervector search")
app.add_typer(index_app, name="index")


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
            with open(file_path, 'r') as f:
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
            "type": "hypervector_index"
        }
        
        # Create output directory if needed
        out.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index_file = out / "index.json"
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        output = {
            "success": True,
            "index_file": str(index_file),
            "vectors_indexed": len(all_vectors),
            "files_processed": len(vector_files)
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
        with open(query, 'r') as f:
            query_data = json.load(f)
        query_vec = np.array(query_data.get("vector", query_data.get("vectors", [None])[0]))
        
        if query_vec is None:
            raise ValueError("Could not extract query vector")
        
        # Load index
        if index.is_dir():
            index_file = index / "index.json"
        else:
            index_file = index
        
        with open(index_file, 'r') as f:
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
            
            results.append({
                "index": i,
                "score": float(score),
                "metadata": index_data["metadata"][i]
            })
        
        # Sort by score and take top k
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:k]
        
        output = {
            "query": str(query),
            "metric": metric,
            "k": k,
            "results": top_results
        }
        
        typer.echo(json.dumps(output, indent=2))
        
    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@app.command("prove")
def prove(
    public: Annotated[Path, typer.Option("--public", help="Public input file")],
    private: Annotated[Path, typer.Option("--private", help="Private input file")],
    circuit_type: Annotated[str, typer.Option("--circuit-type", help="Type of circuit")] = "variant",
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output proof file")] = None,
):
    """Generate a zero-knowledge proof."""
    try:
        # Load inputs
        with open(public, 'r') as f:
            public_input = json.load(f)
        with open(private, 'r') as f:
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
                private_inputs=private_input
            )
        
        # Prepare output
        output = {
            "proof": proof.dict() if hasattr(proof, 'dict') else str(proof),
            "circuit_type": circuit_type,
            "public_input_hash": str(hash(json.dumps(public_input, sort_keys=True))),
            "success": True
        }
        
        # Write or print output
        if out:
            with open(out, 'w') as f:
                json.dump(output, f, indent=2)
            typer.echo(json.dumps({"success": True, "proof_file": str(out)}))
        else:
            typer.echo(json.dumps(output))
            
    except Exception as e:
        typer.echo(json.dumps({"error": str(e)}))
        raise typer.Exit(1)


@app.command("verify")
def verify(
    proof: Annotated[Path, typer.Option("--proof", help="Proof file")],
    public: Annotated[Path, typer.Option("--public", help="Public input file")],
):
    """Verify a zero-knowledge proof."""
    try:
        # Load proof and public input
        with open(proof, 'r') as f:
            proof_data = json.load(f)
        with open(public, 'r') as f:
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
                proof=proof_obj,
                public_inputs=public_input,
                circuit_type=circuit_type
            )
        
        output = {
            "valid": bool(is_valid),
            "circuit_type": circuit_type,
            "public_input_hash": str(hash(json.dumps(public_input, sort_keys=True)))
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