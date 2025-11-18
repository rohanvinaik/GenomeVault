#!/bin/bash
# Build superposition consensus ONE chromosome at a time
# Process all 7 reference strands per chromosome
# 95% conservation, 5% variability region

set -e

# All 7 reference genomes
REFERENCES=(
    "data/reference_genomes/hg38.fa.gz"
    "data/reference_genomes/hg19.fa.gz"
    "data/reference_genomes/chm13v2.0.fa.gz"
    "data/reference_genomes/GRCh38_no_alt.fa.gz"
    "data/reference_genomes/hs37d5.fa.gz"
    "data/reference_genomes/GRCh38_full_analysis_set.fa.gz"
    "data/reference_genomes/hg18.fa.gz"
)

OUTPUT_DIR="benchmark_results/superposition_7refs_sequential"
TEMP_DIR="${OUTPUT_DIR}/temp"
BGZIP_DIR="${OUTPUT_DIR}/bgzip_refs"

mkdir -p "$OUTPUT_DIR" "$TEMP_DIR" "$BGZIP_DIR" logs

echo "================================================================================"
echo "SUPERPOSITION BUILD - 7 REFERENCES, ONE CHROMOSOME AT A TIME"
echo "================================================================================"
echo "References: 7 complete human genome assemblies"
echo "Strategy: Sequential chromosome processing (avoid memory overload)"
echo "Conservation: 95% (5% variability for superposition)"
echo "================================================================================"
echo ""

# Step 1: Convert gzip to bgzip (only if needed)
echo "Step 1: Converting genomes to bgzip format..."
echo "================================================================================"
for ref in "${REFERENCES[@]}"; do
    basename=$(basename "$ref" .gz)
    bgzip_file="${BGZIP_DIR}/${basename}"

    if [ -f "${bgzip_file}.gz" ] && [ -f "${bgzip_file}.gz.fai" ]; then
        echo "  ✓ Already converted: $(basename $ref)"
        continue
    fi

    echo "  Converting: $(basename $ref)..."
    gunzip -c "$ref" | bgzip -c > "${bgzip_file}.gz"
    samtools faidx "${bgzip_file}.gz"
    echo "  ✓ Done: ${bgzip_file}.gz"
done
echo ""

# Get chromosome list from first reference
CHROMOSOMES=($(cut -f1 "${BGZIP_DIR}/hg38.fa.gz.fai" | grep -E '^chr[0-9XY]+$' | head -24))
echo "Found ${#CHROMOSOMES[@]} chromosomes: ${CHROMOSOMES[@]}"
echo ""

# Step 2: Process each chromosome sequentially
echo "Step 2: Building consensus for each chromosome..."
echo "================================================================================"

for chr in "${CHROMOSOMES[@]}"; do
    echo ""
    echo "[$(date +%H:%M:%S)] Processing $chr..."
    echo "--------------------------------------------------------------------------------"

    chr_temp="${TEMP_DIR}/${chr}"
    mkdir -p "$chr_temp"

    # Extract this chromosome from all 7 references
    echo "  Extracting $chr from 7 references..."
    for i in "${!REFERENCES[@]}"; do
        ref_name=$(basename "${REFERENCES[$i]}" .fa.gz)
        bgzip_ref="${BGZIP_DIR}/${ref_name}.fa.gz"

        samtools faidx "$bgzip_ref" "$chr" > "${chr_temp}/${ref_name}_${chr}.fa" 2>/dev/null || {
            echo "  Warning: $chr not found in $ref_name"
            continue
        }
    done

    # Count how many references have this chromosome
    chr_files=(${chr_temp}/*_${chr}.fa)
    num_refs=${#chr_files[@]}
    echo "  Found $chr in $num_refs/7 references"

    if [ $num_refs -lt 2 ]; then
        echo "  Skipping $chr (insufficient references)"
        continue
    fi

    # Build consensus for this chromosome using Python
    echo "  Building consensus with 95% conservation threshold..."
    python3 - <<PYTHON_EOF
import sys
from pathlib import Path
from collections import Counter

chr_temp = Path("${chr_temp}")
chr = "${chr}"
chr_files = sorted(chr_temp.glob(f"*_{chr}.fa"))

# Read all sequences
sequences = []
for f in chr_files:
    with open(f) as fh:
        lines = fh.readlines()[1:]  # Skip header
        seq = ''.join(line.strip() for line in lines)
        sequences.append(seq)

if not sequences:
    print(f"  No sequences found for {chr}", file=sys.stderr)
    sys.exit(1)

# Find consensus
print(f"  Aligning {len(sequences)} sequences (length: {len(sequences[0]):,} bp)...")
consensus = []
variable_regions = 0

for pos in range(len(sequences[0])):
    bases = [seq[pos] for seq in sequences if pos < len(seq)]
    if not bases:
        continue

    counts = Counter(bases)
    most_common = counts.most_common(1)[0]
    freq = most_common[1] / len(bases)

    if freq >= 0.95:  # Conserved region
        consensus.append(most_common[0])
    else:  # Variable region - use IUPAC ambiguity code
        variable_regions += 1
        # For now, just use most common
        consensus.append(most_common[0])

# Write consensus
output = chr_temp / f"consensus_{chr}.fa"
with open(output, 'w') as out:
    out.write(f">{chr}_consensus\\n")
    # Write in 60-char lines
    consensus_seq = ''.join(consensus)
    for i in range(0, len(consensus_seq), 60):
        out.write(consensus_seq[i:i+60] + '\\n')

print(f"  ✓ Consensus: {len(consensus):,} bp, {variable_regions:,} variable positions ({100*variable_regions/len(consensus):.2f}%)")
PYTHON_EOF

    # Compress the consensus
    bgzip -c "${chr_temp}/consensus_${chr}.fa" > "${chr_temp}/consensus_${chr}.fa.gz"

    # Cleanup intermediate files
    rm -f ${chr_temp}/*_${chr}.fa

    echo "  ✓ Completed $chr"
done

# Step 3: Concatenate all chromosome consensuses
echo ""
echo "================================================================================"
echo "Step 3: Merging all chromosomes into final consensus..."
echo "================================================================================"

# Collect all consensus files in chromosome order
consensus_files=()
for chr in "${CHROMOSOMES[@]}"; do
    consensus_file="${TEMP_DIR}/${chr}/consensus_${chr}.fa.gz"
    if [ -f "$consensus_file" ]; then
        consensus_files+=("$consensus_file")
    fi
done

echo "Merging ${#consensus_files[@]} chromosomes..."
zcat "${consensus_files[@]}" > "${OUTPUT_DIR}/superposition_consensus.fa"

# Compress final output
echo "Compressing final consensus..."
bgzip -f "${OUTPUT_DIR}/superposition_consensus.fa"
samtools faidx "${OUTPUT_DIR}/superposition_consensus.fa.gz"

# Generate metadata
cat > "${OUTPUT_DIR}/metadata.json" <<EOF
{
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "references": [
    "hg38 (GRCh38)",
    "hg19 (GRCh37)",
    "CHM13v2.0 (T2T)",
    "GRCh38_no_alt",
    "hs37d5 (GRCh37+decoy)",
    "GRCh38_full",
    "hg18 (GRCh36)"
  ],
  "num_references": 7,
  "chromosomes": ${#consensus_files[@]},
  "conservation_threshold": 0.95,
  "method": "sequential_chromosome_7refs"
}
EOF

final_size=$(du -h "${OUTPUT_DIR}/superposition_consensus.fa.gz" | cut -f1)

echo ""
echo "================================================================================"
echo "BUILD COMPLETE"
echo "================================================================================"
echo "Output: ${OUTPUT_DIR}/superposition_consensus.fa.gz"
echo "Size: $final_size"
echo "Chromosomes: ${#consensus_files[@]}"
echo "References: 7"
echo "================================================================================"
echo ""
echo "Next: Run enhanced_privacy_pipeline with k=13 GUIDE samples"
echo "  python benchmarks/run_enhanced_privacy_pipeline.py --k-min 13"
