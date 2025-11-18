#!/bin/bash
set -e

# Create 11 full-genome experimental BAMs for k=11 privacy
# Each BAM is experimental genome aligned to one guide reference

EXPERIMENTAL_R1="data/downloaded/fastq/ERR3239334_1.fastq.gz"
EXPERIMENTAL_R2="data/downloaded/fastq/ERR3239334_2.fastq.gz"
GUIDE_DIR="/Volumes/1TBStorage/guide_strands"
OUTPUT_DIR="data/experimental_strands/ERR3239334/alignment/k11_bams"
THREADS=10

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Creating 11 Full-Genome Experimental BAMs"
echo "=========================================="
echo "Experimental: $EXPERIMENTAL_R1, $EXPERIMENTAL_R2"
echo "Guide dir: $GUIDE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Threads: $THREADS"
echo ""

# Create alignments for ref1-ref11
for i in {1..11}; do
    GUIDE_FA="${GUIDE_DIR}/ref${i}.fa.gz"
    OUTPUT_BAM="${OUTPUT_DIR}/experimental_vs_ref${i}.sorted.bam"

    if [ -f "$OUTPUT_BAM" ]; then
        echo "✓ ref${i}: BAM already exists, skipping"
        continue
    fi

    if [ ! -f "$GUIDE_FA" ]; then
        echo "✗ ref${i}: Guide FASTA not found: $GUIDE_FA"
        continue
    fi

    echo "=========================================="
    echo "Aligning experimental → ref${i}"
    echo "Started: $(date)"
    echo "=========================================="

    # Align with minimap2
    minimap2 -ax sr -t $THREADS \
        "$GUIDE_FA" \
        "$EXPERIMENTAL_R1" \
        "$EXPERIMENTAL_R2" \
        2> "${OUTPUT_DIR}/experimental_vs_ref${i}.align.log" \
        | samtools view -h -bt "$GUIDE_FA" - \
        | samtools sort -@ $THREADS -m 8G -o "$OUTPUT_BAM" - \
        2> "${OUTPUT_DIR}/experimental_vs_ref${i}.sort.log"

    # Index BAM
    samtools index "$OUTPUT_BAM"

    # Get stats
    samtools flagstat "$OUTPUT_BAM" > "${OUTPUT_DIR}/experimental_vs_ref${i}.flagstat"

    SIZE=$(du -h "$OUTPUT_BAM" | awk '{print $1}')
    echo "✓ ref${i}: Complete ($SIZE)"
    echo "   $(date)"
    echo ""
done

echo "=========================================="
echo "All 11 BAMs Created!"
echo "=========================================="
ls -lh "$OUTPUT_DIR"/*.bam
echo ""
echo "Total size:"
du -sh "$OUTPUT_DIR"
