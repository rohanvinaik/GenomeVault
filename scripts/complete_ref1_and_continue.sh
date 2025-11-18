#!/bin/bash
#
# Complete ref1 processing and continue pipeline
#

set -e

LAYER2_DIR="benchmark_results/enhanced_privacy_k13_phase123_optimized/layer2_reference_pool"
CONSENSUS="benchmark_results/enhanced_privacy_k13_20251025_183857/layer1_consensus/consensus.fa"

echo "Waiting for ref1 indexing to complete..."

# Wait for indexing to finish
while ps aux | grep -q "[s]amtools index.*ref1.sorted.bam"; do
    echo "  Indexing still running... $(date '+%H:%M:%S')"
    sleep 10
done

echo "✅ Indexing complete!"
echo ""

# Verify index was created
if [ ! -f "$LAYER2_DIR/ref1.sorted.bam.bai" ]; then
    echo "❌ ERROR: Index file not found!"
    exit 1
fi

echo "Step 2: Running variant calling for ref1..."
echo "Started: $(date '+%H:%M:%S')"

# Run variant calling
bcftools mpileup --threads 5 -Ou -f "$CONSENSUS" "$LAYER2_DIR/ref1.sorted.bam" | \
bcftools call --threads 5 -mv -Oz -o "$LAYER2_DIR/ref1.vcf.gz"

# Index VCF
bcftools index "$LAYER2_DIR/ref1.vcf.gz"

echo "✅ ref1 variant calling complete! $(date '+%H:%M:%S')"
echo ""

# Show ref1 completion
ls -lh "$LAYER2_DIR/ref1"*

echo ""
echo "=" * 70
echo "ref1 COMPLETE!"
echo "=" * 70
echo ""
echo "Next: Ready to process ref2-ref12"
echo ""
