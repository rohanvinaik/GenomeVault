#!/bin/bash
# Daily GitHub traffic monitoring script
# Saves snapshots to prevent data loss after 14 days

REPO="rohanvinaik/GenomeVault"
OUTPUT_DIR="security_analysis/traffic_history"
DATE=$(date +%Y%m%d)

mkdir -p "$OUTPUT_DIR"

echo "📊 Capturing GitHub traffic data for $DATE..."

# Save clone data
gh api "/repos/$REPO/traffic/clones" > "$OUTPUT_DIR/clones_$DATE.json"

# Save view data
gh api "/repos/$REPO/traffic/views" > "$OUTPUT_DIR/views_$DATE.json"

# Save referrers (if any)
gh api "/repos/$REPO/traffic/popular/referrers" > "$OUTPUT_DIR/referrers_$DATE.json" 2>/dev/null

# Save popular paths (if any)
gh api "/repos/$REPO/traffic/popular/paths" > "$OUTPUT_DIR/paths_$DATE.json" 2>/dev/null

# Create summary
cat > "$OUTPUT_DIR/summary_$DATE.txt" <<SUMMARY
GitHub Traffic Summary - $DATE
Repository: $REPO

Clones (last 14 days):
$(gh api "/repos/$REPO/traffic/clones" | jq '{total: .count, unique: .uniques}')

Views (last 14 days):
$(gh api "/repos/$REPO/traffic/views" | jq '{total: .count, unique: .uniques}')

Latest daily activity:
$(gh api "/repos/$REPO/traffic/clones" | jq '.clones[-1]')
SUMMARY

echo "✅ Traffic data saved to: $OUTPUT_DIR/"
echo "   • clones_$DATE.json"
echo "   • views_$DATE.json"
echo "   • summary_$DATE.txt"
