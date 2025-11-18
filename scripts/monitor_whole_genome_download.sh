#!/bin/bash
# Monitor whole genome reference download progress

echo "================================================================================"
echo "1000 Genomes Whole Genome Download Monitor"
echo "================================================================================"
echo ""

# Check if download is running
if pgrep -f "download_whole_genome_references.sh" > /dev/null; then
    echo "Status: DOWNLOAD IN PROGRESS"
else
    echo "Status: DOWNLOAD NOT RUNNING (may be complete or not started)"
fi

echo ""
echo "Downloaded files:"
ls -lh vcf_pool/*.vcf.gz 2>/dev/null | awk '{print "  " $9 " - " $5}' || echo "  None yet"

echo ""
echo "Total size so far:"
du -sh vcf_pool/ 2>/dev/null || echo "  0 B"

echo ""
echo "Download log (last 20 lines):"
echo "--------------------------------------------------------------------------------"
tail -20 logs/whole_genome_download_*.log 2>/dev/null || echo "No log file found"

echo ""
echo "================================================================================"
