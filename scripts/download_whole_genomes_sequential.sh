#!/usr/bin/env bash
#
# GenomeVault Whole Genome Sequential Downloader
#
# CRITICAL: Downloads genomes ONE AT A TIME to avoid RAM exhaustion
# fasterq-dump creates 10-20 GB temp files per sample!
#
# Usage: ./scripts/download_whole_genomes_sequential.sh
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
FASTQ_DIR="data/downloaded/fastq"
TEMP_DIR="data/downloaded/tmp/fasterq_temp"  # Dedicated temp directory
MAX_MEM_GB=8  # Limit memory per download to 8 GB
THREADS=8     # Number of threads per download

# SRA accessions for 4 diverse whole genomes (arrays)
ACCESSIONS=("ERR3239334" "ERR3239276" "ERR3239454" "ERR3239475")
DESCRIPTIONS=("East Asian (CHS)" "South Asian (ITU)" "European (CEU)" "African (YRI)")

# Helper function to get description for accession
get_description() {
    local acc=$1
    case $acc in
        ERR3239334) echo "East Asian (CHS)" ;;
        ERR3239276) echo "South Asian (ITU)" ;;
        ERR3239454) echo "European (CEU)" ;;
        ERR3239475) echo "African (YRI)" ;;
        *) echo "Unknown" ;;
    esac
}

# Ensure directories exist
mkdir -p "$FASTQ_DIR"
mkdir -p "$TEMP_DIR"

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} ✓ $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')]${NC} ⚠ $1"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')]${NC} ✗ $1"
}

log_header() {
    echo ""
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════${NC}"
    echo ""
}

check_disk_space() {
    local required_gb=100
    local available_gb=$(df -g "$FASTQ_DIR" | awk 'NR==2 {print $4}')

    log_info "Checking disk space..."
    log_info "  Available: ${available_gb} GB"
    log_info "  Required:  ${required_gb} GB (for all 4 genomes)"

    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Insufficient disk space!"
        log_error "Please free up at least ${required_gb} GB"
        exit 1
    fi

    log_success "Sufficient disk space available"
}

check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v fasterq-dump &> /dev/null; then
        log_error "fasterq-dump not found!"
        log_error "Install SRA Toolkit:"
        log_error "  conda install -c bioconda sra-tools"
        exit 1
    fi

    if ! command -v pigz &> /dev/null; then
        log_warning "pigz not found - compression will be slower"
        log_info "Optional: conda install -c conda-forge pigz"
    fi

    log_success "Dependencies OK"
}

download_single_genome() {
    local accession=$1
    local description=$2
    local index=$3
    local total=$4

    log_header "[$index/$total] Downloading: $description ($accession)"

    # Check if already downloaded
    if [ -f "${FASTQ_DIR}/${accession}_1.fastq" ] || \
       [ -f "${FASTQ_DIR}/${accession}_1.fastq.gz" ]; then
        log_warning "Already exists: ${accession}"
        log_info "Skipping download (delete files to re-download)"
        return 0
    fi

    # Clean temp directory before download
    log_info "Cleaning temp directory..."
    rm -rf "${TEMP_DIR:?}/"*
    log_success "Temp directory cleaned"

    # Step 1: Prefetch (download .sra file to cache)
    log_info "Step 1/3: Downloading ${accession}.sra file from NCBI..."
    if prefetch "$accession" --max-size 100g 2>&1 | tee /tmp/prefetch.log; then
        log_success "Prefetch complete"
    else
        log_error "Prefetch failed for $accession"
        log_info "Check /tmp/prefetch.log for details"
        return 1
    fi

    # Step 2: Extract FASTQ (THIS IS WHERE RAM USAGE HAPPENS)
    log_info "Step 2/3: Extracting FASTQ files..."
    log_warning "This step creates large temp files (~10-20 GB)"
    log_info "Memory limit: ${MAX_MEM_GB}G, Threads: ${THREADS}, Temp: ${TEMP_DIR}"

    if fasterq-dump "$accession" \
        --outdir "$FASTQ_DIR" \
        --split-files \
        --threads "$THREADS" \
        --progress \
        --mem "${MAX_MEM_GB}G" \
        --temp "$TEMP_DIR" \
        --skip-technical 2>&1 | tee /tmp/fasterq-dump.log; then
        log_success "FASTQ extraction complete"
    else
        log_error "FASTQ extraction failed for $accession"
        log_info "Check /tmp/fasterq-dump.log for details"
        return 1
    fi

    # Step 3: Compress output files
    log_info "Step 3/3: Compressing FASTQ files..."

    if command -v pigz &> /dev/null; then
        # Parallel compression (faster)
        log_info "Using pigz for parallel compression..."
        if pigz -p "$THREADS" "${FASTQ_DIR}/${accession}"*.fastq 2>/dev/null; then
            log_success "Compression complete (pigz)"
        fi
    else
        # Standard compression
        log_info "Using gzip for compression..."
        if gzip "${FASTQ_DIR}/${accession}"*.fastq 2>/dev/null; then
            log_success "Compression complete (gzip)"
        fi
    fi

    # Clean up SRA cache
    log_info "Cleaning up SRA cache..."
    rm -rf ~/ncbi/public/sra/${accession}.sra 2>/dev/null || true

    # Clean up temp files
    log_info "Cleaning up temp files..."
    rm -rf "${TEMP_DIR:?}/"*

    # Verify output
    if [ -f "${FASTQ_DIR}/${accession}_1.fastq.gz" ] && \
       [ -f "${FASTQ_DIR}/${accession}_2.fastq.gz" ]; then
        local size1=$(du -h "${FASTQ_DIR}/${accession}_1.fastq.gz" | cut -f1)
        local size2=$(du -h "${FASTQ_DIR}/${accession}_2.fastq.gz" | cut -f1)
        log_success "Download complete!"
        log_info "  R1: ${size1}"
        log_info "  R2: ${size2}"
    else
        log_error "Output files not found!"
        return 1
    fi

    echo ""
}

print_banner() {
    echo ""
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║     ${BOLD}GenomeVault Whole Genome Sequential Downloader${NC}${CYAN}     ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}║  ${YELLOW}Downloads 4 genomes ONE AT A TIME to avoid RAM exhaustion${NC}${CYAN} ║${NC}"
    echo -e "${CYAN}║                                                               ║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}⚠ IMPORTANT:${NC}"
    echo -e "  • Each genome download creates ~10-20 GB temp files"
    echo -e "  • Running in parallel would exhaust 64 GB RAM"
    echo -e "  • This script runs downloads SEQUENTIALLY (safe)"
    echo -e "  • Expected time: ~2-4 hours per genome"
    echo -e "  • Total time: ~8-16 hours for all 4 genomes"
    echo ""
}

print_summary() {
    log_header "Download Summary"

    local completed=0
    local failed=0

    for accession in "${ACCESSIONS[@]}"; do
        local description=$(get_description "$accession")
        if [ -f "${FASTQ_DIR}/${accession}_1.fastq.gz" ]; then
            local size=$(du -sh "${FASTQ_DIR}/${accession}"_*.fastq.gz 2>/dev/null | awk '{sum+=$1} END {print sum "B"}' || echo "Unknown")
            log_success "${accession}: ${description} - $size"
            completed=$((completed + 1))
        else
            log_error "${accession}: ${description} - FAILED"
            failed=$((failed + 1))
        fi
    done

    echo ""
    log_info "Results: ${completed}/4 completed, ${failed}/4 failed"

    if [ $completed -eq 4 ]; then
        log_success "All 4 genomes downloaded successfully!"
        echo ""
        echo -e "${CYAN}Next steps:${NC}"
        echo -e "  1. Monitor downloads: ${YELLOW}./scripts/track_downloads.sh${NC}"
        echo -e "  2. View manifest: ${YELLOW}cat data/downloaded/DATA_MANIFEST.md${NC}"
        echo -e "  3. Run pipeline: ${YELLOW}python benchmarks/run_alignment_optimized_pipeline.py${NC}"
    else
        log_warning "Some downloads failed. Re-run this script to retry failed downloads."
    fi

    echo ""
}

# Trap Ctrl+C for graceful exit
trap 'echo ""; log_warning "Download interrupted by user"; exit 130' INT

# Main execution
main() {
    print_banner

    # Pre-flight checks
    check_dependencies
    check_disk_space

    # Get total number of genomes
    local total=${#ACCESSIONS[@]}
    local index=0

    log_header "Starting Sequential Download"
    log_info "Total genomes: $total"
    log_info "Estimated time: 8-16 hours"
    log_info "Press Ctrl+C to cancel"
    echo ""

    sleep 3

    # Download each genome sequentially
    for accession in "${ACCESSIONS[@]}"; do
        index=$((index + 1))
        description=$(get_description "$accession")

        if download_single_genome "$accession" "$description" "$index" "$total"; then
            log_success "[$index/$total] Successfully downloaded: $accession"
        else
            log_error "[$index/$total] Failed to download: $accession"
            log_warning "Continuing with next genome..."
        fi

        # Brief pause between downloads
        if [ $index -lt $total ]; then
            log_info "Waiting 10 seconds before next download..."
            sleep 10
        fi
    done

    # Print summary
    print_summary
}

# Run main
main
