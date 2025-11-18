#!/bin/bash
#
# GenomeVault SRA Bulk Download Helper
#
# This script helps download multiple samples from SRA efficiently.
# Works with SRA Explorer (https://sra-explorer.info/) output or manual lists.
#
# Usage:
#   # From SRA Explorer generated script
#   ./scripts/sra_bulk_download.sh --from-explorer explorer_script.sh
#
#   # From accession list
#   ./scripts/sra_bulk_download.sh --from-list accessions.txt
#
#   # Download specific project
#   ./scripts/sra_bulk_download.sh --project PRJNA000001
#
#   # Single accession
#   ./scripts/sra_bulk_download.sh --accession SRR000001

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-data/downloaded/fastq}"
THREADS="${THREADS:-4}"
MAX_SIZE_GB="${MAX_SIZE_GB:-100}"
USE_ASPERA="${USE_ASPERA:-false}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

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

check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing=()
    
    if ! command -v prefetch &> /dev/null; then
        missing+=("prefetch")
    fi
    
    if ! command -v fasterq-dump &> /dev/null; then
        missing+=("fasterq-dump")
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Install SRA Toolkit:"
        log_info "  conda install -c bioconda sra-tools"
        exit 1
    fi
    
    log_success "All dependencies found"
    
    # Check optional tools
    if command -v aria2c &> /dev/null; then
        log_info "aria2c found - will use for faster downloads"
    fi
    
    if command -v pigz &> /dev/null; then
        log_info "pigz found - will use for parallel compression"
    fi
}

check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG "$OUTPUT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        exit 1
    fi
    
    log_success "Sufficient disk space: ${available_gb}GB available"
}

get_run_size() {
    local accession=$1
    # Query SRA for run size
    local size=$(esearch -db sra -query "$accession" | \
                 efetch -format runinfo | \
                 awk -F',' 'NR==2 {print $8}')
    
    if [ -n "$size" ]; then
        echo "$size"
    else
        echo "0"
    fi
}

download_single_accession() {
    local accession=$1
    local output_dir=$2
    
    log_info "Processing $accession"
    
    # Check if already downloaded
    if [ -f "${output_dir}/${accession}_1.fastq.gz" ] || \
       [ -f "${output_dir}/${accession}.fastq.gz" ]; then
        log_warning "$accession already exists, skipping"
        return 0
    fi
    
    # Prefetch (download to cache)
    log_info "Downloading $accession to cache..."
    if [ "$USE_ASPERA" = "true" ]; then
        prefetch "$accession" --max-size "$MAX_SIZE_GB"g --ascp-path "$(which ascp)"
    else
        prefetch "$accession" --max-size "$MAX_SIZE_GB"g
    fi
    
    # Extract FASTQ
    log_info "Extracting FASTQ files..."
    fasterq-dump "$accession" \
        --outdir "$output_dir" \
        --split-files \
        --threads "$THREADS" \
        --progress \
        --mem 4G \
        --temp "$output_dir/tmp"
    
    # Compress output
    log_info "Compressing output files..."
    if command -v pigz &> /dev/null; then
        # Parallel compression
        pigz -p "$THREADS" "${output_dir}/${accession}"*.fastq 2>/dev/null || true
    else
        # Standard compression
        gzip "${output_dir}/${accession}"*.fastq 2>/dev/null || true
    fi
    
    # Clean up cache
    log_info "Cleaning up cache..."
    rm -rf ~/ncbi/public/sra/${accession}.sra 2>/dev/null || true
    
    log_success "Completed $accession"
}

download_from_list() {
    local list_file=$1
    local output_dir=$2
    
    if [ ! -f "$list_file" ]; then
        log_error "File not found: $list_file"
        exit 1
    fi
    
    # Count total accessions
    local total=$(wc -l < "$list_file")
    log_info "Found $total accessions to download"
    
    # Read and process each line
    local current=0
    while IFS= read -r accession; do
        # Skip empty lines and comments
        [[ -z "$accession" || "$accession" =~ ^#.*$ ]] && continue
        
        current=$((current + 1))
        log_info "[$current/$total] Processing $accession"
        
        if download_single_accession "$accession" "$output_dir"; then
            log_success "[$current/$total] Downloaded $accession"
        else
            log_error "[$current/$total] Failed $accession"
        fi
        
    done < "$list_file"
    
    log_success "Bulk download complete: $current/$total successful"
}

download_from_project() {
    local project_id=$1
    local output_dir=$2
    
    log_info "Fetching accessions for project $project_id..."
    
    # Check if Entrez Direct is installed
    if ! command -v esearch &> /dev/null; then
        log_error "Entrez Direct not found. Install it:"
        log_info "  conda install -c bioconda entrez-direct"
        exit 1
    fi
    
    # Get all run accessions for the project
    local accession_list="${output_dir}/${project_id}_accessions.txt"
    
    esearch -db sra -query "$project_id" | \
        efetch -format runinfo | \
        awk -F',' 'NR>1 {print $1}' > "$accession_list"
    
    local count=$(wc -l < "$accession_list")
    
    if [ "$count" -eq 0 ]; then
        log_error "No accessions found for project $project_id"
        exit 1
    fi
    
    log_success "Found $count runs for project $project_id"
    
    # Download from the generated list
    download_from_list "$accession_list" "$output_dir"
}

download_from_explorer_script() {
    local explorer_script=$1
    local output_dir=$2
    
    if [ ! -f "$explorer_script" ]; then
        log_error "File not found: $explorer_script"
        exit 1
    fi
    
    log_info "Extracting accessions from SRA Explorer script..."
    
    # Extract SRR/ERR/DRR accessions from the script
    local accession_list="${output_dir}/explorer_accessions.txt"
    grep -oE '(SRR|ERR|DRR)[0-9]+' "$explorer_script" | sort -u > "$accession_list"
    
    local count=$(wc -l < "$accession_list")
    
    if [ "$count" -eq 0 ]; then
        log_error "No accessions found in explorer script"
        exit 1
    fi
    
    log_success "Extracted $count unique accessions"
    
    # Download from the generated list
    download_from_list "$accession_list" "$output_dir"
}

generate_manifest() {
    local output_dir=$1
    local manifest_file="${output_dir}/DOWNLOAD_MANIFEST.json"
    
    log_info "Generating download manifest..."
    
    cat > "$manifest_file" << EOF
{
  "download_date": "$(date -Iseconds)",
  "output_directory": "$output_dir",
  "files": [
EOF
    
    local first=true
    for file in "${output_dir}"/*.fastq.gz; do
        [ -f "$file" ] || continue
        
        local basename=$(basename "$file")
        local size=$(du -h "$file" | cut -f1)
        local accession=$(echo "$basename" | grep -oE '(SRR|ERR|DRR)[0-9]+')
        
        if [ "$first" = true ]; then
            first=false
        else
            echo "    ," >> "$manifest_file"
        fi
        
        cat >> "$manifest_file" << EOF
    {
      "accession": "$accession",
      "filename": "$basename",
      "size": "$size",
      "path": "$file"
    }
EOF
    done
    
    cat >> "$manifest_file" << EOF

  ]
}
EOF
    
    log_success "Manifest created: $manifest_file"
}

print_usage() {
    cat << EOF
GenomeVault SRA Bulk Download Helper

Usage:
  $0 [OPTIONS]

Options:
  --accession SRR000001           Download single accession
  --from-list FILE                Download from accession list (one per line)
  --project PRJNA000001           Download all runs from project
  --from-explorer FILE            Download from SRA Explorer script
  
  --output-dir DIR                Output directory (default: data/downloaded/fastq)
  --threads N                     Number of threads (default: 4)
  --max-size N                    Max size per file in GB (default: 100)
  --use-aspera                    Use Aspera for faster downloads
  
  --help                          Show this help message

Examples:
  # Single accession
  $0 --accession SRR000001
  
  # From accession list
  $0 --from-list my_accessions.txt
  
  # Entire project
  $0 --project PRJNA000001
  
  # From SRA Explorer
  $0 --from-explorer sra_explorer_script.sh
  
  # Custom output and parallel
  $0 --from-list accessions.txt --output-dir my_data --threads 8

Environment Variables:
  OUTPUT_DIR        Output directory
  THREADS           Number of threads
  MAX_SIZE_GB       Max file size in GB
  USE_ASPERA        Use Aspera (true/false)

EOF
}

# Main script
main() {
    local mode=""
    local input_file=""
    local accession=""
    local project=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --accession)
                mode="single"
                accession="$2"
                shift 2
                ;;
            --from-list)
                mode="list"
                input_file="$2"
                shift 2
                ;;
            --project)
                mode="project"
                project="$2"
                shift 2
                ;;
            --from-explorer)
                mode="explorer"
                input_file="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --threads)
                THREADS="$2"
                shift 2
                ;;
            --max-size)
                MAX_SIZE_GB="$2"
                shift 2
                ;;
            --use-aspera)
                USE_ASPERA="true"
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    # Validate mode
    if [ -z "$mode" ]; then
        print_usage
        exit 1
    fi
    
    # Print configuration
    echo ""
    log_info "═══════════════════════════════════════════════════════════"
    log_info "GenomeVault SRA Bulk Downloader"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Mode: $mode"
    log_info "Output: $OUTPUT_DIR"
    log_info "Threads: $THREADS"
    log_info "Max size: ${MAX_SIZE_GB}GB"
    log_info "Aspera: $USE_ASPERA"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Check disk space (rough estimate: 5GB per sample)
    check_disk_space 10
    
    # Create temp directory
    mkdir -p "${OUTPUT_DIR}/tmp"
    
    # Execute based on mode
    case $mode in
        single)
            download_single_accession "$accession" "$OUTPUT_DIR"
            ;;
        list)
            download_from_list "$input_file" "$OUTPUT_DIR"
            ;;
        project)
            download_from_project "$project" "$OUTPUT_DIR"
            ;;
        explorer)
            download_from_explorer_script "$input_file" "$OUTPUT_DIR"
            ;;
    esac
    
    # Generate manifest
    generate_manifest "$OUTPUT_DIR"
    
    # Summary
    echo ""
    log_info "═══════════════════════════════════════════════════════════"
    log_success "Download Complete!"
    log_info "═══════════════════════════════════════════════════════════"
    log_info "Files saved to: $OUTPUT_DIR"
    
    local file_count=$(ls -1 "${OUTPUT_DIR}"/*.fastq.gz 2>/dev/null | wc -l)
    local total_size=$(du -sh "$OUTPUT_DIR" | cut -f1)
    
    log_info "Files downloaded: $file_count"
    log_info "Total size: $total_size"
    log_info "Manifest: ${OUTPUT_DIR}/DOWNLOAD_MANIFEST.json"
    
    echo ""
    log_info "Next steps:"
    log_info "1. Quality check: fastqc ${OUTPUT_DIR}/*.fastq.gz"
    log_info "2. Run GenomeVault: python benchmarks/run_alignment_optimized_pipeline.py"
    echo ""
}

# Run main
main "$@"
