#!/bin/bash
#
# GenomeVault Data Setup - Quick Start
#
# This script automates the complete setup of genomic data for GenomeVault.
# It will:
#   1. Check and install dependencies
#   2. Download reference genomes and test data
#   3. Organize files in the correct structure
#   4. Validate downloads
#   5. Generate configuration for GenomeVault pipeline
#
# Usage:
#   ./scripts/setup_genomic_data.sh [quick|full|custom]
#
#   quick  - Small test dataset (~500 MB, chr22 only)
#   full   - Comprehensive dataset (~5 GB, multiple samples)
#   custom - Use data_config.yaml for custom selection

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/downloaded"

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} ✓ $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')]${NC} ⚠ $1"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')]${NC} ✗ $1"
}

header() {
    echo ""
    echo -e "${MAGENTA}════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_banner() {
    echo ""
    echo -e "${CYAN}"
    cat << "EOF"
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║              GenomeVault Data Setup                   ║
    ║         Automated Genomic Data Acquisition            ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found"
        exit 1
    fi
    success "Python found: $(python3 --version)"
}

check_conda() {
    if command -v conda &> /dev/null; then
        success "Conda found: $(conda --version)"
        return 0
    else
        warning "Conda not found - some features will be limited"
        return 1
    fi
}

install_dependencies() {
    header "Installing Dependencies"
    
    log "Checking for conda..."
    if check_conda; then
        log "Installing bioinformatics tools..."
        
        # Check if tools already installed
        local tools_needed=()
        
        command -v wget &> /dev/null || tools_needed+=("wget")
        command -v aria2c &> /dev/null || tools_needed+=("aria2")
        command -v prefetch &> /dev/null || tools_needed+=("sra-tools")
        command -v samtools &> /dev/null || tools_needed+=("samtools")
        command -v fastqc &> /dev/null || tools_needed+=("fastqc")
        
        if [ ${#tools_needed[@]} -gt 0 ]; then
            log "Installing: ${tools_needed[*]}"
            conda install -c bioconda -c conda-forge "${tools_needed[@]}" -y
            success "Dependencies installed"
        else
            success "All dependencies already installed"
        fi
    else
        warning "Conda not available. Please install manually:"
        echo "  - wget or curl"
        echo "  - aria2c (optional, for faster downloads)"
        echo "  - SRA Toolkit (for SRA data)"
        echo "  - samtools (for BAM processing)"
        echo "  - fastqc (for quality control)"
    fi
}

create_directory_structure() {
    header "Creating Directory Structure"
    
    mkdir -p "$DATA_DIR"/{fastq,vcf,bam,reference,metadata}
    mkdir -p "$DATA_DIR"/tmp
    
    success "Created directory structure in $DATA_DIR"
}

download_quick_test_set() {
    header "Downloading Quick Test Dataset"
    
    log "This will download ~500 MB of data (chr22 only)"
    log "Perfect for quick testing and CI/CD"
    echo ""
    
    python3 "$SCRIPT_DIR/download_genomic_data.py" \
        --preset quick-test \
        --output-dir "$DATA_DIR"
    
    success "Quick test dataset downloaded"
}

download_full_test_set() {
    header "Downloading Comprehensive Test Dataset"
    
    log "This will download ~5 GB of data"
    log "Includes diverse samples and gold standards"
    echo ""
    
    # Reference genome
    python3 "$SCRIPT_DIR/download_genomic_data.py" \
        --reference hg38_chr22 \
        --output-dir "$DATA_DIR"
    
    # Multiple samples
    python3 "$SCRIPT_DIR/download_genomic_data.py" \
        --dataset 1kg_hg00096_exome \
        --output-dir "$DATA_DIR"
    
    python3 "$SCRIPT_DIR/download_genomic_data.py" \
        --dataset 1kg_na19238_wgs \
        --output-dir "$DATA_DIR"
    
    # VCF variants
    python3 "$SCRIPT_DIR/download_genomic_data.py" \
        --vcf 1kg_phase3_chr22 \
        --output-dir "$DATA_DIR"
    
    success "Full test dataset downloaded"
}

download_custom_set() {
    header "Downloading Custom Dataset"
    
    local config_file="${PROJECT_ROOT}/data_config.yaml"
    
    if [ ! -f "$config_file" ]; then
        warning "Configuration file not found: $config_file"
        log "Creating template configuration..."
        
        # Config template already created
        success "Template created at $config_file"
        log "Please edit the configuration and run again"
        exit 0
    fi
    
    log "Using configuration: $config_file"
    python3 "$SCRIPT_DIR/download_genomic_data.py" \
        --config "$config_file"
    
    success "Custom dataset downloaded"
}

validate_downloads() {
    header "Validating Downloads"
    
    local total_files=$(find "$DATA_DIR" -type f \( -name "*.fastq*" -o -name "*.vcf*" -o -name "*.bam" -o -name "*.fa*" \) | wc -l)
    local total_size=$(du -sh "$DATA_DIR" | cut -f1)
    
    log "Files downloaded: $total_files"
    log "Total size: $total_size"
    
    # Check for key files
    local has_reference=false
    local has_samples=false
    local has_variants=false
    
    if find "$DATA_DIR/reference" -name "*.fa" -o -name "*.fasta" | grep -q .; then
        has_reference=true
        success "Reference genome found"
    else
        warning "No reference genome found"
    fi
    
    if find "$DATA_DIR/fastq" -name "*.fastq*" | grep -q . || \
       find "$DATA_DIR/bam" -name "*.bam" | grep -q .; then
        has_samples=true
        success "Sample data found"
    else
        warning "No sample data found"
    fi
    
    if find "$DATA_DIR/vcf" -name "*.vcf*" | grep -q .; then
        has_variants=true
        success "Variant data found"
    else
        warning "No variant data found"
    fi
    
    if [ "$has_reference" = true ] && [ "$has_samples" = true ]; then
        success "Minimum required data present"
        return 0
    else
        error "Missing required data"
        return 1
    fi
}

create_quick_links() {
    header "Creating Quick Access Links"
    
    local current_dir="${PROJECT_ROOT}/data/current"
    mkdir -p "$current_dir"
    
    # Link to latest reference
    local latest_ref=$(find "$DATA_DIR/reference" -name "*.fa" -type f | head -1)
    if [ -n "$latest_ref" ]; then
        ln -sf "$latest_ref" "$current_dir/reference.fa"
        success "Linked reference genome"
    fi
    
    # Link to latest VCF
    local latest_vcf=$(find "$DATA_DIR/vcf" -name "*.vcf.gz" -type f | head -1)
    if [ -n "$latest_vcf" ]; then
        ln -sf "$latest_vcf" "$current_dir/variants.vcf.gz"
        success "Linked VCF file"
    fi
    
    success "Quick access links created in $current_dir"
}

run_quality_checks() {
    header "Running Quality Checks"
    
    if ! command -v fastqc &> /dev/null; then
        warning "FastQC not installed, skipping QC"
        return
    fi
    
    log "Running FastQC on FASTQ files..."
    local qc_dir="${DATA_DIR}/qc_reports"
    mkdir -p "$qc_dir"
    
    find "$DATA_DIR/fastq" -name "*.fastq.gz" | while read -r file; do
        log "QC: $(basename "$file")"
        fastqc "$file" -o "$qc_dir" -q
    done
    
    success "Quality reports in $qc_dir"
}

generate_manifest() {
    header "Generating Data Manifest"
    
    local manifest="${DATA_DIR}/DATA_MANIFEST.md"
    
    cat > "$manifest" << EOF
# GenomeVault Data Manifest

**Generated**: $(date)
**Location**: $DATA_DIR

## Summary

EOF
    
    # Count files by type
    local fastq_count=$(find "$DATA_DIR/fastq" -name "*.fastq*" 2>/dev/null | wc -l)
    local vcf_count=$(find "$DATA_DIR/vcf" -name "*.vcf*" 2>/dev/null | wc -l)
    local bam_count=$(find "$DATA_DIR/bam" -name "*.bam" 2>/dev/null | wc -l)
    local ref_count=$(find "$DATA_DIR/reference" -name "*.fa*" 2>/dev/null | wc -l)
    
    cat >> "$manifest" << EOF
- FASTQ files: $fastq_count
- VCF files: $vcf_count
- BAM files: $bam_count
- Reference genomes: $ref_count

## File Listing

### Reference Genomes
\`\`\`
EOF
    
    find "$DATA_DIR/reference" -name "*.fa*" -exec ls -lh {} \; >> "$manifest" 2>/dev/null || true
    
    cat >> "$manifest" << EOF
\`\`\`

### FASTQ Files
\`\`\`
EOF
    
    find "$DATA_DIR/fastq" -name "*.fastq*" -exec ls -lh {} \; >> "$manifest" 2>/dev/null || true
    
    cat >> "$manifest" << EOF
\`\`\`

### VCF Files
\`\`\`
EOF
    
    find "$DATA_DIR/vcf" -name "*.vcf*" -exec ls -lh {} \; >> "$manifest" 2>/dev/null || true
    
    cat >> "$manifest" << EOF
\`\`\`

## Usage

### Run GenomeVault Pipeline

\`\`\`bash
# With FASTQ input
python benchmarks/run_alignment_optimized_pipeline.py \\
  --format fastq \\
  --fastq data/downloaded/fastq/sample.fastq.gz \\
  --reference data/downloaded/reference/hg38_chr22.fa

# With VCF input
python benchmarks/run_alignment_optimized_pipeline.py \\
  --format vcf \\
  --vcf data/downloaded/vcf/variants.vcf.gz
\`\`\`

### Quick Links

- Reference: \`data/current/reference.fa\`
- Variants: \`data/current/variants.vcf.gz\`

EOF
    
    success "Manifest created: $manifest"
}

print_next_steps() {
    header "Setup Complete!"
    
    echo ""
    echo -e "${GREEN}✓ Genomic data successfully set up${NC}"
    echo ""
    echo -e "${CYAN}Data Location:${NC}"
    echo "  $DATA_DIR"
    echo ""
    echo -e "${CYAN}Quick Access:${NC}"
    echo "  Reference: data/current/reference.fa"
    echo "  Variants:  data/current/variants.vcf.gz"
    echo ""
    echo -e "${CYAN}Next Steps:${NC}"
    echo ""
    echo "1. View data manifest:"
    echo "   ${YELLOW}cat $DATA_DIR/DATA_MANIFEST.md${NC}"
    echo ""
    echo "2. Run GenomeVault pipeline (quick test):"
    echo "   ${YELLOW}python benchmarks/run_alignment_optimized_pipeline.py --preset production --quick${NC}"
    echo ""
    echo "3. Full pipeline run:"
    echo "   ${YELLOW}python benchmarks/run_alignment_optimized_pipeline.py --preset production${NC}"
    echo ""
    echo "4. View detailed guide:"
    echo "   ${YELLOW}cat docs/DATA_ACQUISITION_GUIDE.md${NC}"
    echo ""
    echo -e "${CYAN}For more data:${NC}"
    echo "  - List available datasets: ${YELLOW}python scripts/download_genomic_data.py --list-datasets${NC}"
    echo "  - Download specific dataset: ${YELLOW}python scripts/download_genomic_data.py --dataset <id>${NC}"
    echo "  - Bulk SRA download: ${YELLOW}./scripts/sra_bulk_download.sh --help${NC}"
    echo ""
}

main() {
    print_banner
    
    # Parse mode
    local mode="${1:-quick}"
    
    case $mode in
        quick)
            log "Mode: Quick test dataset (~500 MB)"
            ;;
        full)
            log "Mode: Full test dataset (~5 GB)"
            ;;
        custom)
            log "Mode: Custom configuration"
            ;;
        *)
            error "Unknown mode: $mode"
            echo "Usage: $0 [quick|full|custom]"
            exit 1
            ;;
    esac
    
    echo ""
    
    # Check Python
    check_python
    
    # Install dependencies
    install_dependencies
    
    # Create directory structure
    create_directory_structure
    
    # Download data based on mode
    case $mode in
        quick)
            download_quick_test_set
            ;;
        full)
            download_full_test_set
            ;;
        custom)
            download_custom_set
            ;;
    esac
    
    # Validate downloads
    if ! validate_downloads; then
        error "Validation failed - some required data may be missing"
        exit 1
    fi
    
    # Create quick access links
    create_quick_links
    
    # Optional QC
    if [ "$mode" = "full" ]; then
        run_quality_checks
    fi
    
    # Generate manifest
    generate_manifest
    
    # Print summary
    print_next_steps
}

# Run main
main "$@"
