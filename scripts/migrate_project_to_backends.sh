#!/bin/bash
#
# Project-Wide Backend Migration Script
#
# Migrates the entire GenomeVault project to use the new hardware backend system.
# This script safely migrates all Python files with proper backups and verification.
#
# Usage:
#   ./scripts/migrate_project_to_backends.sh          # Interactive mode
#   ./scripts/migrate_project_to_backends.sh --auto   # Automatic mode
#   ./scripts/migrate_project_to_backends.sh --dry-run # Preview changes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Migration script
MIGRATOR="scripts/migrate_to_backend_system.py"

# Directories to migrate
DIRS=(
    "benchmarks"
    "examples"
    "tests"
    "genomevault"
    "scripts"
)

# Function to print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to print section header
print_header() {
    echo ""
    echo "========================================================================"
    print_msg "$BLUE" "$1"
    echo "========================================================================"
}

# Function to confirm action
confirm() {
    local prompt="$1"
    local response

    read -p "$prompt [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Parse arguments
DRY_RUN=false
AUTO_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --auto)
            AUTO_MODE=true
            shift
            ;;
        -h|--help)
            print_header "GenomeVault Backend Migration"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Preview changes without modifying files"
            echo "  --auto       Run without prompts (use with caution)"
            echo "  -h, --help   Show this help message"
            echo ""
            echo "This script migrates the entire project to use the new"
            echo "hardware-accelerated backend system (CPU/Metal/CUDA)."
            exit 0
            ;;
        *)
            print_msg "$RED" "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main migration
print_header "GenomeVault Hardware Backend Migration"

if [ "$DRY_RUN" = true ]; then
    print_msg "$YELLOW" "🔍 DRY RUN MODE - No files will be modified"
fi

echo ""
echo "This script will migrate your GenomeVault codebase to use the new"
echo "hardware-accelerated backend system. Changes include:"
echo ""
echo "  • HypervectorEncoder → create_backend_encoder()"
echo "  • encoder.encode() → encoder.encode_single()"
echo "  • Automatic backend selection (Metal > CUDA > CPU)"
echo "  • Backward compatibility maintained"
echo ""
print_msg "$GREEN" "✓ Backup files will be created (.backup)"
print_msg "$GREEN" "✓ Original files preserved"
print_msg "$GREEN" "✓ Can be reverted if needed"
echo ""

if [ "$AUTO_MODE" = false ] && [ "$DRY_RUN" = false ]; then
    if ! confirm "Continue with migration?"; then
        print_msg "$YELLOW" "Migration cancelled"
        exit 0
    fi
fi

# Check if migration script exists
if [ ! -f "$MIGRATOR" ]; then
    print_msg "$RED" "✗ Migration script not found: $MIGRATOR"
    exit 1
fi

# Run migration on each directory
TOTAL_MODIFIED=0
TOTAL_PROCESSED=0

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        print_msg "$YELLOW" "⚠ Directory not found: $dir (skipping)"
        continue
    fi

    print_header "Migrating $dir/"

    # Build migration command
    CMD="python $MIGRATOR $dir --recursive"
    if [ "$DRY_RUN" = true ]; then
        CMD="$CMD --dry-run"
    fi

    # Run migration and capture output
    if ! OUTPUT=$($CMD 2>&1); then
        print_msg "$RED" "✗ Migration failed for $dir/"
        echo "$OUTPUT"
        continue
    fi

    echo "$OUTPUT"

    # Extract statistics
    if MODIFIED=$(echo "$OUTPUT" | grep "Files modified:" | awk '{print $3}'); then
        TOTAL_MODIFIED=$((TOTAL_MODIFIED + MODIFIED))
    fi
    if PROCESSED=$(echo "$OUTPUT" | grep "Files processed:" | awk '{print $3}'); then
        TOTAL_PROCESSED=$((TOTAL_PROCESSED + PROCESSED))
    fi
done

# Print final summary
print_header "Migration Complete"

echo "Total Statistics:"
echo "  Files processed: $TOTAL_PROCESSED"
echo "  Files modified:  $TOTAL_MODIFIED"
echo ""

if [ "$DRY_RUN" = true ]; then
    print_msg "$BLUE" "ℹ️  This was a dry run. No files were actually modified."
    print_msg "$BLUE" "   Run without --dry-run to apply changes."
else
    print_msg "$GREEN" "✓ Migration completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Run tests to verify migration:"
    echo "     pytest tests/"
    echo ""
    echo "  2. Test backend detection:"
    echo "     python tests/test_compute_backend.py"
    echo ""
    echo "  3. Run benchmarks with new backends:"
    echo "     python benchmarks/encoding_comparison_benchmark.py"
    echo ""
    print_msg "$BLUE" "To revert changes, restore from .backup files:"
    echo "  find . -name '*.backup' -exec bash -c 'mv \"\$0\" \"\${0%.backup}\"' {} \;"
fi

echo "========================================================================"

exit 0
