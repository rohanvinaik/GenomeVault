#!/bin/bash

# GenomeVault Audit Fix Menu
# Interactive menu for applying fixes based on audit report v2

clear
echo "╔══════════════════════════════════════════════════════╗"
echo "║        GenomeVault Audit Fix Menu v2.0               ║"
echo "║                                                      ║"
echo "║  Based on audit report from 2025-07-27              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Current directory: $(pwd)"
echo ""

while true; do
    echo "═══════════════════════════════════════════════════════"
    echo "Main Menu:"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "  1) Run pre-flight check (show current state)"
    echo "  2) Quick fix: Add missing __init__.py files only"
    echo "  3) Apply ALL fixes (comprehensive)"
    echo "  4) Validate current state (detailed report)"
    echo "  5) View audit fixes documentation"
    echo "  6) Create backup only"
    echo "  7) Exit"
    echo ""
    echo -n "Select an option (1-7): "

    read choice

    case $choice in
        1)
            echo ""
            echo "Running pre-flight check..."
            echo "─────────────────────────────────────────"
            python3 preflight_check.py
            echo ""
            echo "Press Enter to continue..."
            read
            clear
            ;;

        2)
            echo ""
            echo "Adding missing __init__.py files..."
            echo "─────────────────────────────────────────"
            python3 quick_fix_init_files.py
            echo ""
            echo "Press Enter to continue..."
            read
            clear
            ;;

        3)
            echo ""
            echo "═══════════════════════════════════════════════════════"
            echo "COMPREHENSIVE FIX"
            echo "═══════════════════════════════════════════════════════"
            echo ""
            echo "This will:"
            echo "  • Create a backup of your entire codebase"
            echo "  • Add all missing __init__.py files"
            echo "  • Convert print() statements to logging"
            echo "  • Fix broad exception handlers"
            echo "  • Add TODO comments for complex functions"
            echo "  • Update documentation"
            echo ""
            echo -n "Are you sure you want to proceed? (y/n): "
            read confirm

            if [[ $confirm =~ ^[Yy]$ ]]; then
                echo ""
                echo "Applying comprehensive fixes..."
                echo "─────────────────────────────────────────"
                python3 fix_audit_issues.py
                echo ""
                echo "Fixes applied! Check the backup directory for original files."
            else
                echo "Cancelled."
            fi
            echo ""
            echo "Press Enter to continue..."
            read
            clear
            ;;

        4)
            echo ""
            echo "Running validation..."
            echo "─────────────────────────────────────────"
            python3 validate_audit_fixes.py
            echo ""
            echo "Check audit_validation_report.json for detailed results."
            echo ""
            echo "Press Enter to continue..."
            read
            clear
            ;;

        5)
            echo ""
            echo "Opening documentation..."
            if [ -f "AUDIT_FIXES_README.md" ]; then
                less AUDIT_FIXES_README.md
            else
                echo "Documentation not found!"
            fi
            clear
            ;;

        6)
            echo ""
            echo "Creating backup..."
            echo "─────────────────────────────────────────"
            timestamp=$(date +%Y%m%d_%H%M%S)
            backup_dir="../genomevault_backup_$timestamp"
            echo "Creating backup at: $backup_dir"
            cp -r . "$backup_dir"
            echo "Backup created!"
            echo ""
            echo "Press Enter to continue..."
            read
            clear
            ;;

        7)
            echo ""
            echo "Exiting..."
            exit 0
            ;;

        *)
            echo ""
            echo "Invalid option. Please select 1-7."
            echo ""
            echo "Press Enter to continue..."
            read
            clear
            ;;
    esac
done
