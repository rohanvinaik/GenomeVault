#!/bin/bash
# One-command push to clean-slate
cd /Users/rohanvinaik/genomevault && \
git add fix_*.py validate_*.py preflight_check.py quick_fix_init_files.py generate_comparison_report.py *.sh AUDIT*.md REAL_AUDIT_STATUS.md QUICK_REFERENCE.txt PUSH_SUMMARY.md && \
git commit -m "Add audit fix scripts showing code is in good shape (only 334 files, not 45k)" && \
git checkout -b clean-slate 2>/dev/null || git checkout clean-slate && \
git push -u origin clean-slate && \
echo "✅ Successfully pushed to clean-slate branch!"
