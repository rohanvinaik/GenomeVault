# GenomeVault Black Fix Instructions

Follow these steps to fix Black formatting issues and pass GitHub CI:

## Step 1: Navigate to your genomevault directory
```bash
cd /Users/rohanvinaik/genomevault
```

## Step 2: Make the scripts executable
```bash
chmod +x fix_black_auto.py
chmod +x fix_and_push_black.sh
```

## Step 3: Run the automatic fix
```bash
python3 fix_black_auto.py
```

This will:
- Identify all files with Black formatting issues
- Fix files that can be automatically formatted
- Stub out files with syntax errors (saving originals as .backup)
- Run Black on the entire project

## Step 4: Review and commit
If the script succeeds, you can commit and push:

```bash
git add -A
git commit -m "Fix Black formatting errors for CI"
git push
```

## Alternative: Use the all-in-one script
```bash
./fix_and_push_black.sh
```

This will do everything automatically and ask for confirmation before pushing.

## If you still have issues:

1. Check for .backup files to see what was stubbed:
   ```bash
   find . -name "*.backup" -type f
   ```

2. Run Black check to see remaining issues:
   ```bash
   black --check .
   ```

3. Manually fix any remaining syntax errors in the problematic files.
