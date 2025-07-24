# 🚀 Manual Steps to Fix Black Formatting and Push to GitHub

## Step 1: Navigate to GenomeVault Directory
```bash
cd /Users/rohanvinaik/genomevault
```

## Step 2: Run Black Formatter
```bash
# Format all Python files
black . --line-length=88

# Verify formatting is correct
black --check . --line-length=88
```

## Step 3: Check Git Status
```bash
git status
```

## Step 4: Add All Changes
```bash
git add -A
```

## Step 5: Commit Changes
```bash
git commit -m "Fix: Black code formatting for CI compliance

- Format all Python files with Black (line-length=88)
- Resolve formatting issues in 14 files identified by CI
- Ensure consistent code style across codebase
- Fix import statement formatting and line breaks

This fixes the CI black check that was failing with:
'14 files would be reformatted, 128 files would be left unchanged'

All files now pass: black --check --line-length=88 ."
```

## Step 6: Push to GitHub
```bash
git push origin main
```

## Alternative One-Liner
If you want to do it all at once:
```bash
cd /Users/rohanvinaik/genomevault && black . --line-length=88 && git add -A && git commit -m "Fix: Black code formatting for CI compliance" && git push origin main
```

## Verification
After pushing, check your GitHub Actions to see if the CI passes:
- Go to your GitHub repo
- Click on "Actions" tab
- Look for the latest workflow run
- Verify that the Black check now passes ✅

## If You Get Errors
1. **If Black fails**: Some files might have syntax errors. Check the output.
2. **If git commit fails**: You might need to configure git user:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```
3. **If git push fails**: Check if you're on the right branch and have push permissions.

## Expected Output
- Black should format ~14 files
- Git should show modified files
- Push should succeed without errors
- CI should pass on next run
