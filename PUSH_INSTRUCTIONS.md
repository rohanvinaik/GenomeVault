# Instructions to push GenomeVault to GitHub

## Quick Push (Run these commands in Terminal):

```bash
cd /Users/rohanvinaik/genomevault
git add .
git commit -m "Implement GenomeVault 3.0 core architecture"
git push origin main
```

## Or use the script:

```bash
cd /Users/rohanvinaik/genomevault
./push_to_github.sh
```

## If you haven't set up the remote yet:

```bash
cd /Users/rohanvinaik/genomevault
git remote add origin https://github.com/Roh-codeur/genomevault.git
git branch -M main
git push -u origin main
```

## To check current status:

```bash
cd /Users/rohanvinaik/genomevault
git status
git remote -v
```
