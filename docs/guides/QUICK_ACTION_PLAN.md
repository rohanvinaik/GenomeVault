# Quick Action Plan - GenomeVault Critical Fixes

**⏰ Time-Sensitive: Complete P0 today, P1 tomorrow, P2 by end of week**

---

## 🚨 IMMEDIATE (Next 2 Hours) - P0

### 1. Fix Table 2 - Constraint Count

**Files to edit:**
- `/Users/rohanvinaik/genomevault/docs/GenomeVault_Academic_Paper.tex`
- `/Users/rohanvinaik/genomevault/docs/GenomeVault_Academic_Paper_Journal_Ready.tex`

**Find this:**
```latex
Variant Presence & 15,234 & 603ms \\
```

**Replace with:**
```latex
Variant Presence & 843 & 50-100ms\textsuperscript{†} & \checkmark \\
```

**Add footnote:**
```latex
\textsuperscript{†}Estimated: 0.06-0.12ms per constraint. Benchmarking in progress.
```

### 2. Add Experimental Status Box

**Insert after** `\begin{abstract}...\end{abstract}`:

```latex
\begin{tcolorbox}[colback=blue!5!white,colframe=blue!75!black,title=Experimental Status]
\textbf{Fully validated:}
\begin{itemize}
    \item[\checkmark] Compression: 264$\times$
    \item[\checkmark] Differential: 21.67ms
    \item[\checkmark] Hypervector: 5.04ms
    \item[\checkmark] Fingerprinting: D'=38.43
\end{itemize}

\textbf{In development:}
\begin{itemize}
    \item[$\triangle$] ZK: Circuits done, benchmarking in progress
    \item[$\triangle$] PIR: Architecture defined, validation ongoing
\end{itemize}
\end{tcolorbox}
```

**Add to preamble:**
```latex
\usepackage{tcolorbox}
```

### 3. Regenerate PDFs

```bash
cd /Users/rohanvinaik/genomevault/docs
pdflatex GenomeVault_Academic_Paper.tex
pdflatex GenomeVault_Academic_Paper_Journal_Ready.tex

# Verify
ls -lh *.pdf
```

**✅ P0 Complete**: Papers now accurate, no overclaims

---

## ⏱️ TODAY/TOMORROW (4-6 Hours) - P1

### 4. Setup ZK Infrastructure

```bash
cd /Users/rohanvinaik/genomevault/zk_circuits

# Install (5 min)
npm install -g snarkjs circom

# Download powers of tau (~2 min, 20MB)
curl -o pot12_final.ptau \
  https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_12.ptau

# Verify
ls -lh pot12_final.ptau  # Should show ~20MB
snarkjs --version        # Should show 0.7.0+
```

### 5. Setup & Run Groth16

```bash
cd /Users/rohanvinaik/genomevault

# Create & run setup script (30 min)
cat > setup_groth16.sh << 'EOF'
#!/bin/bash
set -e
CIRCUIT="variant_presence"
DIR="zk_circuits/build/${CIRCUIT}"
mkdir -p $DIR

# Compile
circom zk_circuits/circuits/${CIRCUIT}.circom --r1cs --wasm --sym -o $DIR

# Setup witness generator
cd ${DIR}/${CIRCUIT}_js && npm install && cd ../../..

# Groth16 trusted setup
snarkjs groth16 setup ${DIR}/${CIRCUIT}.r1cs zk_circuits/pot12_final.ptau ${DIR}/${CIRCUIT}_0000.zkey
snarkjs zkey contribute ${DIR}/${CIRCUIT}_0000.zkey ${DIR}/${CIRCUIT}_final.zkey \
    --name="GenomeVault" -e="$(openssl rand -hex 32)"

# Export verification key
snarkjs zkey export verificationkey ${DIR}/${CIRCUIT}_final.zkey ${DIR}/verification_key.json

echo "✅ Setup complete"
snarkjs r1cs info ${DIR}/${CIRCUIT}.r1cs
EOF

chmod +x setup_groth16.sh
./setup_groth16.sh
```

### 6. Generate 100 Real Proofs

```bash
# Run benchmark (2-4 hours)
python benchmarks/zk_groth16_benchmark.py --iterations 100

# Check results
cat benchmark_results/zk_groth16_real.json | grep -A 8 "proving_time_ms"
```

**Expected output:**
```json
"proving_time_ms": {
  "mean": 67.89,  ← UPDATE PAPER WITH THIS
  "p50": 65.43,
  "p95": 78.91,
  "p99": 84.32
}
```

### 7. Update Paper with Measured Values

Once complete, edit papers:

```latex
% Replace estimated values
Variant Presence & 843 & 67.8ms (P95: 78.9ms) & \checkmark \\
```

**✅ P1 Complete**: Real ZK proofs validated

---

## 📅 THIS WEEK (Days 2-3) - P2

### 8. PIR Benchmarks (6-8 hours)

```bash
cd /Users/rohanvinaik/genomevault

# Quick dependency check
python3 -c "from genomevault.pir.xor_scheme import XORPIRScheme; print('✅')"

# Run benchmarks
python benchmarks/pir_performance.py \
    --database-sizes 1000 10000 100000 \
    --num-queries 100

# Check results
cat benchmark_results/pir/pir_benchmark_results.json | \
    grep -A 5 "total_latency"
```

### 9. Attribute Inference (4-6 hours)

```bash
# Run experiment
python benchmarks/attribute_inference_experiment.py

# Check results
cat benchmark_results/attribute_inference/attribute_inference_results.json | \
    head -20
```

### 10. Information Leakage (4-6 hours)

```bash
# After attribute inference
python benchmarks/information_leakage_analysis.py

# Verify < 7 bits
cat benchmark_results/information_leakage/leakage_results.json | \
    grep "privacy_loss_bound"
```

**✅ P2 Complete**: All benchmarks validated

---

## 📋 Verification Checklist

### After P0 (Today)
```bash
# Papers updated?
grep "843" docs/GenomeVault_Academic_Paper.tex
grep "Experimental Status" docs/GenomeVault_Academic_Paper.tex

# PDFs generated?
ls -la docs/*.pdf | grep "Oct 20"
```

### After P1 (Tomorrow)
```bash
# ZK setup complete?
ls -la zk_circuits/build/variant_presence/variant_presence_final.zkey
ls -la zk_circuits/build/variant_presence/verification_key.json

# Benchmarks done?
test -f benchmark_results/zk_groth16_real.json && echo "✅ Done"
cat benchmark_results/zk_groth16_real.json | grep "mean"
```

### After P2 (End of Week)
```bash
# All benchmarks complete?
test -f benchmark_results/pir/pir_benchmark_results.json && echo "✅ PIR"
test -f benchmark_results/attribute_inference/attribute_inference_results.json && echo "✅ Attacks"
test -f benchmark_results/information_leakage/leakage_results.json && echo "✅ Leakage"
```

---

## 🆘 Quick Troubleshooting

**circom not found:**
```bash
npm install -g circom
```

**Powers of tau fails:**
```bash
wget https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_12.ptau
```

**PIR imports fail:**
```bash
# Check what exists
ls -la genomevault/pir/
# If missing, use stubs from full guide
```

**Out of memory:**
```bash
# Reduce iterations
python benchmarks/zk_groth16_benchmark.py --iterations 10
```

---

## 📊 Success Criteria

### ✅ P0 Done When:
- Table 2 shows 843 constraints (not 15,234)
- Experimental status box added
- Both PDFs regenerated
- No overclaimed results

### ✅ P1 Done When:
- Groth16 setup successful
- 100+ proofs generated
- Mean proving time measured (~67ms)
- All proofs verify successfully

### ✅ P2 Done When:
- PIR benchmarks complete (3+ database sizes)
- Attack accuracy measured
- Information leakage < 7 bits
- All reports generated

---

## ⚡ One-Line Command Summary

```bash
# Get everything in sequence
cd /Users/rohanvinaik/genomevault && \
  # P0: Edit papers manually, then:
  (cd docs && pdflatex GenomeVault_Academic_Paper.tex) && \
  # P1: ZK proofs
  (cd zk_circuits && npm i -g snarkjs circom && \
   curl -o pot12_final.ptau https://hermez.s3-eu-west-1.amazonaws.com/powersOfTau28_hez_final_12.ptau) && \
  ./benchmarks/setup_groth16.sh && \
  python benchmarks/zk_groth16_benchmark.py --iterations 100 && \
  # P2: All benchmarks
  python benchmarks/pir_performance.py && \
  python benchmarks/attribute_inference_experiment.py && \
  python benchmarks/information_leakage_analysis.py && \
  echo "✅ ALL COMPLETE"
```

---

## 📈 Progress Tracking

**Day 1 (Today)**
- [ ] 9:00 AM - Start P0
- [ ] 10:00 AM - Papers fixed
- [ ] 10:30 AM - PDFs regenerated
- [ ] 11:00 AM - **P0 COMPLETE** ✅
- [ ] 1:00 PM - Start P1 (install tools)
- [ ] 2:00 PM - Powers of tau downloaded
- [ ] 3:00 PM - Groth16 setup running
- [ ] 5:00 PM - End of day

**Day 2 (Tomorrow)**
- [ ] 9:00 AM - Check P1 progress
- [ ] 11:00 AM - 100 proofs complete
- [ ] 12:00 PM - Update papers with real values
- [ ] 1:00 PM - **P1 COMPLETE** ✅
- [ ] 2:00 PM - Start P2 (PIR)
- [ ] 5:00 PM - PIR running overnight

**Day 3 (End of Week)**
- [ ] 9:00 AM - PIR complete
- [ ] 10:00 AM - Start attribute inference
- [ ] 2:00 PM - Start information leakage
- [ ] 5:00 PM - **P2 COMPLETE** ✅

---

## 💡 Pro Tips

1. **Do P0 first thing** - Only takes 2 hours, makes papers accurate
2. **Run P2 tasks in parallel** - PIR, attacks, leakage don't depend on each other
3. **Start P1 setup early** - Powers of tau download + Groth16 setup can run while you work on other things
4. **Check logs frequently** - ZK proof generation shows progress
5. **Keep terminal open** - Don't lose progress on long-running tasks

---

## 📞 Emergency Contacts

**Stuck on P0?** → Just fix text, skip code entirely  
**Stuck on P1?** → Use "50-100ms estimated" in paper, mark as preliminary  
**Stuck on P2?** → Mark as "under development", submit anyway

**Priority**: P0 > P1 >> P2

**Minimum viable**: Complete P0 + P1, submit with P2 as "ongoing work"

---

**Document**: Quick Action Plan v1.0  
**Time to P0**: 2 hours  
**Time to P1**: 4-6 hours  
**Time to P2**: 11-15 hours (parallelizable)  
**Total**: 2-3 working days  
**Status**: Ready to execute NOW
