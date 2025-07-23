#!/bin/bash
# setup_and_test_clinical_validation.sh

echo "🏥 Setting up GenomeVault Clinical Validation Module"
echo "===================================================="

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# 1. Create necessary directories
echo ""
echo "1️⃣ Creating directory structure..."
mkdir -p clinical_validation/reports
mkdir -p clinical_validation/data
check_status "Directory creation"

# 2. Install required packages
echo ""
echo "2️⃣ Installing required packages..."
pip install pandas numpy scikit-learn matplotlib seaborn > /dev/null 2>&1
check_status "Package installation"

# 3. Download sample clinical data if needed
echo ""
echo "3️⃣ Setting up sample clinical data..."
python -c "
import pandas as pd
import numpy as np

# Create sample diabetes dataset
np.random.seed(42)
n_samples = 1000

# Generate correlated features
glucose = np.random.normal(110, 25, n_samples)
hba1c = 4.5 + glucose * 0.02 + np.random.normal(0, 0.5, n_samples)
bmi = np.random.normal(27, 5, n_samples)
age = np.random.uniform(20, 80, n_samples)

# Create diabetes outcome based on risk factors
diabetes_prob = 1 / (1 + np.exp(-(-5 + 
    0.03 * (glucose - 100) + 
    0.8 * (hba1c - 5.5) + 
    0.05 * (bmi - 25) + 
    0.02 * (age - 40))))
diabetes = (np.random.random(n_samples) < diabetes_prob).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'patient_id': range(n_samples),
    'glucose': glucose,
    'hba1c': hba1c,
    'bmi': bmi,
    'age': age,
    'diabetes': diabetes
})

# Save to CSV
df.to_csv('clinical_validation/data/sample_diabetes_data.csv', index=False)
print('Created sample dataset with', n_samples, 'patients')
print('Diabetes prevalence:', (diabetes.sum() / n_samples * 100), '%')
"
check_status "Sample data creation"

# 4. Test the clinical validation module
echo ""
echo "4️⃣ Testing clinical validation module..."
python clinical_validation/test_validation.py
check_status "Module testing"

# 5. Run full clinical validation if tests pass
echo ""
echo "5️⃣ Running full clinical validation..."
echo ""

# Create a runner script
cat > run_clinical_validation.py << 'EOF'
#!/usr/bin/env python3
"""
Run comprehensive clinical validation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from clinical_validation import ClinicalValidator
from clinical_validation.data_sources import PimaDataSource, NHANESDataSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("🧬 GenomeVault Clinical Validation")
    print("Using Privacy-Preserving Components")
    print("=" * 70)
    
    # Initialize validator
    validator = ClinicalValidator()
    
    # Add data sources
    print("\n📊 Loading clinical data sources...")
    
    # Try to add available data sources
    try:
        validator.add_data_source(PimaDataSource())
        print("✅ Added Pima Indians Diabetes dataset")
    except Exception as e:
        print(f"⚠️  Could not load Pima dataset: {e}")
    
    try:
        validator.add_data_source(NHANESDataSource())
        print("✅ Added NHANES dataset")
    except Exception as e:
        print(f"⚠️  Could not load NHANES dataset: {e}")
    
    # Also add our sample data as a custom source
    from clinical_validation.data_sources.base import BaseDataSource
    import pandas as pd
    
    class SampleDataSource(BaseDataSource):
        def __init__(self):
            super().__init__()
            self.name = "SampleDiabetesData"
            
        def load_data(self):
            try:
                return pd.read_csv('clinical_validation/data/sample_diabetes_data.csv')
            except:
                return None
                
        def get_glucose_column(self):
            return 'glucose'
            
        def get_hba1c_column(self):
            return 'hba1c'
            
        def get_outcome_column(self):
            return 'diabetes'
    
    validator.add_data_source(SampleDataSource())
    print("✅ Added sample diabetes dataset")
    
    # Run validation
    print("\n🔐 Running validation...")
    results = validator.run_full_clinical_validation()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print(f"Mode: {results.get('mode', 'Unknown')}")
    print(f"Components tested: {', '.join(results.get('components_tested', []))}")
    print(f"Data sources processed: {len(results.get('data_sources', {}))}")
    
    # Show performance metrics
    if results.get('zk_proof_metrics'):
        print("\n🔐 Zero-Knowledge Proof Performance:")
        for source, metrics in results['zk_proof_metrics'].items():
            print(f"\n  {source}:")
            print(f"    - Implementation: {'REAL' if metrics.get('using_real_zk') else 'Simulated'}")
            print(f"    - Proofs generated: {metrics.get('n_proofs_generated', 0)}")
            print(f"    - Avg generation time: {metrics.get('avg_generation_time_ms', 0):.1f} ms")
            print(f"    - Avg verification time: {metrics.get('avg_verification_time_ms', 0):.1f} ms")
    
    print("\n📄 Full report saved to: genomevault_clinical_validation_report.md")
    
    return results

if __name__ == "__main__":
    results = main()
EOF

python run_clinical_validation.py
check_status "Clinical validation"

# 6. Display report location
echo ""
echo "================================================================"
echo "✅ Clinical validation completed successfully!"
echo ""
echo "📄 Reports generated:"
echo "   - genomevault_clinical_validation_report.md"
echo "   - clinical_validation/data/sample_diabetes_data.csv"
echo ""
echo "🔍 To view the report:"
echo "   cat genomevault_clinical_validation_report.md"
echo ""
echo "📊 To run validation again:"
echo "   python run_clinical_validation.py"
echo "================================================================"
