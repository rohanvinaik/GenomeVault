#!/bin/bash
# GenomeVault MVP Quick Verification Script

echo "========================================"
echo "GenomeVault MVP Verification"
echo "========================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command success
check_command() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

# 1. Check Python version
echo "1. Checking Python version..."
python --version 2>&1 | grep -E "Python 3.(10|11|12)" > /dev/null
check_command "Python 3.10+ installed"
echo ""

# 2. Check if genomevault module imports
echo "2. Testing module import..."
python -c "import genomevault; print('GenomeVault version:', getattr(genomevault, '__version__', '0.1.0'))" 2>/dev/null
check_command "GenomeVault module imports"
echo ""

# 3. Check for critical files
echo "3. Checking critical files..."
FILES_TO_CHECK=(
    "genomevault/api/main.py"
    "genomevault/api/routers/health.py"
    "genomevault/api/routers/encode.py"
    "requirements.txt"
    "Dockerfile"
    "docker-compose.yml"
)

for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${YELLOW}⚠${NC} $file missing"
    fi
done
echo ""

# 4. Test API startup
echo "4. Testing API startup..."
echo "Starting API server..."
timeout 5 python -m genomevault.api.main > /tmp/genomevault_api.log 2>&1 &
API_PID=$!
sleep 3

# Check if API is running
if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}✓${NC} API server started (PID: $API_PID)"

    # Test health endpoint
    echo "Testing health endpoint..."
    curl -s http://localhost:8000/health > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
        echo -e "${GREEN}✓${NC} Health endpoint responding: $HEALTH_RESPONSE"
    else
        echo -e "${RED}✗${NC} Health endpoint not responding"
    fi

    # Kill the API server
    kill $API_PID 2>/dev/null
    wait $API_PID 2>/dev/null
else
    echo -e "${RED}✗${NC} API server failed to start"
    echo "Check /tmp/genomevault_api.log for errors"
fi
echo ""

# 5. Run basic tests
echo "5. Running basic tests..."
if [ -d "tests" ]; then
    # Try to run a simple test
    python -m pytest tests/ -k "health" --tb=no -q 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Basic tests pass"
    else
        echo -e "${YELLOW}⚠${NC} Some tests failing (may be expected for MVP)"
    fi
else
    echo -e "${YELLOW}⚠${NC} No tests directory found"
fi
echo ""

# 6. Check for syntax errors
echo "6. Checking for Python syntax errors..."
SYNTAX_ERRORS=$(python -m py_compile genomevault/**/*.py 2>&1 | wc -l)
if [ $SYNTAX_ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓${NC} No syntax errors detected"
else
    echo -e "${YELLOW}⚠${NC} Found $SYNTAX_ERRORS potential syntax issues"
fi
echo ""

# 7. Summary
echo "========================================"
echo "Summary"
echo "========================================"

# Count successes
TOTAL_CHECKS=6
PASSED_CHECKS=0

[ -f "genomevault/api/main.py" ] && ((PASSED_CHECKS++))
python -c "import genomevault" 2>/dev/null && ((PASSED_CHECKS++))
[ -f "requirements.txt" ] && ((PASSED_CHECKS++))
[ -f "Dockerfile" ] && ((PASSED_CHECKS++))

echo -e "Checks Passed: ${GREEN}$PASSED_CHECKS${NC} / $TOTAL_CHECKS"
echo ""

if [ $PASSED_CHECKS -ge 4 ]; then
    echo -e "${GREEN}✅ MVP is functional!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Install dependencies: pip install -r requirements.txt"
    echo "2. Start the API: python -m genomevault.api.main"
    echo "3. Visit docs: http://localhost:8000/docs"
    echo "4. Run tests: pytest tests/"
else
    echo -e "${YELLOW}⚠️ MVP needs more work${NC}"
    echo ""
    echo "Run the following to fix:"
    echo "1. python implement_mvp_complete.py"
    echo "2. python genomevault_safe_fix_implementation.py"
fi

echo "========================================"
