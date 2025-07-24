#!/bin/bash
# Check PIR implementation status

echo "📊 PIR Implementation Status Check"
echo "================================="

cd /Users/rohanvinaik/genomevault

echo -e "\n🌿 Current branch:"
git branch --show-current

echo -e "\n📝 Last 3 commits:"
git log --oneline -3

echo -e "\n📁 Uncommitted changes:"
git status --short

echo -e "\n🔍 PIR files status:"
ls -la genomevault/pir/ | head -10

echo -e "\n✅ Linting status:"
echo -n "isort: "
if isort --check-only --profile black --line-length 100 genomevault/pir/it_pir_protocol.py &>/dev/null; then
    echo "PASS ✅"
else
    echo "NEEDS FIX ❌"
fi

echo -n "black: "
if black --check --line-length 100 genomevault/pir/it_pir_protocol.py &>/dev/null; then
    echo "PASS ✅"
else
    echo "NEEDS FIX ❌"
fi
