#!/bin/bash
# Setup script for automated traffic monitoring

echo "🔧 GenomeVault Traffic Monitoring Setup"
echo "========================================"
echo ""

# Get the absolute path to the genomevault directory
GENOMEVAULT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCRIPT_PATH="$GENOMEVAULT_DIR/scripts/monitor_github_traffic.sh"

echo "GenomeVault directory: $GENOMEVAULT_DIR"
echo "Monitoring script: $SCRIPT_PATH"
echo ""

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Error: monitor_github_traffic.sh not found at $SCRIPT_PATH"
    exit 1
fi

# Make script executable
chmod +x "$SCRIPT_PATH"
echo "✅ Made monitoring script executable"
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "⚠️  GitHub CLI (gh) is not installed"
    echo "   Install it with: brew install gh"
    echo "   Then run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI found"
echo ""

# Test the script
echo "🧪 Testing monitoring script..."
cd "$GENOMEVAULT_DIR"
"$SCRIPT_PATH"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test successful!"
    echo ""
else
    echo ""
    echo "❌ Test failed. Please check the script."
    exit 1
fi

# Create cron job
CRON_COMMAND="0 2 * * * cd $GENOMEVAULT_DIR && $SCRIPT_PATH >> $GENOMEVAULT_DIR/security_analysis/traffic_monitor.log 2>&1"

echo "📋 Cron job to be installed:"
echo "   $CRON_COMMAND"
echo ""
echo "This will run daily at 2:00 AM"
echo ""

read -p "Install cron job? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if cron job already exists
    if crontab -l 2>/dev/null | grep -q "monitor_github_traffic.sh"; then
        echo "⚠️  Cron job already exists. Skipping..."
    else
        # Add cron job
        (crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -
        echo "✅ Cron job installed!"
        echo ""
        echo "Current crontab:"
        crontab -l | grep monitor_github_traffic.sh
    fi
else
    echo "ℹ️  Skipped cron installation"
    echo ""
    echo "To install manually later, run:"
    echo "  crontab -e"
    echo "  # Add this line:"
    echo "  $CRON_COMMAND"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📊 Traffic data will be saved to:"
echo "   $GENOMEVAULT_DIR/security_analysis/traffic_history/"
echo ""
echo "🔍 View current traffic data:"
echo "   ls -la $GENOMEVAULT_DIR/security_analysis/traffic_history/"
echo ""
echo "📝 Monitor logs:"
echo "   tail -f $GENOMEVAULT_DIR/security_analysis/traffic_monitor.log"
echo ""
