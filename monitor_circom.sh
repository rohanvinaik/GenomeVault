#!/bin/bash

# Monitor Circom installation progress
echo "🔍 Monitoring Circom installation..."
echo "Started at: $(date)"
echo "=" | head -c 60 | tr " " "="
echo ""

check_interval=30  # Check every 30 seconds
elapsed=0

while true; do
    # Check if circom binary exists
    if [ -f ~/circom/target/release/circom ]; then
        echo ""
        echo "✅ CIRCOM COMPILATION COMPLETE!"
        echo "Time taken: $((elapsed/60)) minutes"
        echo ""

        # Test the binary
        echo "Testing Circom binary..."
        ~/circom/target/release/circom --version

        # Create symlink if not exists
        if [ ! -L /usr/local/bin/circom ]; then
            echo "Creating symlink..."
            sudo ln -sf ~/circom/target/release/circom /usr/local/bin/circom 2>/dev/null || \
                ln -sf ~/circom/target/release/circom ~/.local/bin/circom
        fi

        # Compile test circuit
        echo ""
        echo "📝 Compiling test circuits..."
        cd /Users/rohanvinaik/genomevault

        if [ -f /usr/local/bin/circom ] || [ -f ~/.local/bin/circom ]; then
            # Create simple test circuit
            mkdir -p zk_circuits
            cat > zk_circuits/test.circom << 'EOF'
pragma circom 2.0.0;

template Multiplier() {
    signal input a;
    signal input b;
    signal output c;

    c <== a * b;
}

component main = Multiplier();
EOF

            # Compile it
            circom zk_circuits/test.circom --r1cs --wasm --sym -o zk_circuits/

            if [ $? -eq 0 ]; then
                echo "✅ Test circuit compiled successfully!"
                echo ""
                echo "🎉 CIRCOM IS FULLY OPERATIONAL!"

                # Show what was created
                echo ""
                echo "Generated files:"
                ls -la zk_circuits/test* 2>/dev/null | head -5
            else
                echo "⚠️ Circuit compilation failed - may need to check installation"
            fi
        fi

        # Send notification (macOS)
        if command -v osascript &> /dev/null; then
            osascript -e 'display notification "Circom compilation complete!" with title "GenomeVault"'
        fi

        echo ""
        echo "=" | head -c 60 | tr " " "="
        echo "Completed at: $(date)"
        break
    fi

    # Check if build is still running
    if pgrep -f "cargo.*circom" > /dev/null 2>&1; then
        echo -ne "\r⏳ Still building... ($((elapsed/60))m $((elapsed%60))s elapsed)"
    else
        # Check if cargo is installed and attempt to restart if needed
        if command -v cargo &> /dev/null; then
            if [ ! -d ~/circom ]; then
                echo ""
                echo "📦 Circom directory not found. Starting installation..."
                cd ~
                git clone https://github.com/iden3/circom.git
                cd circom
                cargo build --release &
                echo "Build restarted in background"
            elif [ -d ~/circom ] && [ ! -f ~/circom/target/release/circom ]; then
                echo -ne "\r🔧 Build not running. Checking state... ($((elapsed/60))m elapsed)"

                # Check if we can restart
                cd ~/circom
                if [ -f Cargo.toml ]; then
                    # Check last modification time of target directory
                    if [ -d target ]; then
                        last_mod=$(find target -type f -name "*.rs" -exec stat -f "%m" {} \; 2>/dev/null | sort -n | tail -1)
                        current_time=$(date +%s)
                        time_diff=$((current_time - last_mod))

                        if [ $time_diff -gt 300 ]; then  # No activity for 5 minutes
                            echo ""
                            echo "🔄 Restarting build (no activity detected)..."
                            cargo build --release > /tmp/circom_build.log 2>&1 &
                            echo "Build restarted. Check /tmp/circom_build.log for details"
                        fi
                    else
                        echo ""
                        echo "🚀 Starting fresh build..."
                        cargo build --release > /tmp/circom_build.log 2>&1 &
                        echo "Build started. Check /tmp/circom_build.log for details"
                    fi
                fi
            fi
        else
            echo ""
            echo "❌ Cargo not found. Please install Rust first:"
            echo "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
            exit 1
        fi
    fi

    sleep $check_interval
    elapsed=$((elapsed + check_interval))

    # Give status update every 5 minutes
    if [ $((elapsed % 300)) -eq 0 ]; then
        echo ""
        echo "📊 Status update: $((elapsed/60)) minutes elapsed"

        # Check build log if exists
        if [ -f /tmp/circom_build.log ]; then
            echo "Last build output:"
            tail -3 /tmp/circom_build.log | sed 's/^/  /'
        fi

        # Check disk usage
        if [ -d ~/circom/target ]; then
            size=$(du -sh ~/circom/target 2>/dev/null | cut -f1)
            echo "Build directory size: $size"
        fi
    fi
done
