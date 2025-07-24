#!/bin/bash
# Install missing linting tools if needed

echo "Checking and Installing Linting Tools"
echo "===================================="

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Check and install Black
if ! command_exists black; then
    echo "Installing Black..."
    pip install black
else
    echo "✓ Black is already installed"
fi

# Check and install isort
if ! command_exists isort; then
    echo "Installing isort..."
    pip install isort
else
    echo "✓ isort is already installed"
fi

# Check and install flake8
if ! command_exists flake8; then
    echo "Installing flake8..."
    pip install flake8
else
    echo "✓ flake8 is already installed"
fi

echo -e "\nAll linting tools are available!"
echo "You can now run ./fix_catalytic_linting.sh"
