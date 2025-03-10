#!/bin/bash
# Script to install the Rephysco package

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Create a virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv .venv
    
    # Activate the virtual environment
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
    else
        echo "Failed to create virtual environment. Continuing with system Python..."
    fi
fi

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Install the package in development mode
echo "Installing Rephysco..."
pip3 install -e .

echo "Installation complete!"
echo "You can now use Rephysco in your Python projects."
echo ""
echo "Example usage:"
echo "python -m rephysco generate --provider openai \"Hello, world!\""
echo ""
echo "For more information, see the README.md file." 