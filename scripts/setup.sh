#!/usr/bin/env zsh
# Setup script for SDF CoT Monitorability project

set -e

echo "=== Setting up SDF CoT Monitorability ==="

# Get project root
SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"
cd "$PROJECT_ROOT"

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.12"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) and sys.version_info < (3, 13) else 1)" 2>/dev/null; then
    echo "Warning: Python 3.12 required (found $PYTHON_VERSION)"
    echo "Note: Python 3.13 not yet supported due to torchaudio dependency"
fi

# Check for uv
if command -v uv &> /dev/null; then
    echo "Using UV package manager..."
    USE_UV=true
else
    echo "UV not found, using pip..."
    USE_UV=false
fi

# Install main package
echo ""
echo "Installing main package..."
if [ "$USE_UV" = true ]; then
    uv pip install -e .
else
    pip install -e .
fi

# Install external dependencies
echo ""
echo "Installing ImpossibleBench..."
if [ -d "external/impossiblebench" ]; then
    if [ "$USE_UV" = true ]; then
        uv pip install -e external/impossiblebench
    else
        pip install -e external/impossiblebench
    fi
else
    echo "Warning: external/impossiblebench not found"
fi

echo ""
echo "Installing false-facts..."
if [ -d "external/false-facts" ]; then
    if [ "$USE_UV" = true ]; then
        uv pip install -e external/false-facts
    else
        pip install -e external/false-facts
    fi
else
    echo "Warning: external/false-facts not found"
fi

# Install dev dependencies
read -p "Install dev dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ "$USE_UV" = true ]; then
        uv pip install -e ".[dev]" || uv pip install "poethepoet>=0.38.0" "pyright>=1.1.407" "pytest>=9.0.1" "ruff>=0.14.6"
    else
        pip install -e ".[dev]" || pip install "poethepoet>=0.38.0" "pyright>=1.1.407" "pytest>=9.0.1" "ruff>=0.14.6"
    fi
fi

# Setup .env
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env from template"
        echo "  Please edit .env with your API keys"
    fi
else
    echo "✓ .env already exists"
fi

# Check Docker
echo ""
echo "Checking Docker..."
if docker ps > /dev/null 2>&1; then
    echo "✓ Docker is running"
else
    echo "⚠ Docker is not running or not installed"
    echo "  ImpossibleBench requires Docker for sandboxed execution"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/*.sh
chmod +x scripts/*.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)"
echo "2. Start Docker if not already running"
echo "3. Run baseline experiment:"
echo "   ./scripts/run_baseline_experiment.sh"
echo ""
echo "For more information, see README.md"

