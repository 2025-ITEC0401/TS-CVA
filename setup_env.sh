#!/bin/bash
# TS-CVA Environment Setup Script for Lab Server
# This script should be run on the lab server

set -e  # Exit on error

# Load environment variables if available
if [ -f "$HOME/TS-CVA/.env" ]; then
    export $(grep -v '^#' "$HOME/TS-CVA/.env" | xargs)
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ENV_NAME="${CONDA_ENV_NAME:-TS-CVA}"
PROJECT_DIR="${REMOTE_PROJECT_DIR:-$HOME/TS-CVA}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-/hdd/conda_envs/envs}"
PYTHON_PATH="${CONDA_ENVS_DIR}/${ENV_NAME}/bin/python3"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  TS-CVA Environment Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if we're in the correct directory
if [ ! -f "$PROJECT_DIR/env.yaml" ]; then
    echo -e "${RED}Error: env.yaml not found in $PROJECT_DIR${NC}"
    echo "Please make sure you're in the TimeCMA project directory."
    exit 1
fi

cd $PROJECT_DIR

# Step 1: Check if conda is available
echo -e "${BLUE}[1/6] Checking conda installation...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda is not installed or not in PATH${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Conda found${NC}"
echo ""

# Step 2: Deactivate current environment
echo -e "${BLUE}[2/6] Deactivating current conda environment...${NC}"
conda deactivate 2>/dev/null || true
echo -e "${GREEN}✓ Deactivated${NC}"
echo ""

# Step 3: Remove existing environment if it exists
echo -e "${BLUE}[3/6] Checking for existing environment...${NC}"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists. Removing...${NC}"
    conda env remove --name ${ENV_NAME} -y
    echo -e "${GREEN}✓ Removed existing environment${NC}"
else
    echo -e "${GREEN}✓ No existing environment found${NC}"
fi
echo ""

# Step 4: Create new environment from yaml
echo -e "${BLUE}[4/6] Creating conda environment from env.yaml...${NC}"
echo "This may take several minutes..."
conda env create -f env.yaml
echo -e "${GREEN}✓ Environment created${NC}"
echo ""

# Step 5: Verify installation
echo -e "${BLUE}[5/6] Verifying installation...${NC}"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Check Python version
PYTHON_VERSION=$(python --version)
echo "Python version: $PYTHON_VERSION"

# Check key packages
echo ""
echo "Checking key packages:"
python -c "
import sys
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'einops': 'Einops',
    'h5py': 'HDF5',
    'pandas': 'Pandas',
    'numpy': 'NumPy'
}

all_ok = True
for pkg, name in packages.items():
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {name}: {version}')
    except ImportError:
        print(f'  ✗ {name}: NOT INSTALLED')
        all_ok = False

sys.exit(0 if all_ok else 1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All key packages installed successfully${NC}"
else
    echo -e "${RED}✗ Some packages failed to install${NC}"
    exit 1
fi
echo ""

# Step 6: Display usage information
echo -e "${BLUE}[6/6] Setup complete!${NC}"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Installation Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Environment name: ${ENV_NAME}"
echo "Project directory: ${PROJECT_DIR}"
echo "Python path: ${PYTHON_PATH}"
echo ""
echo -e "${YELLOW}To activate the environment:${NC}"
echo "  conda activate ${ENV_NAME}"
echo ""
echo -e "${YELLOW}To run Python scripts:${NC}"
echo "  ${PYTHON_PATH} train.py --data_path ETTm1 ..."
echo ""
echo -e "${YELLOW}Or activate first, then run:${NC}"
echo "  conda activate ${ENV_NAME}"
echo "  python train.py --data_path ETTm1 ..."
echo ""
echo -e "${GREEN}✓ Setup completed successfully!${NC}"
