#!/bin/bash
# Multimodal Retrieval Demo - Setup Script
# Run: chmod +x setup.sh && ./setup.sh

set -e

echo "=========================================="
echo "Multimodal Retrieval Demo - Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Check Docker
echo -e "\n${YELLOW}Checking Docker...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker found${NC}"
else
    echo -e "${RED}✗ Docker not found. Please install Docker${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate and install dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check CUDA
echo -e "\n${YELLOW}Checking CUDA...${NC}"
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo -e "${GREEN}✓ CUDA available with $GPU_COUNT GPU(s)${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not available - will use CPU (slow)${NC}"
fi

# Start Qdrant
echo -e "\n${YELLOW}Starting Qdrant vector database...${NC}"
docker-compose up -d qdrant
sleep 3

# Check Qdrant
if curl -s http://localhost:6333/health > /dev/null; then
    echo -e "${GREEN}✓ Qdrant is running on http://localhost:6333${NC}"
else
    echo -e "${RED}✗ Qdrant failed to start${NC}"
    exit 1
fi

# Create data directories
echo -e "\n${YELLOW}Creating data directories...${NC}"
mkdir -p data/images data/audio data/pdfs
echo -e "${GREEN}✓ Data directories created${NC}"

# Download sample data (optional)
echo -e "\n${YELLOW}Would you like to download sample data? (y/n)${NC}"
read -r DOWNLOAD_DATA

if [ "$DOWNLOAD_DATA" = "y" ]; then
    echo -e "\n${YELLOW}Downloading sample images...${NC}"
    python scripts/download_data.py --dataset sample-images
    
    echo -e "\n${YELLOW}Downloading ESC-50 audio dataset...${NC}"
    python scripts/download_audio.py --dataset esc50
    
    echo -e "\n${YELLOW}Downloading IKEA manuals...${NC}"
    python scripts/download_ikea_manuals.py --skip-convert
fi

echo -e "\n=========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo -e "=========================================="
echo -e "\nNext steps:"
echo -e "  1. Add images to data/images/"
echo -e "  2. Add audio to data/audio/"
echo -e "  3. Index data:"
echo -e "     ${YELLOW}python scripts/index_images.py${NC}"
echo -e "     ${YELLOW}python scripts/index_audio.py${NC}"
echo -e "  4. Start the server:"
echo -e "     ${YELLOW}uvicorn backend.main:app --host 0.0.0.0 --port 8080${NC}"
echo -e "  5. Open http://localhost:8080 in your browser"
echo -e "\nOr use: ${YELLOW}make run${NC}"
