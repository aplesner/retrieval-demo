# Multimodal Retrieval Demo - Makefile
# Usage: make <target>

.PHONY: help setup install run dev index index-images index-audio download clean stop logs docker-build docker-up docker-down docker-restart docker-logs docker-clean docker-up-prod docker-build-cpu docker-up-cpu docker-up-prod-cpu

# Default target
help:
	@echo "Multimodal Retrieval Demo"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Full setup (venv, deps, Qdrant)"
	@echo "  install        - Install Python dependencies"
	@echo "  download       - Download sample datasets"
	@echo ""
	@echo "Running:"
	@echo "  run            - Start the backend server"
	@echo "  dev            - Start in development mode (auto-reload)"
	@echo "  frontend       - Serve frontend (separate terminal)"
	@echo ""
	@echo "Data:"
	@echo "  index          - Index all data (images + audio + PDFs)"
	@echo "  index-images   - Index images only"
	@echo "  index-audio    - Index audio only"
	@echo "  index-pdfs     - Index PDFs only"
	@echo "  download-ikea  - Download IKEA manuals"
	@echo "  download-esc50 - Download ESC-50 audio dataset"
	@echo ""
	@echo "Infrastructure:"
	@echo "  qdrant-start   - Start Qdrant container"
	@echo "  qdrant-stop    - Stop Qdrant container"
	@echo "  logs           - Show Qdrant logs"
	@echo "  clean          - Remove generated files"
	@echo ""
	@echo "Docker (GPU - Recommended if available):"
	@echo "  docker-build       - Build Docker images (GPU)"
	@echo "  docker-up          - Start all services (GPU)"
	@echo "  docker-up-prod     - Start with nginx reverse proxy (GPU)"
	@echo ""
	@echo "Docker (CPU - Use when GPU unavailable):"
	@echo "  docker-build-cpu   - Build Docker images (CPU-only)"
	@echo "  docker-up-cpu      - Start all services (CPU-only)"
	@echo "  docker-up-prod-cpu - Start with nginx reverse proxy (CPU-only)"
	@echo ""
	@echo "Docker (Common):"
	@echo "  docker-down        - Stop all services"
	@echo "  docker-restart     - Restart all services"
	@echo "  docker-logs        - Show logs from all services"
	@echo "  docker-clean       - Stop and remove all containers/volumes"

# Python interpreter
PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python

# Setup
setup: $(VENV) install qdrant-start
	@echo "Setup complete!"

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt

# Running
run: qdrant-start
	$(PYTHON_VENV) -m uvicorn backend.main:app --host 0.0.0.0 --port 8080

dev: qdrant-start
	$(PYTHON_VENV) -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload

frontend:
	cd frontend && $(PYTHON) -m http.server 3000

# Indexing
index: index-images index-audio index-pdfs

index-images:
	$(PYTHON_VENV) scripts/index_images.py --data-dir data/images

index-audio:
	$(PYTHON_VENV) scripts/index_audio.py --data-dir data/audio

index-pdfs:
	$(PYTHON_VENV) scripts/index_pdfs.py --data-dir data/pdfs

count-database:
	$(PYTHON_VENV) scripts/count_database.py

# Downloads
download: download-coco download-ikea download-esc50  download-sample

download-coco:
	$(PYTHON_VENV) scripts/download_data.py --dataset coco-val2017

download-ikea:
	$(PYTHON_VENV) scripts/download_ikea_manuals.py

download-esc50:
	$(PYTHON_VENV) scripts/download_audio.py --dataset esc50

download-sample:
	$(PYTHON_VENV) scripts/download_data.py --dataset sample-images

# Infrastructure
qdrant-start:
	docker-compose up -d qdrant
	@echo "Waiting for Qdrant..."
	@sleep 3
	@curl -s http://localhost:6333/health > /dev/null && echo "Qdrant is ready" || echo "Qdrant failed to start"

qdrant-stop:
	docker-compose down

logs:
	docker-compose logs -f qdrant

# Cleanup
clean:
	rm -rf __pycache__ backend/__pycache__ scripts/__pycache__
	rm -rf .pytest_cache .mypy_cache
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

clean-all: clean
	rm -rf $(VENV)
	docker-compose down -v

# Docker targets
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be healthy (this may take 1-2 minutes on first run)..."
	@timeout 180 sh -c 'until curl -sf http://localhost:8080/api/health > /dev/null 2>&1; do sleep 3; done' && echo "✓ Services are ready! Access at http://localhost:8080" || echo "✗ Services failed to start after 3 minutes"

docker-down:
	docker-compose down

docker-restart:
	docker-compose restart

docker-logs:
	docker-compose logs -f --tail 100

docker-clean:
	docker-compose down -v
	docker system prune -f

docker-up-prod:
	docker-compose --profile production up -d
	@echo "Services started with nginx reverse proxy"
	@echo "Access at http://localhost (port 80)"

# CPU-only Docker targets (for systems without GPU/nvidia-docker)
docker-build-cpu:
	docker-compose -f docker-compose.cpu.yml build

docker-up-cpu:
	docker-compose -f docker-compose.cpu.yml up -d
	@echo "Waiting for services to be healthy (CPU mode - may be slower on first run)..."
	@timeout 180 sh -c 'until curl -sf http://localhost:8080/api/health > /dev/null 2>&1; do sleep 3; done' && echo "✓ Services are ready! Access at http://localhost:8080" || echo "✗ Services failed to start after 3 minutes"

docker-up-prod-cpu:
	docker-compose -f docker-compose.cpu.yml --profile production up -d
	@echo "Services started with nginx reverse proxy (CPU mode)"
	@echo "Access at http://localhost (port 80)"
