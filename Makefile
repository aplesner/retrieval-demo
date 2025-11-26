# Multimodal Retrieval Demo - Makefile
# Usage: make <target>

.PHONY: help setup install run dev index index-images index-audio download clean stop logs

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
	@echo "  index          - Index all data (images + audio)"
	@echo "  index-images   - Index images only"
	@echo "  index-audio    - Index audio only"
	@echo "  download-ikea  - Download IKEA manuals"
	@echo "  download-esc50 - Download ESC-50 audio dataset"
	@echo ""
	@echo "Infrastructure:"
	@echo "  qdrant-start   - Start Qdrant container"
	@echo "  qdrant-stop    - Stop Qdrant container"
	@echo "  logs           - Show Qdrant logs"
	@echo "  clean          - Remove generated files"

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
index: index-images index-audio

index-images:
	$(PYTHON_VENV) scripts/index_images.py --data-dir data/images

index-audio:
	$(PYTHON_VENV) scripts/index_audio.py --data-dir data/audio

# Downloads
download: download-esc50

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
