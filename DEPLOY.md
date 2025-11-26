# Deployment Guide

Quick guide to deploy the Multimodal Retrieval Demo on your VM with 2× GTX 1080.

## Prerequisites

- Ubuntu 20.04+ or similar Linux
- Docker & Docker Compose
- Python 3.10+
- NVIDIA drivers + CUDA 11.8+
- 2× GTX 1080 (8GB VRAM each)

## Quick Start (5 minutes)

```bash
# 1. Clone/copy the project
cd /path/to/retrieval-demo

# 2. Run setup
chmod +x setup.sh
./setup.sh

# 3. Download sample data
make download-sample      # Small image set
make download-esc50       # ESC-50 audio (~600MB)
make download-ikea        # IKEA manuals

# 4. Index the data
make index

# 5. Start the server
make run
```

Open http://your-server:8000 in your browser.

---

## Detailed Setup

### 1. System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3-pip python3-venv docker.io docker-compose
sudo apt install -y poppler-utils  # For PDF conversion
sudo apt install -y ffmpeg         # For audio processing

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 2. NVIDIA Setup (if not already done)

```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA toolkit (if needed)
# See: https://developer.nvidia.com/cuda-downloads
```

### 3. Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 4. Start Qdrant

```bash
docker-compose up -d qdrant

# Verify it's running
curl http://localhost:6333/health
```

### 5. Prepare Data

**Images:**
```bash
# Option A: Sample images
python scripts/download_data.py --dataset sample-images

# Option B: IKEA manuals (convert PDF pages to images)
python scripts/download_ikea_manuals.py

# Option C: Your own images
cp -r /path/to/your/images/* data/images/
```

**Audio:**
```bash
# Option A: ESC-50 dataset (recommended)
python scripts/download_audio.py --dataset esc50

# Option B: Your own audio
cp -r /path/to/your/audio/* data/audio/
```

### 6. Index Data

```bash
# Index images (uses GPU 0)
python scripts/index_images.py --device cuda:0

# Index audio (uses GPU 1)
python scripts/index_audio.py --device cuda:1
```

### 7. Start the Server

```bash
# Production
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Development (auto-reload)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## GPU Configuration

With 2× GTX 1080, the default config uses:
- **GPU 0**: CLIP model (images + text)
- **GPU 1**: CLAP model (audio + text)

To change this, edit `backend/config.py`:

```python
clip_device: str = "cuda:0"  # First GPU
clap_device: str = "cuda:1"  # Second GPU
```

If you only have 1 GPU, set both to `"cuda:0"` (models will share memory).

---

## Production Deployment

### Using systemd

Create `/etc/systemd/system/retrieval-demo.service`:

```ini
[Unit]
Description=Multimodal Retrieval Demo
After=network.target docker.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/retrieval-demo
Environment="PATH=/path/to/retrieval-demo/venv/bin"
ExecStart=/path/to/retrieval-demo/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable retrieval-demo
sudo systemctl start retrieval-demo
```

### Using nginx (reverse proxy)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend static files
    location / {
        root /path/to/retrieval-demo/frontend;
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static media
    location /images/ {
        alias /path/to/retrieval-demo/data/images/;
    }

    location /audio/ {
        alias /path/to/retrieval-demo/data/audio/;
    }
}
```

---

## Scaling for Multiple Users

The demo supports ~12 concurrent users with cached embeddings.

**Bottlenecks:**
1. Text encoding: ~50ms per query (fast)
2. Vector search: ~10ms per query (very fast)
3. Model loading: ~5-10s at startup (one-time)

**To scale further:**
- Add more GPU workers with `uvicorn --workers 2`
- Use Redis for caching frequent queries
- Deploy Qdrant in cluster mode

---

## Troubleshooting

### CUDA out of memory

```bash
# Check GPU memory usage
nvidia-smi

# Use smaller model (edit config.py)
clip_model: str = "openai/clip-vit-base-patch16"  # Smaller
```

### Qdrant connection refused

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart qdrant

# Check logs
docker-compose logs qdrant
```

### Audio not playing in browser

Ensure audio files are in a supported format (WAV, MP3, OGG).

```bash
# Convert to WAV if needed
ffmpeg -i input.flac -ar 48000 -ac 1 output.wav
```

### Model download fails

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Or use offline mode after first download
export TRANSFORMERS_OFFLINE=1
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/suggestions` | GET | Get search suggestions |
| `/api/images` | GET | List indexed images |
| `/api/audio` | GET | List indexed audio |
| `/api/search/text` | POST | Search images by text |
| `/api/search/image` | POST | Search images by image |
| `/api/search/multimodal` | POST | Search by text + image |
| `/api/search/audio/text` | POST | Search audio by text |
| `/api/search/audio/audio` | POST | Search audio by audio |

---

## Performance Tips

1. **Pre-warm models**: Hit `/api/health` after startup to ensure models are loaded
2. **Batch indexing**: Index data before presentation, not during
3. **Use SSD**: Qdrant and image/audio files benefit from fast storage
4. **Limit results**: Default is 50, reduce if needed for faster response
