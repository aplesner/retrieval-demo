# Multimodal Retrieval Demo

A demonstration website for multimodal search using CLIP/CLAP embeddings and vector databases.

## Features

- **Text → Image**: Search images using natural language
- **Image → Image**: Find visually similar images
- **Text + Image → Image**: Combine text and image queries
- **Text → Audio**: Search audio clips using natural language

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Static)                        │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐   │
│  │Text→Image │ │Image→Image│ │Text+Image │ │  Text→Audio   │   │
│  └───────────┘ └───────────┘ └───────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                              │
│  ┌────────────────────────┐    ┌────────────────────────────┐   │
│  │ CLIP Model (GPU 0)     │    │ CLAP Model (GPU 1)         │   │
│  │ - Image encoder        │    │ - Audio encoder            │   │
│  │ - Text encoder         │    │ - Text encoder             │   │
│  └────────────────────────┘    └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Qdrant Vector Database                       │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │ Collection: images      │  │ Collection: audio           │   │
│  │ 512-dim CLIP embeddings │  │ 512-dim CLAP embeddings     │   │
│  └─────────────────────────┘  └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Models

### Image/Text (CLIP)

| Model | Dim | Speed | Quality | VRAM |
|-------|-----|-------|---------|------|
| `openai/clip-vit-base-patch32` | 512 | Fast | Good | ~1.5GB |
| `openai/clip-vit-large-patch14` | 768 | Medium | Better | ~3GB |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | 1024 | Slow | Best | ~6GB |

### Audio/Text (CLAP)

| Model | Dim | Speed | Quality | VRAM |
|-------|-----|-------|---------|------|
| `laion/clap-htsat-unfused` | 512 | Medium | Good | ~2GB |
| `laion/larger_clap_music` | 512 | Medium | Best for music | ~2GB |

**Default**: `clip-vit-base-patch32` + `clap-htsat-unfused` (fits on 2× GTX 1080)

## Datasets

### Images

| Dataset | Size | Description |
|---------|------|-------------|
| COCO 2017 val | 5K images | General objects |
| IKEA Manuals | ~200 pages | Assembly instructions |
| Custom | Variable | Your own images |

### Audio

| Dataset | Size | Description |
|---------|------|-------------|
| **ESC-50** | 2,000 clips | Environmental sounds, 50 classes |
| FSD50K | 51K clips | Freesound, diverse sounds |
| AudioCaps | 50K clips | AudioSet with captions |

**Recommended**: ESC-50 (small, diverse, easy to download)

## Quick Start

```bash
# 1. Setup
chmod +x setup.sh && ./setup.sh

# 2. Download data
make download-sample    # Images
make download-esc50     # Audio (ESC-50)
make download-ikea      # IKEA manuals

# 3. Index data
make index

# 4. Run
make run
```

Open http://localhost:8000

## Project Structure

```
retrieval-demo/
├── backend/
│   ├── main.py           # FastAPI app
│   ├── models.py         # CLIP + CLAP encoders
│   ├── database.py       # Qdrant client
│   ├── config.py         # Configuration
│   └── requirements.txt
├── frontend/
│   ├── index.html        # Text → Image
│   ├── image-search.html # Image → Image
│   ├── multimodal.html   # Text + Image
│   ├── audio-search.html # Text → Audio
│   └── static/
├── scripts/
│   ├── index_images.py   # Image indexing
│   ├── index_audio.py    # Audio indexing
│   ├── download_audio.py # ESC-50 downloader
│   ├── download_ikea_manuals.py
│   └── download_data.py
├── data/
│   ├── images/
│   └── audio/
├── docker-compose.yml
├── Makefile
├── setup.sh
├── DEPLOY.md             # Deployment guide
└── README.md
```

## Hardware Requirements

- **Minimum**: 1× GPU with 8GB VRAM (shared CLIP+CLAP)
- **Recommended**: 2× GTX 1080 (8GB each)
  - GPU 0: CLIP (images)
  - GPU 1: CLAP (audio)
- **RAM**: 16GB+
- **Storage**: 10GB+ for models and data

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/suggestions` | GET | Search suggestions |
| `/api/images` | GET | List images |
| `/api/audio` | GET | List audio |
| `/api/search/text` | POST | Text → Images |
| `/api/search/image` | POST | Image → Images |
| `/api/search/multimodal` | POST | Text+Image → Images |
| `/api/search/audio/text` | POST | Text → Audio |
| `/api/search/audio/audio` | POST | Audio → Audio |

## Configuration

Edit `backend/config.py` or set environment variables:

```bash
export RETRIEVAL_CLIP_DEVICE=cuda:0
export RETRIEVAL_CLAP_DEVICE=cuda:1
export RETRIEVAL_QDRANT_HOST=localhost
```

## License

MIT
