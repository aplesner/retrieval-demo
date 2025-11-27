"""CLIP, CLAP, and ColPali model wrappers for encoding text, images, audio, and PDFs."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging

from .config import settings

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_optimal_dtype(device: str) -> torch.dtype:
    """Get optimal dtype for the given device.
    
    Args:
        device: Device string (e.g., 'cuda:0', 'cuda:1', 'cpu')
        
    Returns:
        torch.float32 for CPU, torch.float16 for GTX 1080 and similar,
        torch.bfloat16 for newer GPUs (compute capability 8.0+)
    """
    if device == "cpu":
        return torch.float32
    
    # Extract GPU index
    gpu_idx = int(device.split(":")[-1]) if ":" in device else 0
    
    # Check compute capability
    if not torch.cuda.is_available():
        return torch.float32
        
    compute_capability = torch.cuda.get_device_capability(gpu_idx)
    if compute_capability[0] >= 8:
        return torch.bfloat16
    else:
        return torch.float16


# ============================================================================
# CLIP Encoder (Text + Images)
# ============================================================================

class CLIPEncoder:
    """Wrapper for CLIP model to encode text and images."""
    
    def __init__(
        self,
        model_name: str = settings.clip_model,
        device: str = settings.clip_device,
    ):
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = device
        self.dtype = get_optimal_dtype(device)
        
        logger.info(f"Loading CLIP model on {device} with dtype {self.dtype}")
        
        self.model = CLIPModel.from_pretrained(
            model_name,
            dtype=self.dtype,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text queries to embeddings."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        embeddings = self.model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.float().cpu()  # Return as float32 for compatibility
    
    @torch.no_grad()
    def encode_image(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode images to embeddings."""
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype.is_floating_point else v.to(self.device) 
                  for k, v in inputs.items()}
        
        embeddings = self.model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.float().cpu()  # Return as float32 for compatibility
    
    @torch.no_grad()
    def encode_image_from_path(self, image_paths: list[Path]) -> torch.Tensor:
        """Encode images from file paths."""
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self.encode_image(images)


# ============================================================================
# CLAP Encoder (Text + Audio)
# ============================================================================

class CLAPEncoder:
    """Wrapper for CLAP model to encode text and audio."""
    
    def __init__(
        self,
        model_name: str = settings.clap_model,
        device: str = settings.clap_device,
    ):
        from transformers import ClapModel, ClapProcessor
        
        self.device = device
        self.dtype = get_optimal_dtype(device)
        
        logger.info(f"Loading CLAP model on {device} with dtype {self.dtype}")
        
        self.model = ClapModel.from_pretrained(
            model_name,
            dtype=self.dtype,
        ).to(device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model.eval()
        self.sample_rate = 48000  # CLAP expects 48kHz
    
    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text queries to audio-aligned embeddings."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        embeddings = self.model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.float().cpu()  # Return as float32 for compatibility
    
    @torch.no_grad()
    def encode_audio(self, audio_arrays: list[np.ndarray], sample_rates: list[int]) -> torch.Tensor:
        """Encode audio waveforms to embeddings.
        
        Args:
            audio_arrays: List of audio waveforms as numpy arrays.
            sample_rates: Sample rate for each audio (will be resampled to 48kHz).
        """
        import librosa
        
        # Resample all audio to 48kHz
        resampled = []
        for audio, sr in zip(audio_arrays, sample_rates):
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            resampled.append(audio)
        
        inputs = self.processor(
            audio=resampled,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        # Move to device and convert floating point tensors to correct dtype
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype.is_floating_point else v.to(self.device) 
                  for k, v in inputs.items()}
        
        embeddings = self.model.get_audio_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.float().cpu()  # Return as float32 for compatibility
    
    @torch.no_grad()
    def encode_audio_from_path(self, audio_paths: list[Path]) -> torch.Tensor:
        """Encode audio files from paths."""
        import librosa
        
        audio_arrays = []
        sample_rates = []
        
        for path in audio_paths:
            audio, sr = librosa.load(path, sr=None, mono=True)
            audio_arrays.append(audio)
            sample_rates.append(sr)
        
        return self.encode_audio(audio_arrays, sample_rates)


# ============================================================================
# ColPali Encoder (Text + PDF Pages)
# ============================================================================

class ColPaliEncoder:
    """Wrapper for ColPali model to encode text queries and PDF pages."""

    def __init__(
        self,
        model_name: str = settings.colpali_model,
        device: str = settings.colpali_device,
    ):
        from colpali_engine.models import ColPali, ColPaliProcessor

        self.device = device
        self.dtype = get_optimal_dtype(device)

        logger.info(f"Loading ColPali model on {device} with dtype {self.dtype}")

        self.model = ColPali.from_pretrained(
            model_name,
            dtype=self.dtype,
            device_map=device,
        ).to(device).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text queries to multi-vector embeddings.

        Returns:
            Tensor of shape [batch_size, num_query_tokens, embedding_dim]
        """
        batch_queries = self.processor.process_queries(texts).to(self.device)
        embeddings = self.model(**batch_queries)
        return embeddings.cpu()

    @torch.no_grad()
    def encode_pdf_page(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode PDF page images to multi-vector embeddings.

        Args:
            images: List of PIL images (PDF pages converted to images)

        Returns:
            Tensor of shape [batch_size, num_page_tokens, embedding_dim]
        """
        batch_images = self.processor.process_images(images).to(self.device)
        embeddings = self.model(**batch_images)
        return embeddings

    @torch.no_grad()
    def encode_pdf_from_path(self, pdf_path: Path, batch_size: int = 8) -> tuple[torch.Tensor, list[Image.Image]]:
        """Encode PDF file by converting pages to images and encoding them.

        Args:
            pdf_path: Path to PDF file
            batch_size: Number of pages to process at once (default: 4 to avoid OOM)

        Returns:
            Tuple of (embeddings, page_images)
            - embeddings: Tensor [num_pages, num_tokens, embedding_dim]
            - page_images: List of PIL images for each page
        """
        import fitz  # PyMuPDF

        # Convert PDF pages to images
        doc = fitz.open(pdf_path)
        page_images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render at 2x resolution for better quality
            # mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap() # matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_images.append(img)

        doc.close()

        # Encode pages in batches to avoid OOM
        all_embeddings = []
        logger.info(f"Encoding {len(page_images)} pages from {pdf_path.name} in batches of {batch_size}...")
        with torch.no_grad():
            for i in range(0, len(page_images), batch_size):
                batch = page_images[i:i + batch_size]
                batch_embeddings = self.encode_pdf_page(batch)
                all_embeddings.append(batch_embeddings)
            # Clear CUDA cache between batches if using GPU
            # if self.device.startswith("cuda"):
            #     torch.cuda.empty_cache()
        logger.info(f"Finished encoding pages from {pdf_path.name}.")

        # Concatenate all batch embeddings and move to CPU
        embeddings = torch.cat(all_embeddings, dim=0).cpu()

        return embeddings, page_images


# ============================================================================
# Singleton instances
# ============================================================================

_clip_encoder: CLIPEncoder | None = None
_clap_encoder: CLAPEncoder | None = None
_colpali_encoder: ColPaliEncoder | None = None


def get_clip_encoder() -> CLIPEncoder:
    """Get or create the CLIP encoder singleton."""
    global _clip_encoder
    if _clip_encoder is None:
        _clip_encoder = CLIPEncoder()
    return _clip_encoder


def get_clap_encoder() -> CLAPEncoder:
    """Get or create the CLAP encoder singleton."""
    global _clap_encoder
    if _clap_encoder is None:
        _clap_encoder = CLAPEncoder()
    return _clap_encoder


def get_colpali_encoder() -> ColPaliEncoder:
    """Get or create the ColPali encoder singleton."""
    global _colpali_encoder
    if _colpali_encoder is None:
        _colpali_encoder = ColPaliEncoder()
    return _colpali_encoder


# Backwards compatibility
def get_encoder() -> CLIPEncoder:
    """Alias for get_clip_encoder()."""
    return get_clip_encoder()
