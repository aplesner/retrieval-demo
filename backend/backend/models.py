"""CLIP and CLAP model wrappers for encoding text, images, and audio."""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from .config import settings


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
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text queries to embeddings."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        embeddings = self.model.get_text_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu()
    
    @torch.no_grad()
    def encode_image(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode images to embeddings."""
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        embeddings = self.model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu()
    
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
        self.model = ClapModel.from_pretrained(model_name).to(device)
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
        
        return embeddings.cpu()
    
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
            audios=resampled,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        embeddings = self.model.get_audio_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu()
    
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
# Singleton instances
# ============================================================================

_clip_encoder: CLIPEncoder | None = None
_clap_encoder: CLAPEncoder | None = None


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


# Backwards compatibility
def get_encoder() -> CLIPEncoder:
    """Alias for get_clip_encoder()."""
    return get_clip_encoder()
