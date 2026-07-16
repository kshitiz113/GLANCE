"""
Embedding Engine — Dual-Backbone Feature Extraction (FashionCLIP + SigLIP-2)

Provides image and text embeddings from two complementary models:
1. FashionCLIP: CLIP fine-tuned on 800K fashion products — excels at
   fashion-specific vocabulary (fabric, cut, pattern, brand terminology)
2. SigLIP-2: Google's improved VLM with sigmoid loss — superior general
   vision-language alignment and better contextual understanding

Why two backbones instead of one?
- FashionCLIP knows "chiffon" and "midi skirt" but may not understand
  "sitting on a park bench" as well as SigLIP-2
- SigLIP-2 has excellent general vision-language alignment but lacks
  fashion domain specificity
- Combining both via score-level fusion captures both fashion nuance
  AND environmental/contextual understanding
"""

import gc
import os
import sys
import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Dual-backbone embedding engine for fashion image retrieval.

    Produces three types of embeddings per image:
    1. FashionCLIP image embedding (512-d) — fashion-domain visual features
    2. SigLIP-2 image embedding (768-d)   — general visual features
    3. SigLIP-2 text embedding (768-d)    — for encoding captions & queries

    All embeddings are L2-normalized for cosine similarity via inner product.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or config.DEVICE
        self._fashion_clip_model = None
        self._fashion_clip_processor = None
        self._siglip_model = None
        self._siglip_processor = None

    # ------------------------------------------------------------------
    # Lazy loading — only load models when first needed
    # ------------------------------------------------------------------

    def _unload_inactive_on_cpu(self, target: str):
        """On CPU, unload the other model before loading a new one to keep memory footprint light and prevent OS error 1455."""
        if self.device == "cpu":
            if target == "fashion" and self._siglip_model is not None:
                logger.info("Unloading SigLIP-2 from CPU memory before loading FashionCLIP")
                self._siglip_model = None
                self._siglip_processor = None
                gc.collect()
            elif target == "siglip" and self._fashion_clip_model is not None:
                logger.info("Unloading FashionCLIP from CPU memory before loading SigLIP-2")
                self._fashion_clip_model = None
                self._fashion_clip_processor = None
                gc.collect()

    def _load_fashion_clip(self):
        """Load FashionCLIP model."""
        if self._fashion_clip_model is not None:
            return

        self._unload_inactive_on_cpu("fashion")
        from transformers import CLIPModel, CLIPProcessor

        logger.info(f"Loading FashionCLIP: {config.FASHION_CLIP_MODEL}")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._fashion_clip_processor = CLIPProcessor.from_pretrained(config.FASHION_CLIP_MODEL)
        self._fashion_clip_model = CLIPModel.from_pretrained(
            config.FASHION_CLIP_MODEL, dtype=dtype, low_cpu_mem_usage=True
        )
        self._fashion_clip_model.to(self.device)
        self._fashion_clip_model.eval()
        logger.info("FashionCLIP loaded")

    def _load_siglip(self):
        """Load SigLIP-2 model."""
        if self._siglip_model is not None:
            return

        self._unload_inactive_on_cpu("siglip")
        from transformers import AutoModel, AutoProcessor

        logger.info(f"Loading SigLIP-2: {config.SIGLIP2_MODEL}")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._siglip_processor = AutoProcessor.from_pretrained(config.SIGLIP2_MODEL)
        self._siglip_model = AutoModel.from_pretrained(
            config.SIGLIP2_MODEL, dtype=dtype, low_cpu_mem_usage=True
        )
        self._siglip_model.to(self.device)
        self._siglip_model.eval()
        logger.info("SigLIP-2 loaded")

    # ------------------------------------------------------------------
    # FashionCLIP Embeddings
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tensor(features: Any) -> torch.Tensor:
        """Extract tensor from model output (handles transformers 5.x BaseModelOutputWithPooling)."""
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        elif hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            return features.last_hidden_state
        elif isinstance(features, (tuple, list)):
            return features[0]
        return features

    def get_fashion_clip_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get FashionCLIP image embedding (512-d, L2-normalized)."""
        self._load_fashion_clip()

        inputs = self._fashion_clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self._fashion_clip_model.dtype) if v.dtype == torch.float32 else v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._fashion_clip_model.get_image_features(**inputs)
            features = self._extract_tensor(features).float()

        features = F.normalize(features, dim=-1)
        res = features.cpu().numpy().flatten()
        if self.device == "cpu":
            self._fashion_clip_model = None
            self._fashion_clip_processor = None
            gc.collect()
        return res

    def get_fashion_clip_text_embedding(self, text: str) -> np.ndarray:
        """Get FashionCLIP text embedding (512-d, L2-normalized)."""
        self._load_fashion_clip()

        inputs = self._fashion_clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device, dtype=self._fashion_clip_model.dtype) if v.dtype == torch.float32 else v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._fashion_clip_model.get_text_features(**inputs)
            features = self._extract_tensor(features).float()

        features = F.normalize(features, dim=-1)
        res = features.cpu().numpy().flatten()
        if self.device == "cpu":
            self._fashion_clip_model = None
            self._fashion_clip_processor = None
            gc.collect()
        return res

    # ------------------------------------------------------------------
    # SigLIP-2 Embeddings
    # ------------------------------------------------------------------

    def get_siglip_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get SigLIP-2 image embedding (768-d, L2-normalized)."""
        self._load_siglip()

        inputs = self._siglip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self._siglip_model.dtype) if v.dtype == torch.float32 else v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._siglip_model.get_image_features(**inputs)
            features = self._extract_tensor(features).float()

        features = F.normalize(features, dim=-1)
        res = features.cpu().numpy().flatten()
        if self.device == "cpu":
            self._siglip_model = None
            self._siglip_processor = None
            gc.collect()
        return res

    def get_siglip_text_embedding(self, text: str) -> np.ndarray:
        """Get SigLIP-2 text embedding (768-d, L2-normalized)."""
        self._load_siglip()

        inputs = self._siglip_processor(text=text, return_tensors="pt", padding="max_length", truncation=True)
        inputs = {k: v.to(self.device, dtype=self._siglip_model.dtype) if v.dtype == torch.float32 else v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._siglip_model.get_text_features(**inputs)
            features = self._extract_tensor(features).float()

        features = F.normalize(features, dim=-1)
        res = features.cpu().numpy().flatten()
        if self.device == "cpu":
            self._siglip_model = None
            self._siglip_processor = None
            gc.collect()
        return res

    # ------------------------------------------------------------------
    # Batch Processing
    # ------------------------------------------------------------------

    def batch_encode_images(
        self,
        images: List[Image.Image],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch encode images with both backbones.

        Args:
            images: List of PIL Images
            batch_size: Processing batch size
            show_progress: Show tqdm progress bar

        Returns:
            Tuple of (fashion_clip_embeddings, siglip_embeddings)
            Each is ndarray of shape (N, dim)
        """
        fashion_embs = []
        siglip_embs = []

        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding images", total=len(images) // batch_size + 1)

        for start in iterator:
            batch = images[start:start + batch_size]

            for img in batch:
                fashion_embs.append(self.get_fashion_clip_image_embedding(img))
                siglip_embs.append(self.get_siglip_image_embedding(img))

        return np.array(fashion_embs), np.array(siglip_embs)

    def batch_encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Batch encode texts with SigLIP-2.

        Used for encoding BLIP-2 captions into the caption embedding index.

        Args:
            texts: List of text strings (captions)
            batch_size: Processing batch size
            show_progress: Show tqdm progress bar

        Returns:
            ndarray of shape (N, 768) — SigLIP-2 text embeddings
        """
        embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding texts", total=len(texts) // batch_size + 1)

        for start in iterator:
            batch = texts[start:start + batch_size]
            for text in batch:
                embeddings.append(self.get_siglip_text_embedding(text))

        return np.array(embeddings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = EmbeddingEngine()

    # Quick test
    print("Testing FashionCLIP text embedding...")
    text_emb = engine.get_fashion_clip_text_embedding("a red blazer")
    print(f"  Shape: {text_emb.shape}, Norm: {np.linalg.norm(text_emb):.4f}")

    print("Testing SigLIP-2 text embedding...")
    text_emb2 = engine.get_siglip_text_embedding("a red blazer in an office")
    print(f"  Shape: {text_emb2.shape}, Norm: {np.linalg.norm(text_emb2):.4f}")

    print("\nEmbedding engine ready.")
