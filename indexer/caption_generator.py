"""
Caption Generator — BLIP-2 Fashion-Prompted Captioning

Generates rich, multi-aspect captions for fashion images using BLIP-2.
Uses a multi-prompt strategy to capture general appearance, fashion details,
and style inference in separate passes, then combines them into a single
enriched description per image.

Why BLIP-2 over CLIP for captioning?
- BLIP-2's Q-Former bridges frozen image encoders with LLMs, enabling
  detailed, free-form text generation (not just embedding similarity)
- Prompted captioning lets us steer output toward fashion-specific details
  that generic captions would miss (fabric, pattern, cut, style)
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class CaptionGenerator:
    """
    Generates rich captions for fashion images using BLIP-2.

    Multi-prompt strategy:
    1. General: Open-ended caption for overall scene understanding
    2. Fashion: Prompted for clothing items, colors, patterns, fabrics
    3. Style:   Prompted for style classification (formal, casual, etc.)

    The combined caption provides much richer semantic content than any
    single pass, enabling better downstream attribute extraction and
    embedding-based retrieval.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the BLIP-2 captioning model.

        Args:
            model_name: HuggingFace model identifier. Defaults to config.BLIP2_MODEL.
                        Use config.BLIP2_MODEL_SMALL for CPU/low-VRAM setups.
            device: Computation device. Auto-detected if None.
        """
        self.device = device or config.DEVICE
        self.model_name = model_name or (
            config.BLIP2_MODEL if self.device == "cuda" else config.BLIP2_MODEL_SMALL
        )
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self):
        """Lazy-load the captioning model."""
        if self._loaded:
            return

        logger.info(f"Loading captioning model: {self.model_name} on {self.device}")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if "blip2" in self.model_name.lower():
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            )
        else:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            )

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Captioning model loaded successfully")

    def generate_caption(
        self, image: Image.Image, prompt: str = "", max_new_tokens: int = None
    ) -> str:
        """
        Generate a single caption for an image.

        Args:
            image: PIL Image to caption.
            prompt: Optional text prompt to guide caption generation.
                    Empty string = open-ended captioning.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated caption string.
        """
        self.load_model()
        max_new_tokens = max_new_tokens or config.CAPTION_MAX_NEW_TOKENS

        # Process inputs
        if prompt:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        else:
            inputs = self.processor(images=image, return_tensors="pt")

        # Move to device with correct dtype
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        inputs = {k: v.to(self.device, dtype=dtype) if v.dtype == torch.float32 else v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                early_stopping=True,
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption

    def generate_multi_prompt_caption(self, image: Image.Image) -> Dict[str, str]:
        """
        Generate captions using all configured prompts.

        Returns dict with keys: 'general', 'fashion', 'style', 'combined'
        The 'combined' field concatenates all captions into a single rich description.
        """
        captions = {}
        for prompt_name, prompt_text in config.CAPTION_PROMPTS.items():
            captions[prompt_name] = self.generate_caption(image, prompt=prompt_text)

        # Combine all captions into a single enriched description
        parts = [v for v in captions.values() if v]
        captions["combined"] = " ".join(parts)

        return captions

    def process_directory(
        self,
        image_dir: str,
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
        resume: bool = True,
        image_files: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, str]]:
        """
        Process all images in a directory and generate captions.

        Args:
            image_dir: Path to directory containing images.
            output_path: Path to save captions JSON. Defaults to config.CAPTIONS_PATH.
            limit: Maximum number of images to process (for testing).
            resume: If True, skip already-captioned images.
            image_files: Optional list of filenames to process instead of scanning image_dir.

        Returns:
            Dictionary mapping image filenames to their caption dicts.
        """
        output_path = output_path or config.CAPTIONS_PATH
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load existing captions for resume support
        existing_captions = {}
        if resume and os.path.exists(output_path):
            with open(output_path, "r") as f:
                existing_captions = json.load(f)
            logger.info(f"Resuming: {len(existing_captions)} images already captioned")

        # Collect image files
        if image_files is None:
            image_files = sorted([
                f for f in os.listdir(image_dir)
                if os.path.splitext(f)[1].lower() in config.SUPPORTED_EXTENSIONS
            ])

        if limit:
            image_files = image_files[:limit]

        logger.info(f"Processing {len(image_files)} images from {image_dir}")

        captions = dict(existing_captions)  # Start with existing
        new_count = 0

        for filename in tqdm(image_files, desc="Generating captions"):
            if filename in captions:
                continue  # Skip already processed

            image_path = os.path.join(image_dir, filename)
            try:
                image = Image.open(image_path).convert("RGB")
                captions[filename] = self.generate_multi_prompt_caption(image)
                new_count += 1

                # Periodic save (every 50 images)
                if new_count % 50 == 0:
                    with open(output_path, "w") as f:
                        json.dump(captions, f, indent=2)
                    logger.info(f"Checkpoint: {new_count} new captions saved")

            except Exception as e:
                logger.warning(f"Failed to caption {filename}: {e}")
                captions[filename] = {
                    "general": "", "fashion": "", "style": "", "combined": "",
                    "error": str(e)
                }

        # Final save
        with open(output_path, "w") as f:
            json.dump(captions, f, indent=2)

        logger.info(f"Captioning complete: {new_count} new, {len(captions)} total")
        return captions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    generator = CaptionGenerator()

    # Quick test with a single image
    test_dir = config.IMAGE_DIR
    if not os.path.isdir(test_dir):
        # Fallback: look in data/test
        test_dir = os.path.join(config.DATA_DIR, "test")

    if os.path.isdir(test_dir):
        test_images = [f for f in os.listdir(test_dir)
                       if os.path.splitext(f)[1].lower() in config.SUPPORTED_EXTENSIONS][:2]
        for img_name in test_images:
            img = Image.open(os.path.join(test_dir, img_name)).convert("RGB")
            caps = generator.generate_multi_prompt_caption(img)
            print(f"\n--- {img_name} ---")
            for k, v in caps.items():
                print(f"  {k}: {v}")
