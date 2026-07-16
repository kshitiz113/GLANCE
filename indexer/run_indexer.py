"""
Indexer Orchestrator — Full Indexing Pipeline

Orchestrates the complete Part A pipeline:
1. Load images from data directory
2. Generate BLIP-2 captions (multi-prompt)
3. Extract structured attributes with bindings
4. Compute dual-backbone embeddings (FashionCLIP + SigLIP-2)
5. Build triple FAISS index + save metadata

Supports checkpointing at each stage so you can resume after interruptions.

Usage:
    python indexer/run_indexer.py                    # Full pipeline
    python indexer/run_indexer.py --limit 50         # Process first 50 images
    python indexer/run_indexer.py --skip-captions     # Skip captioning (use existing)
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indexer.caption_generator import CaptionGenerator
from indexer.attribute_extractor import AttributeExtractor
from indexer.embedding_engine import EmbeddingEngine
from indexer.vector_store import FashionVectorStore

logger = logging.getLogger(__name__)


def discover_images(image_dir: str, limit: Optional[int] = None) -> List[str]:
    """Find all supported image files in directory."""
    # Check multiple possible directories
    if not os.path.isdir(image_dir):
        # Try data/test if data/images doesn't exist
        alt_dir = os.path.join(config.DATA_DIR, "test")
        if os.path.isdir(alt_dir):
            image_dir = alt_dir
            logger.info(f"Using alternative image directory: {alt_dir}")
        else:
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

    files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in config.SUPPORTED_EXTENSIONS
    ])

    if limit:
        files = files[:limit]

    logger.info(f"Discovered {len(files)} images in {image_dir}")
    return files, image_dir


def run_captioning(
    image_dir: str,
    image_files: List[str],
    skip: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Stage 1: Generate BLIP-2 captions."""
    if skip and os.path.exists(config.CAPTIONS_PATH):
        logger.info("Loading existing captions (--skip-captions)")
        with open(config.CAPTIONS_PATH, "r") as f:
            return json.load(f)

    generator = CaptionGenerator()
    captions = generator.process_directory(
        image_dir=image_dir,
        output_path=config.CAPTIONS_PATH,
        limit=len(image_files) if image_files else None,
        resume=True,
        image_files=image_files,
    )
    return captions


def run_attribute_extraction(
    captions: Dict[str, Dict[str, str]],
) -> Dict[str, Dict]:
    """Stage 2: Extract structured attributes from captions."""
    logger.info("Extracting structured attributes...")
    extractor = AttributeExtractor()
    attributes = extractor.extract_from_captions(captions)

    # Save attributes
    attrs_path = os.path.join(config.INDEX_DIR, "attributes.json")
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    with open(attrs_path, "w") as f:
        json.dump(attributes, f, indent=2)

    # Log statistics
    env_counts = {}
    style_counts = {}
    garment_counts = {}
    for attrs in attributes.values():
        env = attrs.get("environment")
        if env:
            env_counts[env] = env_counts.get(env, 0) + 1
        style = attrs.get("style")
        if style:
            style_counts[style] = style_counts.get(style, 0) + 1
        for g in attrs.get("garments", []):
            garment_counts[g] = garment_counts.get(g, 0) + 1

    logger.info(f"Attribute extraction complete for {len(attributes)} images")
    logger.info(f"  Environments: {env_counts}")
    logger.info(f"  Styles: {style_counts}")
    logger.info(f"  Top garments: {dict(sorted(garment_counts.items(), key=lambda x: -x[1])[:10])}")

    return attributes


def run_embedding(
    image_dir: str,
    image_files: List[str],
    captions: Dict[str, Dict[str, str]],
) -> tuple:
    """Stage 3: Compute dual-backbone embeddings."""
    engine = EmbeddingEngine()

    # Check for cached embeddings
    cache_dir = os.path.join(config.INDEX_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fashion_cache = os.path.join(cache_dir, "fashion_embeddings.npy")
    siglip_cache = os.path.join(cache_dir, "siglip_embeddings.npy")
    caption_cache = os.path.join(cache_dir, "caption_embeddings.npy")

    if all(os.path.exists(f) for f in [fashion_cache, siglip_cache, caption_cache]):
        logger.info("Loading cached embeddings...")
        return (
            np.load(fashion_cache),
            np.load(siglip_cache),
            np.load(caption_cache),
        )

    logger.info("Computing embeddings (this may take a while)...")

    fashion_embs = []
    siglip_embs = []
    caption_embs = []

    for filename in tqdm(image_files, desc="Embedding images"):
        image_path = os.path.join(image_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")

            # Image embeddings from both backbones
            f_emb = engine.get_fashion_clip_image_embedding(image)
            s_emb = engine.get_siglip_image_embedding(image)

            # Caption embedding (SigLIP-2 text encoder on BLIP-2 caption)
            combined_caption = captions.get(filename, {}).get("combined", "")
            if combined_caption:
                c_emb = engine.get_siglip_text_embedding(combined_caption)
            else:
                c_emb = np.zeros(config.SIGLIP2_DIM, dtype=np.float32)

            fashion_embs.append(f_emb)
            siglip_embs.append(s_emb)
            caption_embs.append(c_emb)

        except Exception as e:
            logger.warning(f"Failed to embed {filename}: {e}")
            fashion_embs.append(np.zeros(config.FASHION_CLIP_DIM, dtype=np.float32))
            siglip_embs.append(np.zeros(config.SIGLIP2_DIM, dtype=np.float32))
            caption_embs.append(np.zeros(config.SIGLIP2_DIM, dtype=np.float32))

    fashion_embs = np.array(fashion_embs, dtype=np.float32)
    siglip_embs = np.array(siglip_embs, dtype=np.float32)
    caption_embs = np.array(caption_embs, dtype=np.float32)

    # Cache embeddings
    np.save(fashion_cache, fashion_embs)
    np.save(siglip_cache, siglip_embs)
    np.save(caption_cache, caption_embs)

    logger.info(
        f"Embeddings computed: fashion={fashion_embs.shape}, "
        f"siglip={siglip_embs.shape}, caption={caption_embs.shape}"
    )

    return fashion_embs, siglip_embs, caption_embs


def run_indexing(
    image_dir: str,
    image_files: List[str],
    captions: Dict[str, Dict[str, str]],
    attributes: Dict[str, Dict],
    fashion_embs: np.ndarray,
    siglip_embs: np.ndarray,
    caption_embs: np.ndarray,
):
    """Stage 4: Build FAISS indices and save everything."""
    logger.info("Building FAISS indices...")

    # Build metadata list (one entry per image, aligned with FAISS index positions)
    metadata_list = []
    for filename in image_files:
        metadata_list.append({
            "filename": filename,
            "path": os.path.join(image_dir, filename),
            "captions": captions.get(filename, {}),
            "attributes": attributes.get(filename, {}),
        })

    # Build and save vector store
    store = FashionVectorStore()
    store.build_index(fashion_embs, siglip_embs, caption_embs, metadata_list)
    store.save()

    logger.info(f"Indexing complete: {store.size} images indexed")
    return store


def main():
    parser = argparse.ArgumentParser(description="Fashion Image Indexer")
    parser.add_argument("--data-dir", default=config.IMAGE_DIR, help="Image directory")
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--skip-captions", action="store_true", help="Skip captioning, use existing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    start_time = time.time()

    # Stage 0: Discover images
    logger.info("=" * 60)
    logger.info("FASHION IMAGE INDEXER — Starting Pipeline")
    logger.info("=" * 60)

    image_files, image_dir = discover_images(args.data_dir, limit=args.limit)

    # Stage 1: Caption generation
    logger.info("\n--- Stage 1: BLIP-2 Captioning ---")
    captions = run_captioning(image_dir, image_files, skip=args.skip_captions)

    # Filter to only images we have
    captions = {k: v for k, v in captions.items() if k in image_files}

    # Stage 2: Attribute extraction
    logger.info("\n--- Stage 2: Attribute Extraction ---")
    attributes = run_attribute_extraction(captions)

    # Stage 3: Embedding computation
    logger.info("\n--- Stage 3: Dual-Backbone Embeddings ---")
    fashion_embs, siglip_embs, caption_embs = run_embedding(
        image_dir, image_files, captions
    )

    # Stage 4: Index building
    logger.info("\n--- Stage 4: FAISS Index Building ---")
    store = run_indexing(
        image_dir, image_files, captions, attributes,
        fashion_embs, siglip_embs, caption_embs,
    )

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"INDEXING COMPLETE")
    logger.info(f"  Images indexed: {store.size}")
    logger.info(f"  Time elapsed:   {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Index saved to: {config.INDEX_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
