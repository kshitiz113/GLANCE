"""
Vector Store — Triple FAISS Index + Metadata Store

Manages three separate FAISS indices for the three embedding types:
1. FashionCLIP image embeddings (512-d) — fashion-domain similarity
2. SigLIP-2 image embeddings (768-d)   — general visual similarity
3. SigLIP-2 caption embeddings (768-d) — semantic/textual similarity

Plus a JSON metadata store that maps each index position to:
- image filename and path
- BLIP-2 generated captions
- structured attributes with attribute-object bindings

Why three indices?
- Different queries benefit from different embedding spaces
- "bright yellow raincoat" → FashionCLIP excels (fashion domain)
- "sitting on a park bench" → SigLIP-2 image excels (scene understanding)
- "casual weekend outfit" → Caption embedding excels (semantic matching)
- Score-level fusion combines all three perspectives

FAISS Choice:
- IndexFlatIP for exact inner product search (cosine sim on normalized vectors)
- Swap to IndexIVFFlat or IndexHNSWFlat for 1M+ image scalability
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class FashionVectorStore:
    """
    Triple FAISS vector index with metadata store for fashion image retrieval.

    Stores and searches across three complementary embedding spaces,
    with metadata for attribute-based pre-filtering.
    """

    def __init__(self, index_dir: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            index_dir: Directory to store/load FAISS indices and metadata.
        """
        import faiss

        self.index_dir = index_dir or config.INDEX_DIR
        os.makedirs(self.index_dir, exist_ok=True)

        # FAISS indices — initialized empty, built during indexing
        self.fashion_index = None   # FashionCLIP (512-d)
        self.siglip_index = None    # SigLIP-2 image (768-d)
        self.caption_index = None   # SigLIP-2 caption (768-d)

        # Metadata: list indexed by position in FAISS index
        # Each entry: {filename, path, captions, attributes}
        self.metadata: List[Dict[str, Any]] = []

    def build_index(
        self,
        fashion_embeddings: np.ndarray,
        siglip_embeddings: np.ndarray,
        caption_embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ):
        """
        Build all three FAISS indices from pre-computed embeddings.

        Args:
            fashion_embeddings: (N, 512) FashionCLIP image embeddings
            siglip_embeddings:  (N, 768) SigLIP-2 image embeddings
            caption_embeddings: (N, 768) SigLIP-2 caption embeddings
            metadata_list: List of metadata dicts, one per image
        """
        import faiss

        assert len(fashion_embeddings) == len(siglip_embeddings) == len(caption_embeddings) == len(metadata_list), \
            "All inputs must have the same number of entries"

        n = len(metadata_list)
        logger.info(f"Building FAISS indices for {n} images...")

        # Ensure float32 for FAISS
        fashion_embeddings = fashion_embeddings.astype(np.float32)
        siglip_embeddings = siglip_embeddings.astype(np.float32)
        caption_embeddings = caption_embeddings.astype(np.float32)

        # Build indices using Inner Product (= cosine similarity on normalized vectors)
        self.fashion_index = faiss.IndexFlatIP(fashion_embeddings.shape[1])
        self.fashion_index.add(fashion_embeddings)

        self.siglip_index = faiss.IndexFlatIP(siglip_embeddings.shape[1])
        self.siglip_index.add(siglip_embeddings)

        self.caption_index = faiss.IndexFlatIP(caption_embeddings.shape[1])
        self.caption_index.add(caption_embeddings)

        self.metadata = metadata_list

        logger.info(
            f"Indices built: fashion={self.fashion_index.ntotal}, "
            f"siglip={self.siglip_index.ntotal}, "
            f"caption={self.caption_index.ntotal}"
        )

    def _check_indices_ready(self):
        """Verify that FAISS indices are built/loaded before searching."""
        if self.fashion_index is None or self.siglip_index is None or self.caption_index is None:
            raise RuntimeError(
                "FAISS vector indices are not loaded or built yet! "
                f"Please build the index first by running:\n"
                f"    python indexer/run_indexer.py --limit 50\n"
                f"or for full indexing across all images:\n"
                f"    python indexer/run_indexer.py\n"
                f"(No index files found in: {self.index_dir})"
            )

    def search_fashion(self, query_embedding: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Search the FashionCLIP index. Returns (scores, indices)."""
        self._check_indices_ready()
        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.fashion_index.search(query, top_k)
        return scores[0], indices[0]

    def search_siglip(self, query_embedding: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Search the SigLIP-2 image index. Returns (scores, indices)."""
        self._check_indices_ready()
        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.siglip_index.search(query, top_k)
        return scores[0], indices[0]

    def search_caption(self, query_embedding: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Search the SigLIP-2 caption index. Returns (scores, indices)."""
        self._check_indices_ready()
        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.caption_index.search(query, top_k)
        return scores[0], indices[0]

    def search_all(
        self,
        fashion_query: np.ndarray,
        siglip_query: np.ndarray,
        top_k: int = 100,
    ) -> Dict[int, Dict[str, float]]:
        """
        Search all three indices and collect scores per candidate.

        Note: For the caption index, we use the siglip_query since both
        captions and queries are encoded by SigLIP-2's text encoder.

        Args:
            fashion_query: FashionCLIP text embedding (512-d)
            siglip_query:  SigLIP-2 text embedding (768-d)
            top_k: Number of results per index

        Returns:
            Dict mapping index_id → {fashion_score, siglip_score, caption_score}
        """
        candidates = {}

        # Search each index
        f_scores, f_ids = self.search_fashion(fashion_query, top_k)
        s_scores, s_ids = self.search_siglip(siglip_query, top_k)
        c_scores, c_ids = self.search_caption(siglip_query, top_k)

        for score, idx in zip(f_scores, f_ids):
            if idx == -1:
                continue
            candidates.setdefault(int(idx), {}).update({"fashion_score": float(score)})

        for score, idx in zip(s_scores, s_ids):
            if idx == -1:
                continue
            candidates.setdefault(int(idx), {}).update({"siglip_score": float(score)})

        for score, idx in zip(c_scores, c_ids):
            if idx == -1:
                continue
            candidates.setdefault(int(idx), {}).update({"caption_score": float(score)})

        return candidates

    def get_metadata(self, index_id: int) -> Dict[str, Any]:
        """Get metadata for a specific index position."""
        if 0 <= index_id < len(self.metadata):
            return self.metadata[index_id]
        return {}

    def save(self):
        """Save all indices and metadata to disk."""
        import faiss

        logger.info(f"Saving indices to {self.index_dir}")

        if self.fashion_index:
            faiss.write_index(self.fashion_index, config.FAISS_FASHION_INDEX)
        if self.siglip_index:
            faiss.write_index(self.siglip_index, config.FAISS_SIGLIP_INDEX)
        if self.caption_index:
            faiss.write_index(self.caption_index, config.FAISS_CAPTION_INDEX)

        # Save metadata
        with open(config.METADATA_PATH, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved {len(self.metadata)} entries")

    def load(self):
        """Load all indices and metadata from disk."""
        import faiss

        logger.info(f"Loading indices from {self.index_dir}")

        if os.path.exists(config.FAISS_FASHION_INDEX):
            self.fashion_index = faiss.read_index(config.FAISS_FASHION_INDEX)
        if os.path.exists(config.FAISS_SIGLIP_INDEX):
            self.siglip_index = faiss.read_index(config.FAISS_SIGLIP_INDEX)
        if os.path.exists(config.FAISS_CAPTION_INDEX):
            self.caption_index = faiss.read_index(config.FAISS_CAPTION_INDEX)

        if os.path.exists(config.METADATA_PATH):
            with open(config.METADATA_PATH, "r") as f:
                self.metadata = json.load(f)

        logger.info(
            f"Loaded: fashion={self.fashion_index.ntotal if self.fashion_index else 0}, "
            f"siglip={self.siglip_index.ntotal if self.siglip_index else 0}, "
            f"caption={self.caption_index.ntotal if self.caption_index else 0}, "
            f"metadata={len(self.metadata)}"
        )

    @property
    def size(self) -> int:
        """Number of indexed images."""
        return len(self.metadata)
