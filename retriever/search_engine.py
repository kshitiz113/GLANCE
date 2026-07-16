"""
Search Engine — Hybrid Retrieval with Triple-Vector Fusion + Metadata Filtering

The core retrieval logic that combines three search signals:
1. FashionCLIP similarity (fashion-domain)
2. SigLIP-2 image similarity (general visual)
3. SigLIP-2 caption similarity (semantic/textual)

Plus optional metadata pre-filtering for attribute-specific queries.

Score Fusion:
    final_score = α·fashion_score + β·siglip_score + γ·caption_score + δ·attr_boost

Where attr_boost rewards candidates whose metadata attributes match
the query's structured constraints (including attribute-object bindings).

This hybrid approach outperforms any single-vector search because different
query types benefit from different embedding spaces.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from indexer.embedding_engine import EmbeddingEngine
from indexer.vector_store import FashionVectorStore
from retriever.query_decomposer import QueryDecomposer

logger = logging.getLogger(__name__)


class SearchEngine:
    """
    Hybrid search engine combining triple-vector similarity with metadata filtering.

    Pipeline:
    1. Decompose query → structured constraints
    2. Encode query with both FashionCLIP and SigLIP-2
    3. Search all 3 FAISS indices
    4. Fuse scores with configurable weights
    5. Apply metadata attribute boost
    6. Return ranked candidates
    """

    def __init__(
        self,
        vector_store: Optional[FashionVectorStore] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        self.store = vector_store or FashionVectorStore()
        self.engine = embedding_engine or EmbeddingEngine()
        self.decomposer = QueryDecomposer()

        # Load index if not already loaded
        if self.store.size == 0:
            self.store.load()

    def search(
        self,
        query: str,
        top_k: int = None,
        alpha: float = None,
        beta: float = None,
        gamma: float = None,
        use_metadata_boost: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Full hybrid search pipeline.

        Args:
            query: Natural language search query
            top_k: Number of results to return (before VQA re-ranking)
            alpha: FashionCLIP weight override
            beta: SigLIP-2 image weight override
            gamma: Caption embedding weight override
            use_metadata_boost: Whether to apply attribute matching bonus

        Returns:
            List of result dicts, sorted by score descending:
            [{index_id, filename, path, score, score_breakdown, metadata}, ...]
        """
        top_k = top_k or config.VQA_CANDIDATES
        alpha = alpha if alpha is not None else config.ALPHA
        beta = beta if beta is not None else config.BETA
        gamma = gamma if gamma is not None else config.GAMMA

        # Step 1: Decompose query
        decomposed = self.decomposer.decompose(query)
        logger.info(f"Query decomposed: {len(decomposed['constraints'])} constraints, "
                     f"env={decomposed['environment']}, style={decomposed['style']}")

        # Step 2: Encode query with both backbones
        fashion_query = self.engine.get_fashion_clip_text_embedding(query)
        siglip_query = self.engine.get_siglip_text_embedding(query)

        # Step 3: Search all three indices
        candidates = self.store.search_all(
            fashion_query=fashion_query,
            siglip_query=siglip_query,
            top_k=config.TOP_K_RETRIEVAL,
        )

        # Step 4: Fuse scores
        results = []
        for idx, scores in candidates.items():
            f_score = scores.get("fashion_score", 0.0)
            s_score = scores.get("siglip_score", 0.0)
            c_score = scores.get("caption_score", 0.0)

            fused_score = alpha * f_score + beta * s_score + gamma * c_score

            # Step 5: Metadata attribute boost
            attr_boost = 0.0
            if use_metadata_boost:
                metadata = self.store.get_metadata(idx)
                attr_boost = self._compute_attribute_boost(
                    decomposed, metadata.get("attributes", {})
                )
                fused_score += config.ATTR_BOOST * attr_boost

            metadata = self.store.get_metadata(idx)
            results.append({
                "index_id": idx,
                "filename": metadata.get("filename", ""),
                "path": metadata.get("path", ""),
                "score": fused_score,
                "score_breakdown": {
                    "fashion_clip": f_score,
                    "siglip_image": s_score,
                    "caption": c_score,
                    "attr_boost": attr_boost,
                    "fused": fused_score,
                },
                "metadata": metadata,
            })

        # Sort by fused score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def _compute_attribute_boost(
        self,
        decomposed: Dict[str, Any],
        image_attrs: Dict[str, Any],
    ) -> float:
        """
        Compute attribute matching bonus between query and image metadata.

        Checks:
        1. Garment overlap (query garments ∩ image garments)
        2. Color overlap
        3. Binding match (garment+color pair match — the compositionality check)
        4. Environment match
        5. Style match

        Returns a score between 0 and 1.
        """
        if not image_attrs:
            return 0.0

        total_checks = 0
        matches = 0

        # Garment overlap
        query_garments = set(decomposed.get("garments", []))
        image_garments = set(image_attrs.get("garments", []))
        if query_garments:
            total_checks += len(query_garments)
            matches += len(query_garments & image_garments)

        # Color overlap
        query_colors = set(decomposed.get("colors", []))
        image_colors = set(image_attrs.get("colors", []))
        if query_colors:
            total_checks += len(query_colors)
            matches += len(query_colors & image_colors)

        # Binding match (THE compositionality check)
        query_bindings = decomposed.get("constraints", [])
        image_bindings = image_attrs.get("bindings", [])
        if query_bindings:
            for qb in query_bindings:
                total_checks += 1
                q_garment = qb.get("garment")
                q_color = qb.get("color")
                for ib in image_bindings:
                    if (ib.get("garment") == q_garment and
                            ib.get("color") == q_color):
                        matches += 1
                        break

        # Environment match
        query_env = decomposed.get("environment")
        image_env = image_attrs.get("environment")
        if query_env:
            total_checks += 1
            if query_env == image_env:
                matches += 1

        # Style match
        query_style = decomposed.get("style")
        image_style = image_attrs.get("style")
        if query_style:
            total_checks += 1
            if query_style == image_style:
                matches += 1

        return matches / max(total_checks, 1)

    def search_baseline_clip(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Baseline: FashionCLIP-only search (for ablation comparison).

        This shows what vanilla CLIP-style retrieval produces,
        demonstrating why our hybrid approach is better.
        """
        return self.search(
            query=query,
            top_k=top_k,
            alpha=1.0, beta=0.0, gamma=0.0,
            use_metadata_boost=False,
        )

    def search_without_vqa(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search WITHOUT VQA re-ranking (for ablation).

        Shows the benefit of triple-vector fusion + metadata boost
        without the VQA stage.
        """
        return self.search(query=query, top_k=top_k)
